# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import os  # → File Directory
import tarfile  # → Extracting Datasets
from tqdm import tqdm  # → Progress Bar Visualization
from io import BytesIO  # → Working with in-memory data
from zipfile import ZipFile  # → Extracting downloaded zip
from urllib.request import urlopen  # → Downloading files

import torch  # → Core Torch Library
import torch.optim as optim  # → Torch Optimizers
from torchsummaryX import summary  # → Better model summary

# MLTU Utilities
from mltu.torch.model import Model  # → Base Training Wrapper
from mltu.torch.losses import CTCLoss  # → Implement CTC Loss
from mltu.torch.dataProvider import DataProvider  # → Batching, Augmentation, Preprocessing
from mltu.torch.metrics import CERMetric, WERMetric  # → Character Error Rate, Word Error Rate
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, ReduceLROnPlateau  # → Training callbacks

from mltu.preprocessors import ImageReader  # → Loads and Decodes Images
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2  # → Transformers
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen  # → Augmentors
from mltu.annotations.images import CVImage  # → OpenCV Image Manipulation

from model import Network  # → PyTorch Model
from configs import ModelConfigs  # → Training Hyperparameters


# -------------------------------------------------------------
# Dataset Downloader
# -------------------------------------------------------------
def download_and_unzip(url, extract_to="Datasets", chunk_size=1024 * 1024):
    """
    Using `urlopen` we download a zip file and extract it into the `Datasets` Folder.

    Checks if the IAM Words Dataset exists or not. 
    If not, downloads and extracts the raw `.tgz` file into the specified directory.
    """
    http_response = urlopen(url)

    data = b""
    iterations = http_response.length // chunk_size + 1
    for _ in tqdm(range(iterations)):
        data += http_response.read(chunk_size)

    zipfile = ZipFile(BytesIO(data))
    zipfile.extractall(path=extract_to)


# -------------------------------------------------------------
# Dataset Parsing
# -------------------------------------------------------------
dataset_path = os.path.join("..","Datasets", "IAM_Words")
if not os.path.exists(dataset_path):
    download_and_unzip("https://git.io/J0fjL", extract_to="Datasets")

    file = tarfile.open(os.path.join(dataset_path, "words.tgz"))
    file.extractall(os.path.join(dataset_path, "words"))

dataset, vocab, max_len = [], set(), 0

"""
`words.txt` contains metadata about each image → Filename and Corresponding Label

We skip erroneous samples and headers to then build the correct relative path to the image file to extract the actual words for training

This process collects:
- Full dataset paths and labels
- Set of unique characters
- Maximum label length for padding
"""
words = open(os.path.join(dataset_path, "words.txt"), "r").readlines()
for line in tqdm(words):
    if line.startswith("#"):
        continue

    line_split = line.split(" ")
    if line_split[1] == "err":
        continue

    folder1 = line_split[0][:3]
    folder2 = "-".join(line_split[0].split("-")[:2])
    file_name = line_split[0] + ".png"
    label = line_split[-1].rstrip("\n")

    rel_path = os.path.join(dataset_path, "words", folder1, folder2, file_name)
    if not os.path.exists(rel_path):
        print(f"File not found: {rel_path}")
        continue

    dataset.append([rel_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))


# -------------------------------------------------------------
# Config Save
# -------------------------------------------------------------
"""
`ModelConfigs` stores critical hyperparameters and architecture settings

It saves the complete character set and longest word length
"""
configs = ModelConfigs()
configs.vocab = "".join(sorted(vocab))
configs.max_text_length = max_len
configs.save()


# -------------------------------------------------------------
# Data Pipeline
# -------------------------------------------------------------
"""
Modular Pipeline for:

- Reading and Preprocessing Images → `DataReader`
- Resize Images to Uniform Dimensions
- Convert Labels to Index Tensors
- Pad Labels to Max Length
- 90/10 Train-Validation Split
- Apply randomized augmentations like brightness, rotation, erosion, sharpening to boost generalization
"""
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],  # type: ignore
    transformers=[
        # ImageShowCV2(),  # Uncomment to show images when iterating
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),  # type: ignore
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
    ],
    use_cache=True,
)

# Split into training and validation sets
train_dataProvider, test_dataProvider = data_provider.split(split=0.9)

# Training Data Augmentations
train_dataProvider.augmentors = [
    RandomBrightness(),
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotate(angle=10),
]


# -------------------------------------------------------------
# Model Setup
# -------------------------------------------------------------
"""
Initialize the CNN + LSTM model

Uses CTC Loss → Perfect for unaligned text sequences with a `blank` index for end of vocab in CTC

Adam Optimizer is used from the Learning Rate specified in the `configs`
"""
network = Network(len(configs.vocab), activation="leaky_relu", dropout=0.3)
loss = CTCLoss(blank=len(configs.vocab))
optimizer = optim.Adam(network.parameters(), lr=configs.learning_rate)

# Print network summary (optional)
# summary(network, torch.zeros((1, configs.height, configs.width, 3)))

# Move to CUDA if available
if torch.cuda.is_available():
    network = network.cuda()


# -------------------------------------------------------------
# Training Callbacks
# -------------------------------------------------------------
"""
`EarlyStopping` → Stop Training if CER doesn't improve

`ModelCheckpoint` → Save only the best model

`TensorBoard` → Logs Training Metrics

`ReduceLROnPlateau` → Decay LR if performance stagnates

`Model2onnx` → Export to ONNX Format for inference
"""
earlyStopping = EarlyStopping(monitor="val_CER", patience=20, mode="min", verbose=1)  # type: ignore
modelCheckpoint = ModelCheckpoint(configs.model_path + "/model.pt", monitor="val_CER", mode="min", save_best_only=True, verbose=1)  # type: ignore
tb_callback = TensorBoard(configs.model_path + "/logs")
reduce_lr = ReduceLROnPlateau(monitor="val_CER", factor=0.9, patience=10, verbose=1, mode="min", min_lr=1e-6)
model2onnx = Model2onnx(
    saved_model_path=configs.model_path + "/model.pt",
    input_shape=(1, configs.height, configs.width, 3),
    verbose=1,  # type: ignore
    metadata={"vocab": configs.vocab}
)


# -------------------------------------------------------------
# Model Training Loop
# -------------------------------------------------------------
"""
Wraps training logic for 1000 epochs with CER and WER as Performance Metrics

Saves Dataset Splits as CSVs for Inference
"""
model = Model(network, optimizer, loss, metrics=[CERMetric(configs.vocab), WERMetric(configs.vocab)])
model.fit(
    train_dataProvider,
    test_dataProvider,
    epochs=1000,
    callbacks=[earlyStopping, modelCheckpoint, tb_callback, reduce_lr, model2onnx]
)

# Save train/val split to CSV
train_dataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
test_dataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))
