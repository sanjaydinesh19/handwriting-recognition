import os
import cv2
import numpy as np
from tqdm import tqdm

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder


class ImageToWordModel(OnnxInferenceModel):
    """
    A custom wrapper over the base OnnxInferenceModel for doing image-to-text inference
    using a CTC-decoded ONNX model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, image: np.ndarray) -> str:
        """
        Takes an input image and outputs the decoded text string prediction.

        Uses OpenCV to resize image to expected model input shape, adds batch dimension,
        casts to float32, and performs ONNX inference. Output is decoded via CTC.
        """
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        text = ctc_decoder(preds, self.metadata["vocab"])[0]  # type: ignore
        return text


if __name__ == "__main__":
    """
    Predicts handwritten words from 1.png to 119.png stored in a directory
    and saves them as a paragraph into a .txt file.

    Steps:
    1. Load ONNX model from disk
    2. Loop through 1.png to 119.png
    3. Read each image and predict using the model
    4. Save all predictions in one paragraph to `test_predictions.txt`
    """
    # Path to model and test images
    model_path = r"C:\Sanjay\College\Research\Handwriting Recognition\Scripts\Models\pytorch_recognition\202508032158\model.onnx"
    test_image_dir = r"C:\Sanjay\College\Research\Handwriting Recognition\Datasets\Test"

    # Load model
    model = ImageToWordModel(model_path=model_path)

    predicted_words = []

    # Loop over all numbered test images (1.png to 119.png)
    for i in tqdm(range(1, 120)):
        image_path = os.path.join(test_image_dir, f"{i}.png")

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue

        predicted_word = model.predict(image)
        predicted_words.append(predicted_word)

    # Join into paragraph
    paragraph = " ".join(predicted_words)

    # Write to text file
    with open("test_predictions.txt", "w", encoding="utf-8") as f:
        f.write(paragraph)

    print("✅ Saved predictions to test_predictions.txt.")
