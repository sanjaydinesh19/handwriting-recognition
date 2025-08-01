import cv2
import numpy as np
import typing

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    """
    A custom wrapper over the base OnnxInferenceModel for doing image-to-text inference
    using a CTC-decoded ONNX model.
    
    Passes ONNX model path and other metadata from the constructor (model_path=...).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, image: np.ndarray) -> str:
        """
        Takes an input image and outputs the decoded text string prediction.

        Uses OpenCV to resize image to expected model input shape, reverses H,W to W,H for OpenCV.

        Also adds a batch dimension as 1 and casts the image as `Float32`.

        It then runs the ONNX Model Inference using `self.model.run()` to execute the forward pass.

        This forward pass gives raw logits output which is decoded using the `ctc_decoder`.

        Uses the vocabulary to convert the predicted indices to readable characters.
        """
        # Resize image to the model's expected input size
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        # Add batch dimension and cast to float32
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        # Run ONNX model and get predictions
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        # Decode the output using CTC and return text
        text = ctc_decoder(preds, self.metadata["vocab"])[0]  # type: ignore
        return text


if __name__ == "__main__":
    """
    Loops through validated images, predicts text and saves as a paragraph

    1. Loads ONNX Model
    2. Loads the validation data (path + Label) from CSV
    3. Reads Image from Disk and fixes path separators
    4. Handles missing or unreadable images
    5. Uses the model to predict text and stores the predictions for paragraph generation
    6. The predictions are joined into a single paragraph and written into `predictions.txt`
    """
    import pandas as pd
    import csv
    from tqdm import tqdm

    # Load ONNX model
    model = ImageToWordModel(model_path=r"C:\Sanjay\College\Research\Handwriting Recognition\Models\pytorch_recognition\202507022235\model.onnx")

    # Load validation CSV data [image_path, label]
    df = pd.read_csv(r"C:\Sanjay\College\Research\Handwriting Recognition\Models\pytorch_recognition\202507022235\val.csv").values.tolist()

    predicted_words = []

    # Loop through images and predict
    for image_path, label in tqdm(df):
        # Ensure correct file path formatting
        image = cv2.imread(image_path.replace("\\", "/"))

        # Skip if image can't be loaded
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Predict text and append to list
        prediction_text = model.predict(image)
        predicted_words.append(prediction_text)

    # Convert predictions to a single paragraph
    paragraph = " ".join(predicted_words)

    # Save to text file
    with open("predictions.txt", "w", encoding="utf-8") as f:
        f.write(paragraph)

    print("Saved all predictions to predictions.txt.")
