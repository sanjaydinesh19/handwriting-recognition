# Handwriting Recognition using CNN + BiLSTM + CTC (PyTorch)

This project implements an end-to-end handwriting recognition pipeline trained on the [I-Am-Words Dataset]. It uses a **CNN + Bidirectional LSTM + CTC Loss** architecture built in PyTorch, and supports inference via **ONNX** export.

---

## 🧠 Architecture Overview

The model consists of:
- ✅ **9 Residual CNN Blocks**  
  For learning deep spatial representations with efficient gradient flow
- ✅ **Bidirectional LSTM (BiLSTM)**  
  For capturing temporal dependencies along the image width (word sequences)
- ✅ **CTC Loss (Connectionist Temporal Classification)**  
  For training without explicit alignment between characters and image pixels
- ✅ **ONNX Export**  
  For lightweight, hardware-independent inference

---

## ⚙️ How It Works

1. **Data Preparation**  
   `trainTorch.py`:
   - Downloads the IAM dataset and extracts it
   - Builds label vocabulary and image-label pairs
   - Splits into train/val sets (90/10)
   - Applies preprocessing & augmentations

2. **Model Training**  
   - Trains the model using PyTorch with:
     - `CTCLoss`
     - `CER` and `WER` metrics
     - `EarlyStopping`, `ModelCheckpoint`, `TensorBoard`, and ONNX export

3. **Inference**  
   `inferenceModel.py`:
   - Loads exported `.onnx` model
   - Runs predictions on `val.csv` images
   - Outputs decoded words as a paragraph in `predictions.txt`

---

## Project Structure

```
├── Scripts/                       # All model, training, and inference scripts
│   ├── model.py                  # Model architecture (CNN + BiLSTM)
│   ├── configs.py                # Training hyperparameters & model metadata
│   ├── trainTorch.py            # Dataset download, preprocessing, training
│   ├── inferenceModel.py        # ONNX model loader & image-to-text inference
│   ├── inferenceModelTestImage.py  # Inference on manually labeled word images
│   └── comparePredictions.py    # Evaluate predictions vs ground truth (CER/WER)
├── Datasets/                     # IAM_Words dataset (auto-downloaded)
│   └── Test/                     # Optional test images (e.g., 1.png to N.png)
├── Models/                       # Saved PyTorch & ONNX models (timestamped)
├── predictions.txt              # Paragraph output of inference
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Train the Model
```bash
python Scripts/trainTorch.py
```
 - This will:
    - Download the IAM dataset
    - Train the CNN+BiLSTM+CTC model
    - Save checkpoints and export the best model to ONNX

### 3. Run Inference
```bash
python Scripts/inferenceModel.py
```
 - This will:
    - Load images from val.csv
    - Predict text using ONNX model
    - Save results to predictions.txt

## Dependencies
1. PyTorch

2. OpenCV

3. ONNX Runtime

4. MLTU 
```bash
pip install mltu
```

5. tqdm, numpy, pandas

## Acknowledgements

- IAM Dataset
- MLTU Library for Streamlined Model Training/Inference

