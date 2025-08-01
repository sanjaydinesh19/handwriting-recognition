import os
from datetime import datetime
from mltu.configs import BaseModelConfigs

class ModelConfigs(BaseModelConfigs):
    """
    Custom configuration class for PyTorch-based handwriting recognition model.
    
    Inherits from BaseModelConfigs to include standard fields and adds 
    specific parameters for training and inference.
    """
    def __init__(self):
        super().__init__()

        # Path to store model files, organized by timestamp
        self.model_path = os.path.join(
            "Models/pytorch_recognition",
            datetime.strftime(datetime.now(), "%Y%m%d%H%M")
        )

        # Vocabulary will be filled after scanning dataset labels
        self.vocab = ""

        # Input image dimensions expected by the model
        self.height = 32
        self.width = 128

        # Maximum label length to pad the outputs for training
        self.max_text_length = 0

        # Training batch size
        self.batch_size = 64

        # Learning rate for optimizer
        self.learning_rate = 0.002

        # Number of training epochs
        self.train_epochs = 1000
