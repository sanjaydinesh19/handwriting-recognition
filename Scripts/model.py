import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_layer(activation: str = "relu", alpha: float = 0.1, inplace: bool = True):
    """
    Activation layer to switch between LeakyReLU and ReLU activation functions.

    Args:
        activation: str, activation function name (default: 'relu')
        alpha: float, LeakyReLU activation function parameter
        inplace: bool, whether to modify the input tensor in-place

    Returns:
        torch.nn.Module: selected activation layer

    Rather than hardcoding activation layers, this function allows switching between functions easily.
    `alpha` refers to the slope of the negative value functions in LeakyReLU.
    `inplace` tells PyTorch to modify the tensor directly to save memory.
    """
    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    elif activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=alpha, inplace=inplace)


class ConvBlock(nn.Module):
    """
    Convolutional block with batch normalization.

    Convolution → Extracts spatial features using filters

    Batch Normalization → Normalizes the feature maps to speed up training and improve stability

    Forward Pass applies the Convolution first on torch.Tensor x and then the Batch Normalization.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    """
    Custom Residual Block inspired by ResNet.

    Allows the input to skip through the block which helps in:
    - Gradient Flow
    - Better Convergence
    - Feature Preservation

    Internal Layers:
        - ConvBlock → Activation → ConvBlock → Dropout (Main path)
        - Optional 1x1 convolution (Shortcut path) if shape mismatch due to stride or channel change
        - Second Activation and Dropout after skip addition

    Forward Pass:
        - Apply a Conv → Activation → Conv layer
        - If shortcut exists, align and add the shortcut to the tensor
        - Apply second Activation → Dropout layer
    """
    def __init__(self, in_channels, out_channels, skip_conv=True, stride=1, dropout=0.2, activation="leaky_relu"):
        super(ResidualBlock, self).__init__()

        self.convb1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.act1 = activation_layer(activation)

        self.convb2 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(p=dropout)

        # Optional skip connection transformation
        self.shortcut = None
        if skip_conv:
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        self.act2 = activation_layer(activation)

    def forward(self, x):
        skip = x

        out = self.act1(self.convb1(x))
        out = self.convb2(out)

        if self.shortcut is not None:
            out += self.shortcut(skip)

        out = self.act2(out)
        out = self.dropout(out)

        return out


class Network(nn.Module):
    """
    Handwriting recognition network for CTC loss.

    This network is specifically designed for CTC-based handwriting recognition where 
    CTC (Connectionist Temporal Classification) is used when the input-output alignment 
    is unknown—ideal for image-to-sequence tasks like handwriting or speech.

    Key Architectural Notes:
        - `stride = 2` in some blocks means that the spatial resolution is gradually reduced,
          while channel depth increases → like a Feature Pyramid
        - Activations and Dropouts are consistent across blocks
        - 9 Residual Blocks for hierarchical feature extraction
        - Bi-directional LSTM (128 hidden units per direction) models sequence context
        - Linear layer maps LSTM output to character logits
        - Final activation: log_softmax for CTC Loss

    Forward Pass:
        - Normalize input image pixel values to [0,1]
        - Change shape from BHWC to BCHW for PyTorch
        - Pass through 9 Residual Blocks
        - Flatten into sequential format and process through Bi-LSTM
        - Output log-softmax probabilities over characters
    """
    def __init__(self, num_chars: int, activation: str = "leaky_relu", dropout: float = 0.2):
        super(Network, self).__init__()

        # Residual Blocks (RB1 to RB9) for feature extraction
        self.rb1 = ResidualBlock(3, 16, skip_conv=True, stride=1, activation=activation, dropout=dropout)
        self.rb2 = ResidualBlock(16, 16, skip_conv=True, stride=2, activation=activation, dropout=dropout)
        self.rb3 = ResidualBlock(16, 16, skip_conv=False, stride=1, activation=activation, dropout=dropout)

        self.rb4 = ResidualBlock(16, 32, skip_conv=True, stride=2, activation=activation, dropout=dropout)
        self.rb5 = ResidualBlock(32, 32, skip_conv=False, stride=1, activation=activation, dropout=dropout)

        self.rb6 = ResidualBlock(32, 64, skip_conv=True, stride=2, activation=activation, dropout=dropout)
        self.rb7 = ResidualBlock(64, 64, skip_conv=True, stride=1, activation=activation, dropout=dropout)

        self.rb8 = ResidualBlock(64, 64, skip_conv=False, stride=1, activation=activation, dropout=dropout)
        self.rb9 = ResidualBlock(64, 64, skip_conv=False, stride=1, activation=activation, dropout=dropout)

        # LSTM Layer to model temporal dependencies in feature sequence
        self.lstm = nn.LSTM(64, 128, bidirectional=True, num_layers=1, batch_first=True)
        self.lstm_dropout = nn.Dropout(p=dropout)

        # Linear output layer maps BiLSTM output (256) to number of classes (vocab + CTC blank)
        self.output = nn.Linear(256, num_chars + 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Normalize images between 0 and 1
        images_float = images / 255.0

        # Change image format from NHWC to NCHW for PyTorch
        images_float = images_float.permute(0, 3, 1, 2)

        # Apply 9 Residual Blocks sequentially
        x = self.rb1(images_float)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.rb6(x)
        x = self.rb7(x)
        x = self.rb8(x)
        x = self.rb9(x)

        # Flatten CNN output to sequence: (batch, time_steps, features)
        x = x.reshape(x.size(0), -1, x.size(1))

        # BiLSTM processes the sequential features
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)

        # Final character class logits (with log_softmax for CTC Loss)
        x = self.output(x)
        x = F.log_softmax(x, dim=2)

        return x
