import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Convolutional block with convolution, ReLU, and batch normalization."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class StandardCNN(nn.Module):
    """Standard CNN classifier for facial detection."""

    def __init__(self, in_channels, img_h, img_w, n_filters, hidden_dim, n_outputs=1):
        super().__init__()

        # CNN Layers to extract features
        self.features = nn.Sequential(
            ConvBlock(in_channels, n_filters, kernel_size=5, stride=2, padding=2),
            ConvBlock(n_filters, 2 * n_filters, kernel_size=5, stride=2, padding=2),
            ConvBlock(2 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1),
            ConvBlock(4 * n_filters, 6 * n_filters, kernel_size=3, stride=2, padding=1),
        )

        # Calculate flattened feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, img_h, img_w)
            dummy_output = self.features(dummy_input)
            feature_size = dummy_output.numel()

        # Classifier to convert features to logits
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict(self, x):
        """Predict classification logits for given input x."""
        return self.forward(x)

    def get_device(self):
        """Get the device this model is on."""
        return next(self.parameters()).device
