import torch
import torch.nn as nn

class BinaryCNN(nn.Module):
    """
    Simple CNN for binary classification.
    Input: (B, 3, 224, 224) → Output: (B, 1)
    """
    def __init__(self, num_classes: int = 1, dropout_rate: float = 0.5) -> None:
        super(BinaryCNN, self).__init__()
        self.features = nn.Sequential(
            # conv block 1
            nn.Conv2d(in_channels=3,    out_channels=32,  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv block 2
            nn.Conv2d(in_channels=32,   out_channels=64,  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv block 3
            nn.Conv2d(in_channels=64,   out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # after 3×2×2×2 pooling: spatial dims = 224→112→56→28
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def _initialize_weights(self) -> None:
        """He/Kaiming init for conv, Xavier for fc, BN weights →1, biases →0."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)