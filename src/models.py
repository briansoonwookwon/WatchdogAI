import torch
import torch.nn as nn
from torchvision import models

class Simple3BlockCNN(nn.Module):
    def __init__(self):
        super(Simple3BlockCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # for RGB images
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # reduces spatial dimensions by 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # After 3 poolings: 224 / 2 / 2 / 2 = 28
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)  # Binary output
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.classifier(x)
        return x

class BNDropout3BlockCNN(nn.Module):
    """
    Simple CNN for binary classification.
    Input: (B, 3, 224, 224) → Output: (B, 1)
    """
    def __init__(self, num_classes: int = 1, dropout_rate: float = 0.5) -> None:
        super(BNDropout3BlockCNN, self).__init__()
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

class AIDetectorResNet(nn.Module):
    def __init__(self, freeze_backbone: bool = False, dropout_rate: float = 0.5) -> None:
        super(AIDetectorResNet, self).__init__()

        # Pretrained ResNet50
        self.backbone = models.resnet50(pretrained=True)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features

        # Replace classifier head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

        self._initialize_head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def _initialize_head(self) -> None:
        """Initialize the new classifier head layers."""
        for module in self.backbone.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)