import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class AggressionClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super().__init__()
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # Expected input shape: (B, C, T, H, W) -> (Batch, 3, 16, 112, 112)
        return self.backbone(x)