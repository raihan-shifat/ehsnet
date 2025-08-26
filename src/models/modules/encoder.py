import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        self.efficientnet = efficientnet_b4(weights=weights).features
        
        # Extract feature maps at different scales
        self.feature_indices = [0,2,3,5,6]  # Indices for EfficientNet-B4 stages to get 5 features
        
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.efficientnet):
            x = layer(x)
            if i in self.feature_indices:
                features.append(x)
        return features