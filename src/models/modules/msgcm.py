import torch
import torch.nn as nn
import torch.nn.functional as F

class MSGCM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Multi-scale pyramid pooling
        self.pyramid_pool = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, in_channels // reduction, 1),
                nn.ReLU(inplace=True)
            ) for pool_size in [1, 2, 4, 8]
        ])
        
        # Self-attention mechanism
        self.self_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // reduction * 4, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Multi-scale pyramid features
        pyramid_features = []
        for pool in self.pyramid_pool:
            pooled = pool(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            pyramid_features.append(upsampled)
        pyramid_out = torch.cat(pyramid_features, dim=1)
        
        # Self-attention
        attention = self.self_attention(x)
        attended = x * attention
        
        # Fusion
        out = self.fusion(torch.cat([attended, pyramid_out], dim=1))
        
        return out