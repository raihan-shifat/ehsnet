import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.encoder import Encoder
from models.modules.msgcm import MSGCM  
from models.modules.decoder import HybridAttentionDecoder

class EHSNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        # Encoder
        self.encoder = Encoder(pretrained)
        
        # Get feature channels dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 384, 384)
            features = self.encoder(dummy_input)
            feature_channels = [f.shape[1] for f in features]
            last_feature_channels = features[-1].shape[1]
        
        # Print channel dimensions for debugging
        print(f"Encoder feature channels: {feature_channels}")
        print(f"Last feature channels: {last_feature_channels}")
        
        # Multi-Scale Global Context Module
        self.msgcm = MSGCM(in_channels=last_feature_channels)
        
        # Hybrid Attention Decoder
        encoder_channels = feature_channels[:-1]  # Remove the last feature
        
        # Adjust decoder channels to handle the MSGCM output
        decoder_channels = [last_feature_channels, 256, 128, 64, 32]
        
        print(f"Passing to decoder - encoder_channels: {encoder_channels}")
        print(f"Passing to decoder - decoder_channels: {decoder_channels}")
        
        self.decoder = HybridAttentionDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes
        )
        
        # Deep supervision heads
        self.ds_heads = nn.ModuleList([ 
            nn.Conv2d(ch, num_classes, 1) for ch in [256, 128, 64, 32]
        ])
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.shape[2:]
        # Encoder features
        features = self.encoder(x)
        
        # Global context
        global_features = self.msgcm(features[-1])
        
        # Decoder with deep supervision
        out, decoder_features = self.decoder(global_features, features[:-1][::-1])
        
        # Upsample the main output to input size
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        # Deep supervision outputs
        ds_outs = []
        for i, feature in enumerate(decoder_features):
            ds_out = self.ds_heads[i](feature)
            ds_out = F.interpolate(ds_out, size=input_size, mode='bilinear', align_corners=False)
            ds_outs.append(ds_out)
        
        if self.training:
            return [out] + ds_outs
        else:
            return out
