import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.attention import ChannelAttention, SpatialAttention
from models.modules.conv_blocks import UpBlock, DoubleConv  # সঠিক পাথ

class HybridAttentionDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, num_classes):
        super().__init__()
        # Reverse the encoder channels to match the order of skip connections
        reversed_encoder_channels = encoder_channels[::-1]
        
        # Decoder blocks with attention
        self.up_blocks = nn.ModuleList()
        self.channel_attention = nn.ModuleList()
        self.spatial_attention = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        
        # Remaining upsampling blocks with dynamically sized attention modules
        for i in range(len(reversed_encoder_channels)):
            in_ch = decoder_channels[i]
            skip_ch = reversed_encoder_channels[i]
            out_ch = decoder_channels[i+1] if i+1 < len(decoder_channels) else decoder_channels[i]
            
            self.up_blocks.append(UpBlock(in_ch, out_ch))
            self.channel_attention.append(ChannelAttention(skip_ch))
            self.spatial_attention.append(SpatialAttention())
            self.conv_blocks.append(DoubleConv(out_ch + skip_ch, out_ch))
        
        # Final output
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, 1)
        
    def forward(self, x, skip_features):
        decoder_features = []
        
        # Loop over skip connections and upsampling blocks
        for i, (skip, up_block, ca, sa, conv_block) in enumerate(
            zip(skip_features, self.up_blocks, self.channel_attention, 
                self.spatial_attention, self.conv_blocks)
        ):
            # Upsample
            x = up_block(x)
            
            # Apply attention to skip connection
            skip_ca = ca(skip)  # Channel attention
            skip_sa = sa(skip)  # Spatial attention
            skip_attended = skip * skip_ca * skip_sa
            
            # Concatenate and apply convolution
            x = torch.cat([x, skip_attended], dim=1)
            x = conv_block(x)
            decoder_features.append(x)
        
        # Final output
        out = self.final_conv(x)
        
        return out, decoder_features
