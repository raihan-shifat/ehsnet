import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.actf = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.actf(self.bn(self.conv(x)))

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
    def forward(self, x):
        return self.up(x)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3,1,1),
            ConvBNAct(out_ch, out_ch, 3,1,1))
    def forward(self, x):
        return self.block(x)
