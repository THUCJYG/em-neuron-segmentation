"""Small U-Net. Two output logits describe right/down nearest-neighbor edges."""
from math import gcd

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    """Group normalization works with one crop per optimization step."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = gcd(out_channels, 4)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )
        self.skip = (nn.Identity() if in_channels == out_channels else
                     nn.Conv2d(in_channels, out_channels, 1, bias=False))

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class SmallUNet(nn.Module):
    def __init__(self, base_channels=8, architecture="small_unet", auxiliary=False, long_channels=0):
        super().__init__()
        c = base_channels
        block = {"small_unet": DoubleConv, "residual_unet": ResidualBlock}[architecture]
        self.enc1 = block(1, c)
        self.enc2 = block(c, c * 2)
        self.enc3 = block(c * 2, c * 4)
        self.bottom = block(c * 4, c * 8)
        self.dec3 = block(c * 12, c * 4)
        self.dec2 = block(c * 6, c * 2)
        self.dec1 = block(c * 3, c)
        self.head = nn.Conv2d(c, 2, 1)
        self.coarse_boundary = nn.Conv2d(c * 4, 1, 1) if auxiliary else None
        self.long_affinity = nn.Conv2d(c, long_channels, 1) if long_channels else None
        self.pool = nn.MaxPool2d(2)

    @staticmethod
    def join(low, skip):
        return torch.cat((F.interpolate(low, size=skip.shape[-2:], mode="nearest"), skip), 1)

    def forward(self, x, with_auxiliary=False, with_long=False):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        d = self.bottom(self.pool(c))
        coarse = self.dec3(self.join(d, c))
        x = self.dec2(self.join(coarse, b))
        x = self.dec1(self.join(x, a))
        logits = self.head(x)
        if with_long:
            coarse_logits = self.coarse_boundary(coarse) if with_auxiliary else None
            return logits, coarse_logits, self.long_affinity(x)
        if with_auxiliary:
            return logits, self.coarse_boundary(coarse)
        return logits
