import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------
#   Channel Attention Module
# ----------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 4)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze
        avg = self.avg_pool(x).view(b, c)
        maxv = self.max_pool(x).view(b, c)

        # Shared MLP
        avg_out = self.mlp(avg)
        max_out = self.mlp(maxv)

        # Excitation
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)

        return x * out


# ----------------------------------------------------
#   Spatial Attention Module
# ----------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        # Conv over concatenated [avgpool, maxpool] maps
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise statistics
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)

        merged = torch.cat([avg, maxv], dim=1)
        out = self.conv(merged)
        out = self.sigmoid(out)

        return x * out


# ----------------------------------------------------
#   CBAM = Channel Attention + Spatial Attention
# ----------------------------------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
