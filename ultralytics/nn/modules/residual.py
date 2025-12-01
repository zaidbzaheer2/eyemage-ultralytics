# Option A: Ghost Module Integration
from ultralytics import nn
import torch

class GhostLightResBlock(nn.Module):
    def __init__(self, c, ratio=2):
        super().__init__()
        hidden_c = c // ratio
        self.primary_conv = nn.Conv2d(c, hidden_c, 1, 1, 0, bias=False)
        self.cheap_operation = nn.Conv2d(hidden_c, hidden_c, 3, 1, 1, groups=hidden_c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c // 2, 1, 1, 0, bias=False)
        self.pw_expand = nn.Conv2d(c // 2, c, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        # Ghost features
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        ghost = torch.cat([x1, x2], dim=1)

        y = self.act(self.bn1(ghost))
        y = self.dw(y)
        y = self.act(self.pw(y))
        y = self.pw_expand(y)
        return self.act(self.bn2(y) + x)


# Option B: Inverted Residual with Squeeze-Excitation
class MobileViTLightBlock(nn.Module):
    def __init__(self, c, expansion=2, se_ratio=0.25):
        super().__init__()
        hidden_c = c * expansion
        se_c = max(1, int(c * se_ratio))

        self.pw_expand = nn.Conv2d(c, hidden_c, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_c)
        self.dw = nn.Conv2d(hidden_c, hidden_c, 3, 1, 1, groups=hidden_c, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_c)

        # Squeeze-and-Excitation
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Conv2d(hidden_c, se_c, 1, 1, 0, bias=True)
        self.se_fc2 = nn.Conv2d(se_c, hidden_c, 1, 1, 0, bias=True)

        self.pw_project = nn.Conv2d(hidden_c, c, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.act(self.bn1(self.pw_expand(x)))
        y = self.act(self.bn2(self.dw(y)))

        # SE attention
        se = self.se_pool(y)
        se = self.act(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        y = y * se

        y = self.bn3(self.pw_project(y))
        return x + y


# Option C: Ultra-Lightweight with Shared Weights
class UltraLightResBlock(nn.Module):
    def __init__(self, c, compress_ratio=4):
        super().__init__()
        mid_c = c // compress_ratio
        self.pw1 = nn.Conv2d(c, mid_c, 1, 1, 0, bias=False)
        self.dw = nn.Conv2d(mid_c, mid_c, 3, 1, 1, groups=mid_c, bias=False)
        self.pw2 = nn.Conv2d(mid_c, c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.act(self.pw1(x))
        y = self.act(self.dw(y))
        y = self.pw2(y)
        return self.act(self.bn(y + x))


class DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, s, k // 2, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))