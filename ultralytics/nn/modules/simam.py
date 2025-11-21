import torch
import torch.nn as nn

class SimAM(nn.Module):
    """
    Zero-param attention (SimAM). Safe to add into heads.
    Reference: "SimAM: A Simple, Parameter-free Attention Module for CNNs" (adapted)
    """
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x: B,C,H,W
        b, c, h, w = x.shape
        # mean over spatial dims per channel
        mu = x.mean(dim=(2,3), keepdim=True)                           # B,C,1,1
        # variance term over spatial dims (unbiased-ish)
        var = ((x - mu)**2).sum(dim=(2,3), keepdim=True) / (h * w - 1 + 1e-12)  # B,C,1,1
        # energy term
        e = (x - mu)**2 / (4.0 * (var + self.e_lambda)) + 0.5
        return x * torch.sigmoid(e)
