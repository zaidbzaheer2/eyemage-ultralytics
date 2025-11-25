import torch
import torch.nn as nn


class SimAM(nn.Module):
    """SimAM attention module with improved stability"""

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        n, c, h, w = x.size()
        # Compute mean and variance with numerical stability
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        x_var = ((x - x_mean) ** 2).mean(dim=[2, 3], keepdim=True)

        # Add epsilon for stability
        energy = self.alpha * (x - x_mean) ** 2 / (x_var + self.e_lambda)

        # Apply attention
        out = x * torch.sigmoid(energy)
        return out