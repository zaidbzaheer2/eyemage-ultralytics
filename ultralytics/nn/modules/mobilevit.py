import torch
import torch.nn as nn

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, kernel_size=3, patch_size=2, mlp_ratio=2):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size

        self.local_rep = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True)
        )

        self.project_in = nn.Conv2d(dim, dim, kernel_size=1)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, mlp_ratio)
            for _ in range(depth)
        ])

        # Projection back to spatial
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

        # Fusion
        self.fusion = nn.Conv2d(2 * dim, dim, kernel_size=1)

    def unfolding(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.ph, self.ph).unfold(3, self.pw, self.pw)
        x = x.contiguous().view(B, C, -1, self.ph, self.pw)
        x = x.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        return x, H, W

    def folding(self, x, H, W):
        B, N, C = x.shape
        h, w = H // self.ph, W // self.pw
        x = x.view(B, h, w, self.ph, self.pw, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, C, H, W)

    def forward(self, x):
        local = self.local_rep(x)
        y = self.project_in(local)
        tokens, H, W = self.unfolding(y)

        for layer in self.layers:
            tokens = layer(tokens)

        y = self.folding(tokens, H, W)
        y = self.project_out(y)

        # Fuse local + global features
        out = self.fusion(torch.cat([local, y], dim=1))
        return out


# ---------------------------------------------
#  Simple Transformer Encoder Layer for MobileViT
# ---------------------------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.SiLU(inplace=True),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
