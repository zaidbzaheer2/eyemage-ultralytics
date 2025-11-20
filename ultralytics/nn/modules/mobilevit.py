import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, mlp_ratio=2, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.SiLU(inplace=True),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        # x: (B, N, C)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """
    MobileViTBlock with robust dynamic shape handling.
    - Accepts any H, W
    - Pads internally to multiples of patch_size, then crops back
    - Keeps channel dims and spatial dims identical on output
    """
    def __init__(self, dim, depth=2, kernel_size=3, patch_size=2, mlp_ratio=2, num_heads=4):
        super().__init__()
        assert patch_size >= 1 and isinstance(patch_size, int)
        self.ph = patch_size
        self.pw = patch_size
        self.dim = dim

        # Local conv representation
        self.local_rep = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

        # Project to token channels (1x1 conv)
        self.project_in = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # Transformer layers
        self.layers = nn.ModuleList([TransformerEncoderLayer(dim, mlp_ratio, num_heads) for _ in range(depth)])

        # Project back to spatial
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # Fuse local + global features
        self.fusion = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=False)

    def _pad_to_multiple(self, x, ph, pw):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        pad_h = (ph - (H % ph)) % ph
        pad_w = (pw - (W % pw)) % pw
        # pad order: (left, right, top, bottom)
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, 0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h))  # pad right then bottom
        return x, (0, pad_w, 0, pad_h)

    def _remove_padding(self, x, pad):
        # x: (B, C, Hp, Wp)
        if pad == (0, 0, 0, 0):
            return x
        _, _, Hp, Wp = x.shape
        left, right, top, bottom = pad
        H = Hp - top - bottom
        W = Wp - left - right
        return x[..., :H, :W]

    def unfolding(self, x):
        # x assumed padded so H%ph==0 and W%pw==0
        B, C, H, W = x.shape
        ph, pw = self.ph, self.pw
        h_blocks = H // ph
        w_blocks = W // pw
        # reshape to (B, h_blocks, ph, w_blocks, pw, C) then to tokens
        x = x.view(B, C, h_blocks, ph, w_blocks, pw)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # B, hb, wb, ph, pw, C
        tokens = x.view(B, h_blocks * w_blocks, ph * pw * C)  # B, N, ph*pw*C
        # project token dimension back to dim via linear-like mapping: we'll first reshape differently:
        # We want tokens shape (B, N, C) (channel preserved), so instead use patch flatten across ph,pw then reshape to C
        # But simpler: our project_in ensures channels are dim, so here we produce (B, N, C) by:
        tokens = tokens.view(B, h_blocks * w_blocks, ph * pw, C).mean(2)  # average within patch -> (B, N, C)
        return tokens, H, W, h_blocks, w_blocks

    def folding(self, tokens, H, W, h_blocks, w_blocks):
        # tokens: (B, N, C) with N = h_blocks * w_blocks
        B, N, C = tokens.shape
        # reshape back to B, hb, wb, C
        x = tokens.view(B, h_blocks, w_blocks, C)
        # expand each token to ph x pw by repeating (this approximates reconstruction)
        ph, pw = self.ph, self.pw
        x = x.unsqueeze(-1).unsqueeze(-1)  # B, hb, wb, C, 1, 1
        x = x.repeat(1, 1, 1, 1, ph, pw)   # B, hb, wb, C, ph, pw
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # B, C, hb, ph, wb, pw
        x = x.view(B, C, H, W)  # B, C, H, W
        return x

    def forward(self, x):
        """
        x: B, C, H, W
        returns: B, C, H, W (identical spatial shape)
        """
        # 1) local representation
        local = self.local_rep(x)

        # 2) project to dim
        y = self.project_in(local)

        # 3) pad to multiples of patch_size
        y_padded, pad = self._pad_to_multiple(y, self.ph, self.pw)
        B, C, Hp, Wp = y_padded.shape

        # 4) unfold into tokens (B, N, C)
        tokens, H_orig, W_orig, hb, wb = self.unfolding(y_padded)

        # 5) normalize tokens channel-wise for transformer
        # tokens shape is (B, N, C) already
        for layer in self.layers:
            tokens = layer(tokens)

        # 6) folding back
        y_f = self.folding(tokens, Hp, Wp, hb, wb)

        # 7) project out and remove padding
        y_out = self.project_out(y_f)
        y_out = self._remove_padding(y_out, pad)  # crop back to original HxW

        # 8) fuse local and global (local uses original x's shape)
        fused = self.fusion(torch.cat([local, y_out], dim=1))
        return fused
