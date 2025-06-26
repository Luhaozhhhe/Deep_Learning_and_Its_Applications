# model/Hor_gMLPBlock.py

import torch
import torch.nn as nn
from model.CNN_HorNet import gnconv
from model.CNN_gMLP import gMLPBlock  

class Hor_gMLPBlock(nn.Module):
    """
    先gMLP的token-mixing（即SpatialGatingUnit/FeedForward残差）
    再HorNet gnConv空间门控，两者都用残差连接
    """
    def __init__(self, dim, seq_len, mlp_ratio=4, gn_order=5, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.gmlp_block = gMLPBlock(
            dim=dim, dim_ff=dim*mlp_ratio, seq_len=seq_len
        )
        self.norm2 = nn.LayerNorm(dim)
        self.gn_conv = gnconv(dim, order=gn_order)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.seq_len = seq_len
        self.hw = int(seq_len ** 0.5)
        assert self.hw * self.hw == seq_len, "seq_len必须是平方数"

    def forward(self, x):
        # x: [B, N, C]
        # gMLP token-mixing路径
        x1 = x + self.drop_path(self.gmlp_block(self.norm1(x)))
        # HorNet gnConv空间门控路径
        B, N, C = x1.shape
        x_img = x1.transpose(1, 2).contiguous().view(B, C, self.hw, self.hw)  # [B, C, H, W]
        gn_out = self.gn_conv(x_img)                     # [B, C, H, W]
        gn_out = gn_out.flatten(2).transpose(1, 2)       # [B, N, C]
        x2 = x1 + self.drop_path(self.norm2(gn_out))
        return x2

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
class Hor_gMLPNet(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_chans=3, num_classes=100,
                 dim=64, depth=12, mlp_ratio=4, gn_order=5, drop_path=0.1):
        super().__init__()
        # Patch Embedding
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2

        # 多层Hor_gMLPBlock
        self.layers = nn.ModuleList([
            Hor_gMLPBlock(dim=dim, seq_len=num_patches, mlp_ratio=mlp_ratio, gn_order=gn_order, drop_path=drop_path)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.patch_embed(x)                     # [B, dim, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)           # [B, N, C]
        for blk in self.layers:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)                          # Global Average Pooling
        x = self.head(x)
        return x
