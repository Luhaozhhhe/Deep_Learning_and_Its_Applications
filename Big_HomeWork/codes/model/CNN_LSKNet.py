import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)

class LSKAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Large-small kernel attention
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=3, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        # Channel attention
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_after_concat = nn.Conv2d(dim//2, dim, 1)
        # Add normalization layers
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        identity = x
        
        # First attention path
        attn1 = self.conv0(x)
        attn1 = self.norm1(attn1)
        attn1 = self.conv1(attn1)

        # Second attention path
        attn2 = self.conv_spatial(x)
        attn2 = self.norm2(attn2)
        attn2 = self.conv2(attn2)

        # Combine attentions
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        attention = torch.cat([avg_attn, max_attn], dim=1)
        attention = self.conv_squeeze(attention).sigmoid()

        # Apply attention
        out = attn1 * attention[:,0,:,:].unsqueeze(1) + \
              attn2 * attention[:,1,:,:].unsqueeze(1)
        out = self.conv_after_concat(out)

        # Residual connection
        return identity + out

class LSKBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0.):
        super().__init__()
        # Attention
        self.attn = LSKAttention(dim)
        # MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_ratio, 1),
            nn.GELU(),
            DWConv(dim * mlp_ratio),
            nn.Conv2d(dim * mlp_ratio, dim, 1)
        )
        # Layer Norm
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        # Drop Path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class LSKNet(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 num_classes=100,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 depths=[2, 2, 6, 2],
                 drop_path_rate=0.1):
        super().__init__()
        
        # Stem layer - reduced stride for CIFAR100
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], 3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU()
        )
        
        # Drop path rate increasing progressively
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[LSKBlock(
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop_path=dpr[cur + j]
                ) for j in range(depths[i])]
            )
            
            if i < len(depths) - 1:
                # Transition layer
                transition = nn.Sequential(
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], 
                             kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(embed_dims[i+1]),
                    nn.GELU()
                )
                stage.add_module('transition', transition)
            
            self.stages.append(stage)
            cur += depths[i]
        
        # Classification head
        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(embed_dims[-1], num_classes)
        
        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
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