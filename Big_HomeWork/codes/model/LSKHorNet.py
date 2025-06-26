import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)

# Add LSKSplitAttention class
class LSKSplitAttention(nn.Module):
    def __init__(self, dim, groups=2):
        super().__init__()
        assert dim % groups == 0, f"Dimension {dim} must be divisible by groups {groups}"
        
        # LSK components
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=3, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        
        # Split attention components
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.group_channel = dim // groups
        
        # Channel mixing
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim)
        )
        
        # Final fusion
        self.conv_final = nn.Conv2d(dim, dim, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        
        # LSK path
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        # Split attention path
        B, C, H, W = x.shape
        pooled = self.avg_pool(x).view(B, C)
        attention = self.mlp(pooled).view(B, C, 1, 1)
        attention = torch.sigmoid(attention)
        
        # Combine paths
        lsk_out = torch.cat([attn1, attn2], dim=1)
        out = lsk_out * attention
        
        # Final processing
        out = self.conv_final(out)
        out = self.norm(out)
        out = self.act(out)
        
        return identity + out

class GlobalResponseNorm(nn.Module):
    """Global Response Normalization"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class gnConv(nn.Module):
    """HorNet's gn-Conv operation"""
    def __init__(self, dim, order=5, gflayer=False, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dim = dim
        
        # Input projection
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        
        # Depth-wise convolution
        if gflayer:
            self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        else:
            self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        
        # Output projection
        self.proj_out = nn.Conv2d(dim, dim, 1)
        
        self.h = h
        self.w = w
        self.s = s

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Project input
        fused_x = self.proj_in(x)  # [B, 2C, H, W]
        
        # Split channels evenly
        pwa, abc = torch.split(fused_x, [self.dim, self.dim], dim=1)
        
        # Apply depth-wise convolution
        dw_abc = self.dwconv(abc) * self.s  # [B, C, H, W]
        
        # Initialize output tensor
        out = torch.zeros_like(pwa)
        
        # Split into equal chunks and apply progressive multiplication
        chunk_size = self.dim // self.order
        for i in range(self.order):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self.order - 1 else self.dim
            
            # Get current chunks
            pwa_chunk = pwa[:, start_idx:end_idx, :, :]
            dw_chunk = dw_abc[:, start_idx:end_idx, :, :]
            
            # Multiply and store in output tensor
            out[:, start_idx:end_idx, :, :] = pwa_chunk * dw_chunk
        
        # Project back
        return self.proj_out(out)

class LSKHorBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0., groups=2, order=5):
        super().__init__()
        # LSK attention
        self.lsk_attn = LSKSplitAttention(dim, groups=groups)
        # HorNet gnConv
        self.gn_conv = gnConv(dim, order=order)
        # Global Response Norm
        self.grn = GlobalResponseNorm(dim)
        # MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_ratio, 1),
            nn.GELU(),
            DWConv(dim * mlp_ratio),
            nn.Conv2d(dim * mlp_ratio, dim, 1)
        )
        # Norms
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.norm3 = nn.BatchNorm2d(dim)
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # LSK attention path
        x = x + self.drop_path(self.lsk_attn(self.norm1(x)))
        # gnConv path with GRN
        x = x + self.drop_path(self.grn(self.gn_conv(self.norm2(x))))
        # MLP path
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x

class LSKHorNet(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 num_classes=100,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 depths=[2, 2, 6, 2],
                 drop_path_rate=0.1,
                 groups=2,
                 orders=[5, 4, 3, 2]):  # Decreasing order for each stage
        super().__init__()
        
        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], 3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU()
        )
        
        # Drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[LSKHorBlock(
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop_path=dpr[cur + j],
                    groups=groups,
                    order=orders[i]
                ) for j in range(depths[i])]
            )
            
            if i < len(depths) - 1:
                # Transition
                transition = nn.Sequential(
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], 3, stride=2, padding=1),
                    nn.BatchNorm2d(embed_dims[i+1]),
                    nn.GELU()
                )
                stage.add_module('transition', transition)
            
            self.stages.append(stage)
            cur += depths[i]
        
        # Classification head with Global Response Norm
        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.grn = GlobalResponseNorm(embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Replace Sequential with separate layers
        self.fc1 = nn.Linear(embed_dims[-1], embed_dims[-1] // 2)
        self.act = nn.GELU()
        self.head = nn.Linear(embed_dims[-1] // 2, num_classes)  # Final classification layer

    def forward(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.norm(x)
        x = self.grn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # Modified forward through classification head
        x = self.fc1(x)
        x = self.act(x)
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