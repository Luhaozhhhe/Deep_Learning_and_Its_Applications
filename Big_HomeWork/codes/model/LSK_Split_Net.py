import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)
    
class SplitAttention(nn.Module):
    def __init__(self, channel, groups=2):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.group_channel = channel // groups
        
        # Modify MLP to output correct dimensions
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 4, channel)  # Changed from channel * groups to channel
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Add dimension checks
        assert c % self.groups == 0, f"Channel dimension {c} must be divisible by groups {self.groups}"
        
        # Split into groups
        x_groups = x.view(b, self.groups, self.group_channel, h, w)
        
        # Global information
        pooled = self.avg_pool(x).view(b, c)
        mlp_out = self.mlp(pooled)  # [b, c]
        
        # Reshape attention vectors correctly
        attention_vectors = mlp_out.view(b, self.groups, self.group_channel)
        attention_vectors = self.softmax(attention_vectors)
        
        # Apply attention to each group
        out = x_groups * attention_vectors.view(b, self.groups, self.group_channel, 1, 1)
        out = out.view(b, c, h, w)
        
        return out

# Also modify LSKSplitAttention to handle the dimensions correctly
class LSKSplitAttention(nn.Module):
    def __init__(self, dim, groups=2):
        super().__init__()
        assert dim % groups == 0, f"Dimension {dim} must be divisible by groups {groups}"
        
        # Original LSK components
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=3, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        
        # Split-Attention components
        self.split_attn = SplitAttention(dim, groups=groups)
        
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
        lsk_out = torch.cat([attn1, attn2], dim=1)  # Restores original dimension
        
        # Split-Attention path
        split_out = self.split_attn(x)
        
        # Fusion
        out = lsk_out + split_out
        out = self.conv_final(out)
        out = self.norm(out)
        out = self.act(out)
        
        return identity + out

class LSKSplitBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0., groups=2):
        super().__init__()
        # Combined attention
        self.attn = LSKSplitAttention(dim, groups=groups)
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
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class LSKSplitNet(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 num_classes=100,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 depths=[2, 2, 6, 2],
                 drop_path_rate=0.1,
                 groups=2):
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
                *[LSKSplitBlock(
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop_path=dpr[cur + j],
                    groups=groups
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
        
        # Classification head
        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(embed_dims[-1], num_classes)
        
        # Initialize weights
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