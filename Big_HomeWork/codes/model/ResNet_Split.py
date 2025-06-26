import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitAttention(nn.Module):
    """Split-Attention Module"""
    def __init__(self, in_channels, groups=2):
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        self.channels_per_group = in_channels // groups
        
        # Global information fusion
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Modified channel attention with correct dimensions
        reduction_factor = 8
        hidden_channels = max(in_channels // reduction_factor, 32)
        
        # Modified MLP to handle group-wise features
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels)  # Changed output dimension
        )
        
        # Final group-wise transformation
        self.group_fc = nn.Linear(in_channels, groups * in_channels // groups)
        self.softmax = nn.Softmax(dim=1)
        self.gn = nn.GroupNorm(groups, in_channels)
        
    def forward(self, x):
        batch, channels, height, width = x.size()
        
        # Ensure channel consistency
        assert channels == self.in_channels, f'Input channels {channels} does not match initialized channels {self.in_channels}'
        
        # Split into groups
        x_groups = x.view(batch, self.groups, self.channels_per_group, height, width)
        
        # Global context
        attn = self.avg_pool(x)  # [B, C, 1, 1]
        attn = attn.squeeze(-1).squeeze(-1)  # [B, C]
        
        # Transform through MLP
        attn = self.mlp(attn)  # [B, C]
        
        # Group-wise transformation
        attn = self.group_fc(attn)  # [B, G*C//G]
        
        # Reshape attention weights to match grouped input
        attn = attn.view(batch, self.groups, self.channels_per_group)  # [B, G, C//G]
        attn = attn.unsqueeze(-1).unsqueeze(-1)  # [B, G, C//G, 1, 1]
        attn = self.softmax(attn)
        
        # Apply attention weights
        out = x_groups * attn
        out = out.view(batch, channels, height, width)
        out = self.gn(out)
        
        return out

class SplitAttentionBlock(nn.Module):
    """ResNet block with Split-Attention"""
    def __init__(self, inchannel, outchannel, stride=1, groups=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        
        # Split-Attention module
        self.sa = SplitAttention(outchannel, groups=groups)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.sa(out)  # Apply split-attention
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SplitAttentionResNet(nn.Module):
    def __init__(self, num_classes=100, groups=2):
        super().__init__()
        # Base channels for the network
        base_channels = 64
        self.inchannel = base_channels
        
        # Initial convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Main residual layers
        self.layer1 = self.make_layer(base_channels, 2, stride=1, groups=groups)
        self.layer2 = self.make_layer(base_channels * 2, 2, stride=2, groups=groups)
        self.layer3 = self.make_layer(base_channels * 4, 2, stride=2, groups=groups)
        self.layer4 = self.make_layer(base_channels * 8, 2, stride=2, groups=groups)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 8, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def make_layer(self, channels, num_blocks, stride, groups):
        """Helper function to create a layer of blocks"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(SplitAttentionBlock(self.inchannel, channels, stride, groups))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out