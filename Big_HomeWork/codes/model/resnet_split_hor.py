import torch
import torch.nn as nn
import torch.nn.functional as F

class HighOrderBlock(nn.Module):
    """High-Order Feature Interaction Block"""
    def __init__(self, dim, order=3):
        super().__init__()
        self.order = order
        self.dims = [dim//(2**i) for i in range(order)]
        
        # Multi-branch convolutions
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, self.dims[i], 1),
                nn.BatchNorm2d(self.dims[i]),
                nn.GELU(),
                nn.Conv2d(self.dims[i], self.dims[i], 3, padding=1, groups=self.dims[i]),
                nn.BatchNorm2d(self.dims[i]),
                nn.GELU()
            ) for i in range(order)
        ])
        
        # Fusion convolution
        self.fusion = nn.Conv2d(sum(self.dims), dim, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        features = []
        for i in range(self.order):
            feat = self.branches[i](x)
            features.append(feat)
        
        out = torch.cat(features, dim=1)
        out = self.fusion(out)
        out = self.norm(out)
        out = self.act(out)
        return out

class HOBlock(nn.Module):
    """Basic Block with High-Order Feature Interactions"""
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        # First convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.GELU()
        )
        
        # High-Order Feature block
        self.ho_block = HighOrderBlock(outchannel)
        
        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.ho_block(out)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = F.gelu(out)
        return out

class HOResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.inchannel = 64
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # Main layers
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(HOBlock(self.inchannel, channels, stride))
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