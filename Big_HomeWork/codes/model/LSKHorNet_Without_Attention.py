import torch
import torch.nn as nn
import torch.nn.functional as F

class LSKConv(nn.Module):
    """Large Selective Kernel Convolution"""
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        
        # Fusion layer
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//8, 3*dim, 1)
        )
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        u = x.clone()
        
        # Multi-scale convolutions
        conv0 = self.conv0(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        
        # Generate attention weights
        fusion = self.fc(x)
        attention = fusion.reshape(-1, 3, x.shape[1], 1, 1)
        attention = F.softmax(attention, dim=1)
        
        # Combine multi-scale features
        out = conv0 * attention[:, 0] + \
              conv1 * attention[:, 1] + \
              conv2 * attention[:, 2]
        
        out = self.norm(out)
        out = self.act(out)
        return out + u

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
        
        # Concatenate and fuse
        out = torch.cat(features, dim=1)
        out = self.fusion(out)
        out = self.norm(out)
        out = self.act(out)
        return out

class LSKHOBlock(nn.Module):
    """Combined LSK and High-Order Block"""
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            LSKConv(inchannel),
            nn.Conv2d(inchannel, outchannel, 1, stride),
            nn.BatchNorm2d(outchannel),
            nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            HighOrderBlock(outchannel),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1),
            nn.BatchNorm2d(outchannel)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = F.gelu(out)
        return out

class LSKHOResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        
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
            layers.append(LSKHOBlock(self.inchannel, channels, stride))
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