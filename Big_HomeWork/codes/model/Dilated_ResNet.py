import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedBasicBlock(nn.Module):
    """Basic block with dilated convolutions"""
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=dilation,
                              dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=dilation,
                              dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.dropout = nn.Dropout2d(p=dropout)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DilatedResNet(nn.Module):
    def __init__(self, num_classes=100, dropout=0.1):
        super().__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual layers with increasing dilation
        self.layer1 = self.make_layer(64, blocks=2, stride=1, dilation=1, dropout=dropout)
        self.layer2 = self.make_layer(128, blocks=2, stride=2, dilation=1, dropout=dropout)
        self.layer3 = self.make_layer(256, blocks=2, stride=2, dilation=2, dropout=dropout)
        self.layer4 = self.make_layer(512, blocks=2, stride=2, dilation=4, dropout=dropout)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
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
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def make_layer(self, out_channels, blocks, stride, dilation, dropout):
        layers = []
        layers.append(DilatedBasicBlock(self.in_channels, out_channels, 
                                      stride, dilation, dropout))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(DilatedBasicBlock(self.in_channels, out_channels, 
                                          1, dilation, dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
