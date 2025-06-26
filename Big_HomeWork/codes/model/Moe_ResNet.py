import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    """Expert routing module"""
    def __init__(self, in_channels, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_experts, 1)
        )
        
    def forward(self, x):
        weights = self.gate(x)
        weights = weights.view(weights.size(0), -1)
        weights = F.softmax(weights, dim=1)
        return weights

class MoEBasicBlock(nn.Module):
    """Mixture of Experts Basic Block"""
    def __init__(self, inchannel, outchannel, stride=1, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        
        # Create multiple expert paths
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
                nn.BatchNorm2d(outchannel)
            ) for _ in range(num_experts)
        ])
        
        # Router network
        self.router = Router(inchannel, num_experts)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        # Get routing weights
        weights = self.router(x)  # [B, num_experts]
        
        # Apply each expert
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            expert_outputs.append(expert_out)
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, C, H, W]
        
        # Weight and combine expert outputs
        B, E, C, H, W = expert_outputs.size()
        weighted_output = (expert_outputs * weights.view(B, E, 1, 1, 1)).sum(dim=1)
        
        # Add shortcut connection
        out = weighted_output + self.shortcut(x)
        out = F.relu(out)
        
        return out

class MoEResNet18(nn.Module):
    def __init__(self, num_classes=100, num_experts=4):
        super().__init__()
        self.inchannel = 64
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # MoE layers
        self.layer1 = self.make_layer(64, 2, stride=1, num_experts=num_experts)
        self.layer2 = self.make_layer(128, 2, stride=2, num_experts=num_experts)
        self.layer3 = self.make_layer(256, 2, stride=2, num_experts=num_experts)
        self.layer4 = self.make_layer(512, 2, stride=2, num_experts=num_experts)
        
        # Classification head
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
    
    def make_layer(self, channels, num_blocks, stride, num_experts):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(MoEBasicBlock(self.inchannel, channels, stride, num_experts))
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