import torch
import torch.nn as nn
import torch.nn.functional as F
 
# 权重初始化函数
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)  # 使用 Xavier 初始化
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
 
# Wide ResNet 的基本块
class WideBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(WideBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
# Wide ResNet 模型
class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=100):
        super(WideResNet, self).__init__()
        self.in_channels = 16
        n = (depth - 4) // 6  # 每个 stage 的层数
 
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
 
        self.layer1 = self._make_layer(WideBlock, 16 * widen_factor, n, stride=1)
        self.layer2 = self._make_layer(WideBlock, 32 * widen_factor, n, stride=2)
        self.layer3 = self._make_layer(WideBlock, 64 * widen_factor, n, stride=2)
 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * widen_factor, num_classes)
 
        # 参数初始化
        self.apply(weights_init)
 
    def _make_layer(self, block, out_channels, n, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, n):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out