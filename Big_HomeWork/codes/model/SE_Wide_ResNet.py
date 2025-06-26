import torch
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):
    """Squeeze-and-Excitation模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EnhancedWideBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3, use_se=True):
        super().__init__()
        
        # 首个卷积层使用分组卷积增加特征多样性
        groups = 2 if out_channels % 2 == 0 else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 第二个卷积层使用深度可分离卷积减少参数量
        self.conv2_depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                        padding=1, groups=out_channels, bias=False)
        self.conv2_pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # SE注意力模块
        self.se = SEModule(out_channels) if use_se else nn.Identity()
        
        # 改进的shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # 使用GELU激活函数
        self.act = nn.GELU()
        
    def forward(self, x):
        # 主路径
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        
        # 深度可分离卷积
        out = self.conv2_depthwise(out)
        out = self.conv2_pointwise(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # SE注意力
        out = self.se(out)
        
        # 残差连接
        out += self.shortcut(x)
        out = self.act(out)
        return out

class EnhancedWideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=100, dropout_rate=0.3):
        super().__init__()
        self.in_channels = 16
        n = (depth - 4) // 6
        
        # 改进的stem层
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.GELU()
        )
        
        # 主干网络
        self.layer1 = self._make_layer(16 * widen_factor, n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(32 * widen_factor, n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(64 * widen_factor, n, stride=2, dropout_rate=dropout_rate)
        
        # 改进的分类头
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64 * widen_factor, 32 * widen_factor),
            nn.BatchNorm1d(32 * widen_factor),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32 * widen_factor, num_classes)
        )
        
        # 改进的权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, out_channels, n, stride, dropout_rate):
        layers = []
        layers.append(EnhancedWideBlock(self.in_channels, out_channels, 
                                      stride, dropout_rate))
        self.in_channels = out_channels
        for _ in range(1, n):
            layers.append(EnhancedWideBlock(out_channels, out_channels, 
                                          dropout_rate=dropout_rate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

def create_enhanced_wideresnet(depth=28, width=10, num_classes=100):
    return EnhancedWideResNet(depth=depth, 
                             widen_factor=width,
                             num_classes=num_classes,
                             dropout_rate=0.3)