import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial

class LayerNorm(nn.Module):
    """LayerNorm，支持channels_first和channels_last格式"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def get_dwconv(dim, kernel, bias):
    """获取分组卷积（深度卷积）"""
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2, bias=bias, groups=dim)

class GlobalLocalFilter(nn.Module):
    """全局-局部滤波器，使用FFT实现全局交互"""
    def __init__(self, dim, h=8, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        # 对于CIFAR100的小图像，初始化较小的复权重
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3,0,1,2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1,2,3,0)

        weight = torch.view_as_complex(weight.contiguous())
        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x

class gnconv(nn.Module):
    """门控网络卷积模块，HorNet的核心创新"""
    def __init__(self, dim, order=5, gflayer=None, h=8, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )
        self.scale = s

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i+1]

        x = self.proj_out(x)
        return x

# 结合ResNet18和HorNet的基本块
class ResNetHorBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, drop_path=0., layer_scale_init_value=1e-6, order=3):
        super(ResNetHorBlock, self).__init__()
        
        # 使用HorNet的gnconv替换第二个卷积
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        
        # 使用gnconv替换第二个3x3卷积
        self.gnconv = gnconv(outchannel, order=order)
        self.norm = LayerNorm(outchannel, eps=1e-6, data_format='channels_first')
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(outchannel), 
                                requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
 
    def forward(self, x):
        # 第一个卷积层保持ResNet风格
        out = self.relu(self.bn1(self.conv1(x)))
        
        # 第二个卷积使用gnconv
        if self.gamma is not None:
            gamma = self.gamma.view(self.gamma.shape[0], 1, 1)
        else:
            gamma = 1
        
        # 应用gnconv
        gnconv_out = self.gnconv(self.norm(out))
        if self.gamma is not None:
            gnconv_out = gnconv_out * gamma
        
        # 残差连接
        out = out + self.drop_path(gnconv_out) + self.shortcut(x)
        out = self.relu(out)
        
        return out

class ResNet18_HorNet(nn.Module):
    def __init__(self, num_classes=100, drop_path_rate=0.1, layer_scale_init_value=1e-6):
        super(ResNet18_HorNet, self).__init__()
        
        # 基本参数设置
        self.inchannel = 64
        
        # ResNet18的stem层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # 设置drop path rates
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, 8)]  # 8个块的drop path rates
        
        # 创建四个阶段，每个阶段2个块
        self.layer1 = self.make_layer(64, 2, stride=1, dp_rates=dp_rates[0:2], order=2)
        self.layer2 = self.make_layer(128, 2, stride=2, dp_rates=dp_rates[2:4], order=3)
        self.layer3 = self.make_layer(256, 2, stride=2, dp_rates=dp_rates[4:6], order=4)
        self.layer4 = self.make_layer(512, 2, stride=2, dp_rates=dp_rates[6:8], order=5)
        
        # 全连接层
        self.fc = nn.Linear(512, num_classes)
        
        # 初始化权重
        self._init_weights()
 
    def make_layer(self, channels, num_blocks, stride, dp_rates, order):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(ResNetHorBlock(
                self.inchannel, 
                channels, 
                stride=stride, 
                drop_path=dp_rates[i],
                layer_scale_init_value=1e-6,
                order=order
            ))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18HorNet_Tiny(num_classes=100):
    """Tiny版本的ResNet18-HorNet混合模型"""
    return ResNet18_HorNet(
        num_classes=num_classes,
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6
    )

def ResNet18HorNet_Small(num_classes=100):
    """Small版本的ResNet18-HorNet混合模型，增加drop path rate"""
    return ResNet18_HorNet(
        num_classes=num_classes,
        drop_path_rate=0.3,
        layer_scale_init_value=1e-6
    )

def ResNet18HorNet_Base(num_classes=100):
    """Base版本的ResNet18-HorNet混合模型，进一步增加drop path rate"""
    return ResNet18_HorNet(
        num_classes=num_classes,
        drop_path_rate=0.5,
        layer_scale_init_value=1e-6
    ) 