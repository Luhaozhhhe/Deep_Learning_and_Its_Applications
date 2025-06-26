import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    """应用于模块前的标准化层"""
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class SE(nn.Module):
    """Squeeze-and-Excitation注意力模块"""
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FeedForward(nn.Module):
    """Transformer中的前馈网络部分"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """带有相对位置编码的自注意力机制"""
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # 相对位置偏置表
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 获取实际序列长度，确保与预设的图像尺寸匹配
        seq_len = x.shape[1]
        if seq_len != self.ih * self.iw:
            # 如果序列长度与预设的图像尺寸不匹配，动态调整
            actual_size = int(seq_len ** 0.5)  # 假设是正方形
            if actual_size ** 2 == seq_len:  # 确认是完美平方数
                # 重新计算相对位置索引
                coords = torch.meshgrid((torch.arange(actual_size, device=x.device), 
                                         torch.arange(actual_size, device=x.device)))
                coords = torch.flatten(torch.stack(coords), 1)
                relative_coords = coords[:, :, None] - coords[:, None, :]
                
                relative_coords[0] += actual_size - 1
                relative_coords[1] += actual_size - 1
                relative_coords[0] *= 2 * actual_size - 1
                relative_coords = rearrange(relative_coords, 'c h w -> h w c')
                relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
                
                # 插值调整相对位置偏置表的大小
                old_size = (2 * self.ih - 1) * (2 * self.iw - 1)
                new_size = (2 * actual_size - 1) * (2 * actual_size - 1)
                if old_size != new_size:
                    # 使用普通张量而不是nn.Parameter，因为这只是临时计算用
                    new_rel_bias_table = torch.zeros((new_size, self.heads), device=x.device)
                    # 使用最近邻插值
                    if old_size > new_size:
                        # 下采样
                        step = old_size / new_size
                        for i in range(new_size):
                            idx = min(int(i * step), old_size - 1)
                            new_rel_bias_table[i] = self.relative_bias_table[idx]
                    else:
                        # 上采样
                        step = new_size / old_size
                        for i in range(old_size):
                            start_idx = int(i * step)
                            end_idx = int((i + 1) * step)
                            for j in range(start_idx, end_idx):
                                if j < new_size:
                                    new_rel_bias_table[j] = self.relative_bias_table[i]
                    
                    # 使用新的相对位置偏置表和索引
                    rel_bias = new_rel_bias_table.gather(0, relative_index.repeat(1, self.heads))
                    rel_bias = rearrange(rel_bias, '(h w) c -> 1 c h w', 
                                        h=seq_len, w=seq_len)
                else:
                    # 尺寸相同，直接使用原始表
                    rel_bias = self.relative_bias_table.gather(0, relative_index.repeat(1, self.heads))
                    rel_bias = rearrange(rel_bias, '(h w) c -> 1 c h w', 
                                        h=seq_len, w=seq_len)
            else:
                # 如果不是完美平方数，则不使用相对位置编码
                rel_bias = 0
        else:
            # 使用预设的相对位置编码
            rel_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.heads))
            rel_bias = rearrange(rel_bias, '(h w) c -> 1 c h w', 
                                h=self.ih*self.iw, w=self.ih*self.iw)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 添加相对位置偏置
        if isinstance(rel_bias, torch.Tensor):
            # 确保形状匹配
            if dots.shape[-1] == rel_bias.shape[-1] and dots.shape[-2] == rel_bias.shape[-2]:
                dots = dots + rel_bias
        else:
            # 如果rel_bias是标量0，直接跳过
            pass

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class MBConv(nn.Module):
    """MBConv模块，源自MobileNetV2，带有SE模块"""
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)

# 结合ResNet18和CoAtNet的基本块 - 使用注意力机制
class ResNetCoAtBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, image_size=(8, 8), heads=4, dim_head=32, dropout=0.):
        super(ResNetCoAtBlock, self).__init__()
        
        # 第一个卷积层保持ResNet风格
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        
        # 使用CoAtNet的注意力机制替换第二个卷积
        h, w = image_size
        # 计算第一个卷积后的特征图尺寸
        if stride == 2:
            h, w = h // 2, w // 2
        
        # 存储特征图尺寸，用于重排张量
        self.feature_size = (h, w)
            
        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(outchannel, Attention(outchannel, outchannel, (h, w), heads, dim_head, dropout), nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=h, iw=w)
        )
        
        # 添加前馈网络
        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(outchannel, FeedForward(outchannel, outchannel * 4, dropout), nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=h, iw=w)
        )
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
 
    def forward(self, x):
        # 第一个卷积层
        out = self.relu(self.bn1(self.conv1(x)))
        
        # 验证特征图尺寸是否与预期一致
        _, _, h, w = out.shape
        if (h, w) != self.feature_size:
            # 如果不一致，调整Rearrange操作
            attn_out = out.reshape(out.shape[0], out.shape[1], -1).transpose(1, 2)  # B,C,H,W -> B,HW,C
            attn_out = self.attn[1](attn_out)  # 应用注意力和LayerNorm
            attn_out = attn_out.transpose(1, 2).reshape(out.shape[0], out.shape[1], h, w)  # B,HW,C -> B,C,H,W
            
            ff_out = out.reshape(out.shape[0], out.shape[1], -1).transpose(1, 2)  # B,C,H,W -> B,HW,C
            ff_out = self.ff[1](ff_out)  # 应用前馈网络和LayerNorm
            ff_out = ff_out.transpose(1, 2).reshape(out.shape[0], out.shape[1], h, w)  # B,HW,C -> B,C,H,W
        else:
            # 如果尺寸一致，使用原始的顺序处理
            attn_out = self.attn(out)
            ff_out = self.ff(out)
        
        # 应用注意力机制和前馈网络
        out = out + attn_out
        out = out + ff_out
        
        # 残差连接
        out = out + self.shortcut(x)
        out = self.relu(out)
        
        return out

class ResNet18_CoAtNet(nn.Module):
    def __init__(self, num_classes=100, dropout=0.1):
        super(ResNet18_CoAtNet, self).__init__()
        
        # 基本参数设置
        self.inchannel = 64
        
        # ResNet18的stem层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # 假设输入是32x32的CIFAR100图像
        # 经过stem后仍为32x32
        # 第一层后仍为32x32
        # 第二层后变为16x16
        # 第三层后变为8x8
        # 第四层后变为4x4
        
        # 创建四个阶段，每个阶段2个块
        self.layer1 = self.make_layer(64, 2, stride=1, image_size=(32, 32), heads=4)
        self.layer2 = self.make_layer(128, 2, stride=2, image_size=(32, 32), heads=4)  # 第一个块会将尺寸减半
        self.layer3 = self.make_layer(256, 2, stride=2, image_size=(16, 16), heads=8)  # 第一个块会将尺寸减半
        self.layer4 = self.make_layer(512, 2, stride=2, image_size=(8, 8), heads=8)    # 第一个块会将尺寸减半
        
        # 全连接层
        self.fc = nn.Linear(512, num_classes)
        
        # 初始化权重
        self._init_weights()
 
    def make_layer(self, channels, num_blocks, stride, image_size, heads):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        h, w = image_size
        
        for i, stride in enumerate(strides):
            # 创建块
            current_image_size = (h, w)
            layers.append(ResNetCoAtBlock(
                self.inchannel, 
                channels, 
                stride=stride,
                image_size=current_image_size,
                heads=heads
            ))
            
            # 更新通道数
            self.inchannel = channels
            
            # 如果是第一个块且stride=2，则更新图像尺寸
            if i == 0 and stride == 2:
                h, w = h // 2, w // 2
            
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
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

def ResNet18CoAtNet_Tiny(num_classes=100):
    """Tiny版本的ResNet18-CoAtNet混合模型"""
    return ResNet18_CoAtNet(
        num_classes=num_classes,
        dropout=0.1
    )

def ResNet18CoAtNet_Small(num_classes=100):
    """Small版本的ResNet18-CoAtNet混合模型，增加dropout"""
    return ResNet18_CoAtNet(
        num_classes=num_classes,
        dropout=0.3
    )

def ResNet18CoAtNet_Base(num_classes=100):
    """Base版本的ResNet18-CoAtNet混合模型，进一步增加dropout"""
    return ResNet18_CoAtNet(
        num_classes=num_classes,
        dropout=0.5
    ) 