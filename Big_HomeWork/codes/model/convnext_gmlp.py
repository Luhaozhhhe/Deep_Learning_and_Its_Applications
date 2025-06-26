# 在你的原始convnext.py基础上，添加gMLP模块融合
# 假设gMLPBlock已从CNN_gMLP中复制或引入

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from model.CNN_gMLP import gMLPBlock
from model.convnext import Block
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" LayerNorm that supports 'channels_last' (default) or 'channels_first'. """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt_gMLP_Block(nn.Module):
    def __init__(self, dim, input_resolution, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        h, w = input_resolution  # 添加这一行：动态计算 token 数量
        self.gmlp = None  # 延迟初始化

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        b, c, h, w = x.shape
        gmlp_in = x.flatten(2).transpose(1, 2)

        # 👇 动态初始化 gmlp（只初始化一次）
        if self.gmlp is None:
            self.gmlp = gMLPBlock(dim=c, dim_ff=c * 4, seq_len=h * w).to(x.device)
            # self.gmlp = gMLPBlock(...).to(x.device) # 在 ConvNeXt_gMLP_Block 中延迟初始化了 self.gmlp，但是 你没有把它移动到 CUDA。默认情况下新建的模块是在 CPU 上，而模型本体已经 .to(cuda) 了，所以一混用就炸了。确保 gMLP 在正确的设备上


        gmlp_out = self.gmlp(gmlp_in)

        # 添加以下两行截断，防止 NaN 扩散
        gmlp_out = torch.nan_to_num(gmlp_out, nan=0.0, posinf=10.0, neginf=-10.0)
        gmlp_out = torch.clamp(gmlp_out, min=-10.0, max=10.0)

        gmlp_out = gmlp_out.transpose(1, 2).reshape(b, c, h, w)

        x = x + gmlp_out
        x = input + self.drop_path(x)
        return x



# 修改 ConvNeXt_gMLP 模型类中对 Block 的调用
# 示例仅替换 stage[2] 为融合gMLP的Block

class ConvNeXt_gMLP(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            if i == 2:
                stage = nn.Sequential(
                    *[ConvNeXt_gMLP_Block(dim=dims[i],
                              input_resolution=(4, 4),
                              drop_path=dp_rates[cur + j],
                              layer_scale_init_value=layer_scale_init_value)
                    for j in range(depths[i])]
                )
            else:
                stage = nn.Sequential(
                    *[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
