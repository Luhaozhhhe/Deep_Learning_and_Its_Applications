import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange

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
    """HorNet的门控网络卷积模块"""
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

class HorConvBlock(nn.Module):
    """HorNet风格的卷积块，使用gnconv"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv_fn=None):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv_fn(dim) if gnconv_fn else gnconv(dim)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Attention(nn.Module):
    """CoAtNet的自注意力机制，带有相对位置编码"""
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
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 使用gather提高GPU效率
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class TransformerBlock(nn.Module):
    """CoAtNet风格的Transformer块，但使用HorNet的gnconv增强"""
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0., gnconv_fn=None):
        super().__init__()
        hidden_dim = int(oup * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        # 使用gnconv增强的注意力机制
        if gnconv_fn:
            self.gnconv_enhance = gnconv_fn(inp)
            self.use_gnconv = True
        else:
            self.use_gnconv = False

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = nn.Sequential(
            nn.Linear(oup, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, oup),
            nn.Dropout(dropout)
        )

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            LayerNorm(inp, eps=1e-6),
            self.attn,
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            LayerNorm(oup, eps=1e-6),
            self.ff,
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            if self.use_gnconv:
                # 使用gnconv增强自注意力
                x = x + self.gnconv_enhance(x) + self.attn(x)
            else:
                x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class HorCoAtNet(nn.Module):
    """结合HorNet和CoAtNet的混合网络"""
    def __init__(self, image_size=(32, 32), in_channels=3, num_blocks=[2, 2, 3, 5, 2], 
                 channels=[64, 96, 192, 384, 768], num_classes=100,
                 drop_path_rate=0.1, layer_scale_init_value=1e-6, 
                 block_types=['H', 'H', 'T', 'T']):
        super().__init__()
        ih, iw = image_size
        
        # 定义不同类型的块
        blocks = {
            'H': HorConvBlock,    # HorNet风格块
            'T': TransformerBlock # Transformer风格块
        }
        
        # 设置gf卷积的h和w参数，适配CIFAR100图像大小
        h_list = [8, 4, 2, 1]  # 针对32x32图像的特征图大小
        w_list = [8, 4, 2, 1]
        
        # 创建GlobalLocalFilter和gnconv配置
        gf = GlobalLocalFilter
        
        # 创建stem层
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1),
            LayerNorm(channels[0], eps=1e-6, data_format="channels_first")
        )
        
        # 创建下采样层
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(channels[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(channels[i], channels[i+1], kernel_size=2, stride=2),
            ))
        
        # 创建各阶段
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks[1:]))] 
        
        # 第一阶段 - 无下采样的HorNet块
        stage0_blocks = []
        for j in range(num_blocks[0]):
            gnconv_fn = partial(
                gnconv, 
                gflayer=partial(gf, h=h_list[0], w=w_list[0]),
                h=h_list[0], 
                w=w_list[0], 
                order=2
            )
            stage0_blocks.append(
                HorConvBlock(
                    dim=channels[0],
                    drop_path=0,
                    layer_scale_init_value=layer_scale_init_value,
                    gnconv_fn=gnconv_fn
                )
            )
        self.stages.append(nn.Sequential(*stage0_blocks))
        
        # 创建后续阶段
        cur = 0
        for i in range(1, 4):
            stage_blocks = []
            # 为当前阶段创建gnconv
            gnconv_fn = partial(
                gnconv, 
                gflayer=partial(gf, h=h_list[i], w=w_list[i]),
                h=h_list[i], 
                w=w_list[i], 
                order=i+1  # 从2到4递增
            )
            
            # 根据block_types选择块类型
            if block_types[i-1] == 'H':
                # HorNet风格块
                for j in range(num_blocks[i]):
                    stage_blocks.append(
                        blocks[block_types[i-1]](
                            dim=channels[i],
                            drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value,
                            gnconv_fn=gnconv_fn
                        )
                    )
            else:
                # Transformer风格块，增强版
                image_size = (ih // (2**i), iw // (2**i))
                for j in range(num_blocks[i]):
                    stage_blocks.append(
                        blocks[block_types[i-1]](
                            inp=channels[i],
                            oup=channels[i],
                            image_size=image_size,
                            heads=8,
                            dim_head=channels[i] // 8,
                            downsample=False,
                            dropout=0.1,
                            gnconv_fn=gnconv_fn  # 使用gnconv增强Transformer
                        )
                    )
            
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += num_blocks[i]
        
        # 最后一个阶段 - 分类头
        self.norm = nn.LayerNorm(channels[3], eps=1e-6)
        self.head = nn.Linear(channels[3], num_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        
        # 第一阶段
        x = self.stages[0](x)
        
        # 后续阶段
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i+1](x)
        
        # 全局池化和分类
        x = x.mean([-2, -1])  # 全局平均池化
        x = self.norm(x)
        x = self.head(x)
        
        return x

def HorCoAtNet_Tiny(num_classes=100):
    """HorCoAtNet-Tiny变体，适用于CIFAR100"""
    return HorCoAtNet(
        image_size=(32, 32),
        in_channels=3,
        num_blocks=[2, 2, 2, 4],
        channels=[64, 96, 192, 384],
        num_classes=num_classes,
        drop_path_rate=0.1,
        block_types=['H', 'H', 'T']  # 前两个阶段使用HorNet块，后一个阶段使用增强的Transformer块
    )

def HorCoAtNet_Small(num_classes=100):
    """HorCoAtNet-Small变体，适用于CIFAR100"""
    return HorCoAtNet(
        image_size=(32, 32),
        in_channels=3,
        num_blocks=[2, 3, 6, 10],
        channels=[96, 192, 384, 768],
        num_classes=num_classes,
        drop_path_rate=0.2,
        block_types=['H', 'H', 'T']
    )

def HorCoAtNet_Base(num_classes=100):
    """HorCoAtNet-Base变体，适用于CIFAR100"""
    return HorCoAtNet(
        image_size=(32, 32),
        in_channels=3,
        num_blocks=[3, 4, 12, 16],
        channels=[128, 256, 512, 1024],
        num_classes=num_classes,
        drop_path_rate=0.3,
        block_types=['H', 'H', 'T']
    ) 