import torch
import torch.nn as nn
import torch.nn.functional as F
from model.convnext import ConvNeXt, LayerNorm, Block  # 重用已有 ConvNeXt 结构


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBlockWrapper(nn.Module):
    def __init__(self, block: nn.Module, channels: int):
        super().__init__()
        self.block = block
        self.se = SEBlock(channels)

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return out


class SE_ConvNeXt(ConvNeXt):
    def __init__(self, *args, **kwargs):
        super(SE_ConvNeXt, self).__init__(*args, **kwargs)

        # 替换每个 stage 中的 Block 为带 SE 的 Block
        for stage_idx, stage in enumerate(self.stages):
            channels = self.stages[stage_idx][0].dwconv.out_channels
            wrapped_blocks = []
            for block in stage:
                wrapped_blocks.append(SEBlockWrapper(block, channels))
            self.stages[stage_idx] = nn.Sequential(*wrapped_blocks)


# 工厂方法（示例）
def se_convnext_tiny(num_classes=1000):
    return SE_ConvNeXt(
        in_chans=3,
        num_classes=num_classes,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    )
