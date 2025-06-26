#ghoostnet与cspnet的融合
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- GhostNet Components (from your provided code) ---
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


# --- Modified CSPNet Components ---

# Helper for 3x3 Ghost Convolution
def conv3x3_ghost(in_planes, out_planes, stride=1, dilation=1, relu=True):
    """3x3 Ghost convolution with padding"""
    return GhostModule(in_planes, out_planes, kernel_size=3, stride=stride, dw_size=dilation * 2 + 1,
                       relu=relu)  # dw_size approximates padding in GhostModule


# Helper for 1x1 Ghost Convolution
def conv1x1_ghost(in_planes, out_planes, stride=1, relu=True):
    """1x1 Ghost convolution"""
    return GhostModule(in_planes, out_planes, kernel_size=1, stride=stride, relu=relu)


class Linear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class BasicBlock(nn.Module):
    # This BasicBlock is kept for completeness but CSPGhostBottleneck will be used in CSPGhostResNet.
    # If you intend to use BasicBlock with Ghost operations, its conv layers would need to be GhostModule as well.
    expansion = 1
    tran_expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1_layer = conv3x3_ghost(inplanes, planes, stride)  # Using Ghost here
        self.bn1_layer = norm_layer(planes)
        self.relu_act = nn.ReLU(inplace=True)
        self.conv2_layer = conv3x3_ghost(planes, planes)  # Using Ghost here
        self.bn2_layer = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_layer(x)
        out = self.bn1_layer(out)
        out = self.relu_act(out)

        out = self.conv2_layer(out)
        out = self.bn2_layer(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu_act(out)

        return out


class CSPGhostBottleneck(nn.Module):
    expansion = 2
    tran_expansion = 4  # Kept this for compatibility with original CSPBottleneck's tran_expansion

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # The 'width' here is often related to the internal expansion,
        # but for GhostModule, the 'ratio' and 'oup' manage the internal channels.
        # We'll map 'width' to the input for GhostModule to control intermediate size.
        # Original CSPBottleneck used planes * 0.25, let's keep that scaling factor for relative size.
        ghost_module_internal_width = int(planes * 0.25)  # This becomes input for first ghost module if relu=True

        # Note: GhostModule handles its own internal BN and ReLU, but we can add external for consistency if needed.

        self.ghost1 = GhostModule(inplanes, ghost_module_internal_width, kernel_size=1, stride=1, relu=True)
        self.bn1_layer = norm_layer(
            ghost_module_internal_width)  # Adding BN explicitly after GhostModule for consistency

        self.ghost2 = GhostModule(ghost_module_internal_width, ghost_module_internal_width, kernel_size=3,
                                  stride=stride, relu=True)
        self.bn2_layer = norm_layer(
            ghost_module_internal_width)  # Adding BN explicitly after GhostModule for consistency

        self.ghost3 = GhostModule(ghost_module_internal_width, planes * self.expansion, kernel_size=1, stride=1,
                                  relu=False)  # Final projection
        self.bn3_layer = norm_layer(planes * self.expansion)

        self.lrelu_act = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.ghost1(x)
        out = self.bn1_layer(out)
        out = self.lrelu_act(out)

        out = self.ghost2(out)
        out = self.bn2_layer(out)
        out = self.lrelu_act(out)

        out = self.ghost3(out)
        out = self.bn3_layer(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.lrelu_act(out)

        return out


class CSPBlock(nn.Module):
    def __init__(self, block, inplanes, blocks, stride=1, downsample=None, norm_layer=None, activation=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if activation is None:
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = activation()

        self.inplanes = inplanes
        self.norm_layer = norm_layer

        # Using GhostModule for cross-stage convolution
        self.crossstage_conv = conv1x1_ghost(self.inplanes, self.inplanes * 2,
                                             relu=False)  # GhostModule has its own ReLU
        self.bn_crossstage = norm_layer(self.inplanes * 2)

        ## first layer is different from others
        if (self.inplanes <= 64):
            self.conv1_layer = conv1x1_ghost(self.inplanes, self.inplanes, relu=False)  # Using Ghost here
            self.bn1_layer = norm_layer(self.inplanes)
            self.layer_num = self.inplanes
        else:
            self.conv1_layer = conv1x1_ghost(self.inplanes, self.inplanes * 2, relu=False)  # Using Ghost here
            self.bn1_layer = norm_layer(self.inplanes * 2)
            self.layer_num = self.inplanes * 2

        self.layers_module_list = nn.ModuleList(
            self._make_layer(block, self.inplanes, blocks, stride))

        # This was commented out in original CSPBlock, keeping it consistent.
        # self.trans_conv = nn.Conv2d(self.inplanes * 2, self.inplanes * 2, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        cross = self.crossstage_conv(x)
        cross = self.bn_crossstage(cross)
        cross = self.activation(cross)

        origin = self.conv1_layer(x)
        origin = self.bn1_layer(origin)
        origin = self.activation(origin)

        for layer in self.layers_module_list:
            origin = layer(origin)

        out = torch.cat((origin, cross), dim=1)

        return out

    def _make_layer(self, block, planes, blocks, stride=1):

        norm_layer = self.norm_layer
        downsample = None
        current_expansion = block.expansion if hasattr(block, 'expansion') else 1

        # 确定进入 _make_layer 的 block 的第一个子块的输入通道数
        # 这个输入通道数应该与 CSPBlock 内部 origin 分支的输出通道数一致
        if self.inplanes <= 64:
            # 对应 CSPBlock.__init__ 中 conv1_layer(self.inplanes, self.inplanes) 的情况
            # 此时 origin 分支输出通道数是 self.inplanes
            input_channels_for_first_block_in_list = self.inplanes
        else:
            # 对应 CSPBlock.__init__ 中 conv1_layer(self.inplanes, self.inplanes * 2) 的情况
            # 此时 origin 分支输出通道数是 self.inplanes * 2
            input_channels_for_first_block_in_list = self.inplanes * 2

            # 下采样路径也需要使用正确的输入通道数
        # if stride != 1 or self.layer_num != planes * current_expansion: # 原始判断条件
        # self.layer_num 已经等于 input_channels_for_first_block_in_list
        if stride != 1 or input_channels_for_first_block_in_list != planes * current_expansion:
            downsample = nn.Sequential(
                conv1x1_ghost(input_channels_for_first_block_in_list, planes * current_expansion, relu=False),
                norm_layer(planes * current_expansion),
            )

        layers = []

        # 创建第一个 block，使用正确的输入通道数
        layers.append(block(input_channels_for_first_block_in_list, planes, stride, downsample, norm_layer))

        # 更新 self.inplanes，这个值将作为后续 block 的输入通道
        # 这里的 self.inplanes 是 CSPBlock 实例的成员变量
        # 它应该反映 _make_layer 生成的这个层序列的输出通道，或者说是下一个 CSPBlock 的 inplanes
        # 但在当前上下文中，它更重要的是为 for 循环中剩余的 block 提供正确的输入通道
        self.inplanes = planes * current_expansion

        # 创建剩余的 block
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return layers


class CSPGhostResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64

        # Initial stem using GhostModule for conv1
        # Original: nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_layer = GhostModule(3, self.inplanes, kernel_size=7, stride=2,
                                       relu=False)  # GhostModule includes its own BN/ReLU, relu=False means only BN.
        self.bn1_layer = norm_layer(self.inplanes)  # Keeping explicit BN for consistency after GhostModule
        self.lrelu_act = nn.LeakyReLU(inplace=True)

        self.maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1_csp_block = CSPBlock(block, 64, layers[0], stride=1, norm_layer=norm_layer, activation=nn.LeakyReLU)
        self.part_tran1_seq = self._make_tran(64, block.tran_expansion)

        self.layer2_csp_block = CSPBlock(block, 128, layers[1] - 1, stride=1, norm_layer=norm_layer, activation=Linear)
        self.part_tran2_seq = self._make_tran(128, block.tran_expansion)

        self.layer3_csp_block = CSPBlock(block, 256, layers[2] - 1, stride=1, norm_layer=norm_layer, activation=Linear)
        self.part_tran3_seq = self._make_tran(256, block.tran_expansion)

        self.layer4_csp_block = CSPBlock(block, 512, layers[3] - 1, stride=1, norm_layer=norm_layer,
                                         activation=nn.LeakyReLU)

        # Final 1x1 conv layer also using GhostModule
        self.conv2_layer = conv1x1_ghost(512 * block.tran_expansion, 512 * 2, relu=False)
        self.bn2_layer = norm_layer(512 * 2)  # Added BN for consistency
        self.lrelu_conv2 = nn.LeakyReLU(inplace=True)  # Added activation after final 1x1 conv

        self.avgpool_layer = nn.AdaptiveAvgPool2d((1, 1))

        self.fn_linear = nn.Linear(512 * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, GhostModule):  # Initialize GhostModule parts
                if hasattr(m.primary_conv, '0') and isinstance(m.primary_conv[0], nn.Conv2d):
                    nn.init.kaiming_normal_(m.primary_conv[0].weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m.cheap_operation, '0') and isinstance(m.cheap_operation[0], nn.Conv2d):
                    nn.init.kaiming_normal_(m.cheap_operation[0].weight, mode='fan_out', nonlinearity='relu')

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, CSPGhostBottleneck):
                    nn.init.constant_(m.bn3_layer.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2_layer.weight, 0)

    def _make_tran(self, base, tran_expansion):
        # Transition layer using GhostModule for convolutions
        return nn.Sequential(
            # 确保这里的relu=False，因为后面有BatchNorm2d和LeakyReLU
            conv1x1_ghost(base * tran_expansion, base * 2, relu=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(inplace=True),
            # 确保这里的relu=False，因为后面有BatchNorm2d和LeakyReLU
            conv3x3_ghost(base * 2, base * 2, stride=2, relu=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(inplace=True)
        )

    def _forward_impl(self, x):
        x = self.conv1_layer(x)
        x = self.bn1_layer(x)
        x = self.lrelu_act(x)
        x = self.maxpool_layer(x)

        x = self.layer1_csp_block(x)
        # print(f"Shape after layer1_csp_block: {x.shape}") # Debug line 1
        x = self.part_tran1_seq(x)
        # print(f"Shape after part_tran1_seq: {x.shape}") # Debug line 2

        x = self.layer2_csp_block(x)
        x = self.part_tran2_seq(x)

        x = self.layer3_csp_block(x)
        x = self.part_tran3_seq(x)

        x = self.layer4_csp_block(x)

        x = self.conv2_layer(x)
        x = self.bn2_layer(x)  # Added BN
        x = self.lrelu_conv2(x)  # Added activation

        x = self.avgpool_layer(x)
        x = x.view(-1, 512 * 2)
        x = self.fn_linear(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _csp_ghost_resnet(arch, block, layers, pretrained, model_path, **kwargs):
    model = CSPGhostResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    return model


def csp_ghost_resnet50(pretrained=False, model_path="checkpoint.pt", **kwargs):
    return _csp_ghost_resnet('cspghostresnet50', CSPGhostBottleneck, [2, 2, 3, 2], pretrained, model_path=model_path,
                             **kwargs)


def csp_ghost_resnet101(pretrained=False, model_path="checkpoint.pt", **kwargs):
    return _csp_ghost_resnet('cspghostresnet101', CSPGhostBottleneck, [3, 4, 23, 3], pretrained, model_path=model_path,
                             **kwargs)


def csp_ghost_resnet152(pretrained=False, model_path="checkpoint.pt", **kwargs):
    return _csp_ghost_resnet('cspghostresnet152', CSPGhostBottleneck, [3, 8, 36, 3], pretrained, model_path=model_path,
                             **kwargs)


if __name__ == "__main__":
    net = csp_ghost_resnet152(pretrained=False, num_classes=10)
    y = net(torch.randn(1, 3, 112, 112))
    print(y.size())

    # Test with a larger input as GhostNet often handles various input sizes
    net_large = csp_ghost_resnet50(pretrained=False, num_classes=100)
    y_large = net_large(torch.randn(2, 3, 224, 224))
    print(y_large.size())