import torch
import torch.nn as nn
import torchvision.transforms as transforms  # 虽然文件中引入了，但未直接使用
import torch.optim as optim  # 虽然文件中引入了，但未直接使用


# --- 辅助函数 ---
# 原始文件中没有提供autopad，但Conv类中使用。这里保留conv3x3和conv1x1，它们是辅助函数
def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    # 原始的conv3x3中padding是dilation，这里保持一致
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class Linear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class BasicBlock(nn.Module):
    expansion = 1
    tran_expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()  # 显式调用父类初始化
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1_layer = conv3x3(inplanes, planes, stride)  # 直接定义层
        self.bn1_layer = norm_layer(planes)  # 直接定义层
        self.relu_act = nn.ReLU(inplace=True)  # 直接定义层
        self.conv2_layer = conv3x3(planes, planes)  # 直接定义层
        self.bn2_layer = norm_layer(planes)  # 直接定义层
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_layer(x)  # 直接调用层
        out = self.bn1_layer(out)  # 直接调用层
        out = self.relu_act(out)  # 直接调用层

        out = self.conv2_layer(out)  # 直接调用层
        out = self.bn2_layer(out)  # 直接调用层

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu_act(out)

        return out


class CSPBottleneck(nn.Module):
    expansion = 2
    tran_expansion = 4

    # 恢复原始的__init__签名，同时内部使用BaseCNN的显式风格
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super().__init__()  # 显式调用父类初始化
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width =int(planes * 0.25)

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        # 显式定义各个层
        self.conv1_layer = conv1x1(inplanes, width)  # 直接定义层
        self.bn1_layer = norm_layer(width)  # 直接定义层

        self.conv2_layer = conv3x3(width, width, stride)  # 直接定义层
        self.bn2_layer = norm_layer(width)  # 直接定义层

        self.conv3_layer = conv1x1(width, planes * self.expansion)  # 直接定义层
        self.bn3_layer = norm_layer(planes * self.expansion)  # 直接定义层

        self.lrelu_act = nn.LeakyReLU()  # 直接定义层
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_layer(x)  # 直接调用层
        out = self.bn1_layer(out)  # 直接调用层
        out = self.lrelu_act(out)  # 直接调用层

        out = self.conv2_layer(out)  # 直接调用层
        out = self.bn2_layer(out)  # 直接调用层
        out = self.lrelu_act(out)  # 直接调用层

        out = self.conv3_layer(out)  # 直接调用层
        out = self.bn3_layer(out)  # 直接调用层

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.lrelu_act(out)  # 直接调用层

        return out


class CSPBlock(nn.Module):

    def __init__(self, block, inplanes, blocks, stride=1, downsample=None, norm_layer=None, activation=None):
        super().__init__()  # 显式调用父类初始化

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if activation is None:
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = activation()

        self.inplanes = inplanes
        self.norm_layer = norm_layer

        self.crossstage_conv = nn.Conv2d(self.inplanes, self.inplanes * 2, kernel_size=1, stride=1, bias=False)  # 直接定义层
        self.bn_crossstage = norm_layer(self.inplanes * 2)  # 直接定义层

        ## first layer is different from others
        if (self.inplanes <= 64):
            self.conv1_layer = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, bias=False)  # 直接定义层
            self.bn1_layer = norm_layer(self.inplanes)  # 直接定义层
            self.layer_num = self.inplanes
        else:
            self.conv1_layer = nn.Conv2d(self.inplanes, self.inplanes * 2, kernel_size=1, stride=1, bias=False)  # 直接定义层
            self.bn1_layer = norm_layer(self.inplanes * 2)  # 直接定义层
            self.layer_num = self.inplanes * 2

        # Use nn.ModuleList to hold the layers explicitly
        self.layers_module_list = nn.ModuleList(
            self._make_layer(block, self.inplanes, blocks, stride))  # 将_make_layer返回的列表转换为ModuleList

        self.trans_conv = nn.Conv2d(self.inplanes * 2, self.inplanes * 2, kernel_size=1, stride=1, bias=False)  # 直接定义层

    def forward(self, x):
        cross = self.crossstage_conv(x)  # 直接调用层
        cross = self.bn_crossstage(cross)  # 直接调用层
        cross = self.activation(cross)  # 直接调用层

        origin = self.conv1_layer(x)  # 直接调用层
        origin = self.bn1_layer(origin)  # 直接调用层
        origin = self.activation(origin)  # 直接调用层

        # 遍历 ModuleList
        for layer in self.layers_module_list:
            origin = layer(origin)

        # 原始代码中 trans 被注释掉，这里也注释掉
        # origin = self.trans_conv(origin)

        out = torch.cat((origin, cross), dim=1)

        return out

    def _make_layer(self, block, planes, blocks, stride=1):

        norm_layer = self.norm_layer
        downsample = None

        if stride != 1 or self.layer_num != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )

        layers = []

        # 这里的 block 就是 CSPBottleneck 或 BasicBlock，它们都接受 stride 作为位置参数
        if (self.inplanes <= 64):
            layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
            self.inplanes = planes * block.expansion
        else:
            self.inplanes = planes * block.expansion
            layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return layers  # 返回列表，供ModuleList使用


class CSPResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 norm_layer=None):
        super().__init__()  # 显式调用父类初始化
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64

        self.conv1_layer = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)  # 直接定义层
        self.bn1_layer = norm_layer(self.inplanes)  # 直接定义层
        self.lrelu_act = nn.LeakyReLU()  # 直接定义层

        self.maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)  # 直接定义层

        # 调整 CSPBlock 的实例化方式，传入正确的 block 类型和参数
        # 原始 CSPResNet.py 中 CSPBlock 的 __init__ 是 (block, inplanes,blocks, stride=1, downsample=None, norm_layer=None, activation = None)
        self.layer1_csp_block = CSPBlock(block, 64, layers[0], stride=1, norm_layer=norm_layer, activation=nn.LeakyReLU)

        self.part_tran1_seq = self._make_tran(64, block.tran_expansion)

        # 注意：layer2, layer3, layer4 在原始文件中 layers[i]-1，且 CSPBlock 内部控制 stride
        # 原始的 layer2 的 stride 是通过 _make_tran 实现的下采样。
        self.layer2_csp_block = CSPBlock(block, 128, layers[1] - 1, stride=1, norm_layer=norm_layer, activation=Linear)

        self.part_tran2_seq = self._make_tran(128, block.tran_expansion)

        self.layer3_csp_block = CSPBlock(block, 256, layers[2] - 1, stride=1, norm_layer=norm_layer, activation=Linear)

        self.part_tran3_seq = self._make_tran(256, block.tran_expansion)

        self.layer4_csp_block = CSPBlock(block, 512, layers[3] - 1, stride=1, norm_layer=norm_layer,
                                         activation=nn.LeakyReLU)

        self.conv2_layer = nn.Conv2d(512 * block.tran_expansion, 512 * 2, kernel_size=1, stride=1, bias=False)

        self.avgpool_layer = nn.AdaptiveAvgPool2d((1, 1))

        self.fn_linear = nn.Linear(512 * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, CSPBottleneck):
                    nn.init.constant_(m.bn3_layer.weight, 0)  # 修改为 bn3_layer
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2_layer.weight, 0)  # 修改为 bn2_layer

    def _make_tran(self, base, tran_expansion):
        return nn.Sequential(
            conv1x1(base * tran_expansion, base * 2),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(),
            conv3x3(base * 2, base * 2, stride=2),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU()
        )

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1_layer(x)  # 直接调用层
        x = self.bn1_layer(x)  # 直接调用层
        x = self.lrelu_act(x)  # 直接调用层
        x = self.maxpool_layer(x)  # 直接调用层

        x = self.layer1_csp_block(x)  # 直接调用层
        x = self.part_tran1_seq(x)  # 直接调用层

        x = self.layer2_csp_block(x)  # 直接调用层
        x = self.part_tran2_seq(x)  # 直接调用层

        x = self.layer3_csp_block(x)  # 直接调用层
        x = self.part_tran3_seq(x)  # 直接调用层

        x = self.layer4_csp_block(x)  # 直接调用层

        x = self.conv2_layer(x)  # 直接调用层

        x = self.avgpool_layer(x)  # 直接调用层

        x = x.view(-1, 512 * 2)

        x = self.fn_linear(x)  # 直接调用层

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _cspresnet(arch, block, layers, pretrained, model_path, **kwargs):
    model = CSPResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    return model


def csp_resnet50(pretrained=False, model_path="checkpoint.pt", **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _cspresnet('cspresnet50', CSPBottleneck, [2, 2, 3, 2], pretrained, model_path=model_path,
                      **kwargs)


def csp_resnet101(pretrained=False, model_path="checkpoint.pt", **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _cspresnet('cspresnet101', CSPBottleneck, [3, 4, 23, 3], pretrained, model_path=model_path,
                      **kwargs)


def csp_resnet152(pretrained=False, model_path="checkpoint.pt", **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _cspresnet('cspresnet152', CSPBottleneck, [3, 8, 36, 3], pretrained, model_path=model_path,
                      **kwargs)


if __name__ == "__main__":
    net = csp_resnet152(pretrained=False, num_classes=10)
    y = net(torch.randn(1, 3, 112, 112))
    print(y.size())