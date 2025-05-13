import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(planes, max(1, round(planes / 16)))
        self.fc2 = nn.Linear(max(1, round(planes / 16)), planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        # SE模块：全局池化后FC，再利用sigmoid得到注意力权重
        se = self.globalAvgPool(out)
        se = se.view(se.size(0), -1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        se = se.view(se.size(0), se.size(1), 1, 1)
        out = out * se

        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # SE模块部分
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(planes * self.expansion, max(1, round(planes / 4)))
        self.fc2 = nn.Linear(max(1, round(planes / 4)), planes * self.expansion)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # SE模块：自适应池化及全连接得到注意力系数
        se = self.globalAvgPool(out)
        se = se.view(se.size(0), -1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        se = se.view(se.size(0), se.size(1), 1, 1)
        out = out * se

        out += residual
        out = self.relu(out)
        return out

class SENet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(SENet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SE_ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(SE_ResNet18, self).__init__()
        self.model = SENet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class SE_ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(SE_ResNet34, self).__init__()
        self.model = SENet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class SE_ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(SE_ResNet50, self).__init__()
        self.model = SENet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class SE_ResNet101(nn.Module):
    def __init__(self, num_classes):
        super(SE_ResNet101, self).__init__()
        self.model = SENet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class SE_ResNet152(nn.Module):
    def __init__(self, num_classes):
        super(SE_ResNet152, self).__init__()
        self.model = SENet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

    def forward(self, x):
        return self.model(x)