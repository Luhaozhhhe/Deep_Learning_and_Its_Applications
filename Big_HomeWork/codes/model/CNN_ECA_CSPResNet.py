import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- ECA-Net Components ---
class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Using Conv1d for efficient channel attention
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Feature descriptor on the global spatial information
        y = self.avg_pool(x) # Global average pooling: (B, C, H, W) -> (B, C, 1, 1)

        # Reshape for Conv1d: (B, C, 1, 1) -> (B, C) -> (B, 1, C)
        # Apply 1D convolution
        # Reshape back to (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion and normalization
        y = self.sigmoid(y)

        # Scale the input feature map by learned channel weights
        return x * y.expand_as(x)

# --- Auxiliary Functions ---
# Original conv3x3 from your CSPNet code, without dilation for compatibility
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False) # Changed padding to 1 for standard 3x3 conv

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Linear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# --- Core Building Blocks with ECA ---

class ECABasicBlock(nn.Module): # Renamed for clarity in combined model
    expansion = 1
    tran_expansion = 1 # Added for CSP compatibility

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, k_size=3): # Added k_size
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ECABasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ECABasicBlock")
        
        self.conv1_layer = conv3x3(inplanes, planes, stride)
        self.bn1_layer = norm_layer(planes)
        self.relu_act = nn.ReLU(inplace=True)
        self.conv2_layer = conv3x3(planes, planes)
        self.bn2_layer = norm_layer(planes)
        self.eca = eca_layer(planes, k_size) # ECA module here
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_layer(x)
        out = self.bn1_layer(out)
        out = self.relu_act(out)

        out = self.conv2_layer(out)
        out = self.bn2_layer(out)
        out = self.eca(out) # Apply ECA here

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu_act(out)

        return out


class ECACSPBottleneck(nn.Module): # Renamed for clarity in combined model
    expansion = 2 # CSPBottleneck had expansion=2
    tran_expansion = 4 # CSPBottleneck had tran_expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, k_size=3): # Added k_size
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * 0.25) # Original CSPBottleneck internal width logic

        self.conv1_layer = conv1x1(inplanes, width)
        self.bn1_layer = norm_layer(width)

        self.conv2_layer = conv3x3(width, width, stride)
        self.bn2_layer = norm_layer(width)

        self.conv3_layer = conv1x1(width, planes * self.expansion)
        self.bn3_layer = norm_layer(planes * self.expansion)
        self.eca = eca_layer(planes * self.expansion, k_size) # ECA module here

        self.lrelu_act = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_layer(x)
        out = self.bn1_layer(out)
        out = self.lrelu_act(out)

        out = self.conv2_layer(out)
        out = self.bn2_layer(out)
        out = self.lrelu_act(out)

        out = self.conv3_layer(out)
        out = self.bn3_layer(out)
        out = self.eca(out) # Apply ECA here

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.lrelu_act(out)

        return out


class CSPBlock(nn.Module):
    def __init__(self, block, inplanes, blocks, stride=1, downsample=None, norm_layer=None, activation=None, k_size=3): # Added k_size
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if activation is None:
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = activation()

        self.inplanes = inplanes
        self.norm_layer = norm_layer
        self.k_size = k_size # Store k_size for _make_layer

        self.crossstage_conv = nn.Conv2d(self.inplanes, self.inplanes * 2, kernel_size=1, stride=1, bias=False)
        self.bn_crossstage = norm_layer(self.inplanes * 2)

        if (self.inplanes <= 64):
            self.conv1_layer = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, bias=False)
            self.bn1_layer = norm_layer(self.inplanes)
            self.layer_num = self.inplanes
        else:
            self.conv1_layer = nn.Conv2d(self.inplanes, self.inplanes * 2, kernel_size=1, stride=1, bias=False)
            self.bn1_layer = norm_layer(self.inplanes * 2)
            self.layer_num = self.inplanes * 2

        self.layers_module_list = nn.ModuleList(
            self._make_layer(block, self.inplanes, blocks, stride))

        self.trans_conv = nn.Conv2d(self.inplanes * 2, self.inplanes * 2, kernel_size=1, stride=1, bias=False) # Original was commented out

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
        
        # Determine the correct input channels for the first block in this list
        # This aligns with the output channels of CSPBlock's 'origin' branch
        if self.inplanes <= 64:
            input_channels_for_first_block_in_list = self.inplanes
        else:
            input_channels_for_first_block_in_list = self.inplanes * 2 


        if stride != 1 or input_channels_for_first_block_in_list != planes * current_expansion:
            downsample = nn.Sequential(
                conv1x1(input_channels_for_first_block_in_list, planes * current_expansion, stride=stride), # Added stride here
                norm_layer(planes * current_expansion),
            )

        layers = []

        # Pass k_size to the block
        layers.append(block(input_channels_for_first_block_in_list, planes, stride, downsample, norm_layer, k_size=self.k_size))
        
        self.inplanes = planes * current_expansion 

        for _ in range(1, blocks):
            # Pass k_size to the block
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, k_size=self.k_size))

        return layers


class ECA_CSPResNet(nn.Module): # Renamed the main model
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 norm_layer=None, k_size=[3, 3, 3, 3]): # Added k_size parameter
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.k_size = k_size # Store k_size list
        
        self.inplanes = 64

        self.conv1_layer = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_layer = norm_layer(self.inplanes)
        self.lrelu_act = nn.LeakyReLU(inplace=True)

        self.maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        # Pass k_size to CSPBlock
        self.layer1_csp_block = CSPBlock(block, 64, layers[0], stride=1, norm_layer=norm_layer, activation=nn.LeakyReLU, k_size=k_size[0])
        self.part_tran1_seq = self._make_tran(64, block.tran_expansion)

        self.layer2_csp_block = CSPBlock(block, 128, layers[1] - 1, stride=1, norm_layer=norm_layer, activation=Linear, k_size=k_size[1])
        self.part_tran2_seq = self._make_tran(128, block.tran_expansion)

        self.layer3_csp_block = CSPBlock(block, 256, layers[2] - 1, stride=1, norm_layer=norm_layer, activation=Linear, k_size=k_size[2])
        self.part_tran3_seq = self._make_tran(256, block.tran_expansion)

        self.layer4_csp_block = CSPBlock(block, 512, layers[3] - 1, stride=1, norm_layer=norm_layer, activation=nn.LeakyReLU, k_size=k_size[3])

        self.conv2_layer = nn.Conv2d(512 * block.tran_expansion, 512 * 2, kernel_size=1, stride=1, bias=False)
        self.bn2_layer = norm_layer(512 * 2) # Added for consistency
        self.lrelu_conv2 = nn.LeakyReLU(inplace=True) # Added for consistency

        self.avgpool_layer = nn.AdaptiveAvgPool2d((1, 1))

        self.fn_linear = nn.Linear(512 * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # No GhostModule specific initialization needed now

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ECACSPBottleneck): # Changed from CSPBottleneck
                    nn.init.constant_(m.bn3_layer.weight, 0)
                elif isinstance(m, ECABasicBlock): # Changed from BasicBlock
                    nn.init.constant_(m.bn2_layer.weight, 0)

    def _make_tran(self, base, tran_expansion):
        return nn.Sequential(
            conv1x1(base * tran_expansion, base * 2), # Using standard conv1x1
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(base * 2, base * 2, stride=2), # Using standard conv3x3
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(inplace=True)
        )

    def _forward_impl(self, x):
        x = self.conv1_layer(x)
        x = self.bn1_layer(x)
        x = self.lrelu_act(x)
        x = self.maxpool_layer(x)

        x = self.layer1_csp_block(x)
        x = self.part_tran1_seq(x)

        x = self.layer2_csp_block(x)
        x = self.part_tran2_seq(x)

        x = self.layer3_csp_block(x)
        x = self.part_tran3_seq(x)

        x = self.layer4_csp_block(x)

        x = self.conv2_layer(x)
        x = self.bn2_layer(x)
        x = self.lrelu_conv2(x)

        x = self.avgpool_layer(x)
        x = x.view(-1, 512 * 2)
        x = self.fn_linear(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _eca_cspresnet(arch, block, layers, pretrained, model_path, **kwargs):
    model = ECA_CSPResNet(block, layers, **kwargs)
    if pretrained:
        # Assuming checkpoint has exact same keys, if not, state_dict processing needed
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    return model


def eca_csp_resnet50(pretrained=False, model_path="checkpoint.pt", **kwargs):
    return _eca_cspresnet('eca_cspresnet50', ECACSPBottleneck, [2, 2, 3, 2], pretrained, model_path=model_path,
                      **kwargs)


def eca_csp_resnet101(pretrained=False, model_path="checkpoint.pt", **kwargs):
    return _eca_cspresnet('eca_cspresnet101', ECACSPBottleneck, [3, 4, 23, 3], pretrained, model_path=model_path,
                      **kwargs)


def eca_csp_resnet152(pretrained=False, model_path="checkpoint.pt", **kwargs):
    return _eca_cspresnet('eca_cspresnet152', ECACSPBottleneck, [3, 8, 36, 3], pretrained, model_path=model_path,
                      **kwargs)


