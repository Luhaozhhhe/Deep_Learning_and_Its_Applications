import os
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.Hor_gMLP import Hor_gMLPNet # 新增 Hor-gMLP 模型

import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 辅助脚本/模块
from train import Trainer
from plot import plot_loss_accuracy

# 模型定义
# 基础CNN模型
from model.CNN_Base import BaseCNN

# ResNet系列模型
from model.CNN_ResNet import ResNet_18, ResNet_34, ResNet_50, ResNet_101, ResNet_152
from model.CNN_ResNet import BasicBlock, Bottleneck

# DenseNet系列模型
from model.CNN_DenseNet import DenseNet_121, DenseNet_169, DenseNet_201, DenseNet_264
from model.CNN_DenseNet import DenseBlock

# SE-ResNet系列模型
from model.CNN_SE_ResNet import SE_ResNet18, SE_ResNet34, SE_ResNet50, SE_ResNet101, SE_ResNet152
from model.SE_Wide_ResNet import create_enhanced_wideresnet

# MLP类模型
from model.CNN_gMLP import gMLPVision
from model.Hor_gMLP import Hor_gMLPNet # 新增 Hor-gMLP 模型

# ConvNeXt系列模型
from model.convnext import ConvNeXt, convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
from timm.models import create_model # 通常用于加载预训练模型

# HorNet系列模型
from model.CNN_HorNet import HorNet_T, HorNet_S, HorNet_B, HorNet_GF_T

# CoAtNet系列模型
from model.CNN_CoAtNet import CoAtNet_0, CoAtNet_1, CoAtNet_2, CoAtNet_3, CoAtNet_Tiny

# CSPNet系列模型
from model.CNN_CSPNet import csp_resnet50, csp_resnet101, csp_resnet152

# GhostNet模型
from model.CNN_ghostNet import ghostnet

# LSKNet模型
# from model.CNN_LSKNet import LSKNeta

# SegNeXt模型
from model.CNN_SegNeXt import SegNet

# 混合/变体模型
from model.convnext_gmlp import ConvNeXt_gMLP
from model.CNN_SE_ConvNeXt import se_convnext_tiny
from model.CNN_ECAResNet import eca_resnet18, eca_resnet34, eca_resnet50, eca_resnet101, eca_resnet152
from model.CNN_ECA_CSPResNet import eca_csp_resnet50, eca_csp_resnet101, eca_csp_resnet152

from model.LSK_Split_Net import LSKSplitNet
from model.LSKHorNet import LSKHorNet
from model.Moe_ResNet import MoEResNet18
from model.LSKHorNet_Without_Attention import LSKHOResNet
from model.resnet_split_hor import HOResNet
from model.ResNet_Split import SplitAttentionResNet

# 新的混合模型
from model.CNN_ResNet18_HorNet import ResNet18HorNet_Tiny, ResNet18HorNet_Small, ResNet18HorNet_Base
from model.CNN_ResNet18_CoAtNet import ResNet18CoAtNet_Tiny, ResNet18CoAtNet_Small, ResNet18CoAtNet_Base

from model.CNN_ResNet18_SplitCoAtNet import (
    ResNet18SplitCoAtNet_Tiny,
    ResNet18SplitCoAtNet_Small,
    ResNet18SplitCoAtNet_Base
)

from model.CNN_WideResNet18_HorNet import (
    WideResNet18HorNet_Tiny,
    WideResNet18HorNet_Small,
    WideResNet18HorNet_Base
)

from model.CNN_ResNet18_HorNet_SplitAttn import (
    ResNet18HorNetSplitAttn_Tiny,
    ResNet18HorNetSplitAttn_Small,
    ResNet18HorNetSplitAttn_Base
)

from model.Wide_ResNet import WideResNet


# hornet + SE
from model.CNN_ResNet18_HorNet_SE import ResNet18HorNetSE_Tiny, ResNet18HorNetSE_Small, ResNet18HorNetSE_Base

# 
from model.CNN_ResNet18_improved import ResNet_18 as ImporvedResNet18

from model.convnext_isotropic import ConvNeXtIsotropic

supported_models = [
    'cnn_base',
    'cnn_resnet18',
    'cnn_resnet34',
    'cnn_resnet50',
    'cnn_resnet101',
    'cnn_resnet152',
    'cnn_densenet121',
    'cnn_densenet169',
    'cnn_densenet201',
    'cnn_densenet264',
    'cnn_se_resnet18',
    'cnn_se_resnet34',
    'cnn_se_resnet50',
    'cnn_se_resnet101',
    'cnn_se_resnet152',
    'cnn_gmlp',
    'convnext_tiny',    # 新增 ConvNeXt Tiny
    'convnext_small',   # 新增 ConvNeXt Small
    'convnext_base',    # 新增 ConvNeXt Base
    'convnext_large',   # 新增 ConvNeXt Large
    'convnext_xlarge',   # 新增 ConvNeXt XLarge
    'cnn_csp_resnet50',
    'cnn_csp_resnet101',
    'cnn_csp_resnet152',
    'cnn_ghostnet',
    'hornet_tiny',      # 新增 HorNet Tiny
    'hornet_small',     # 新增 HorNet Small
    'hornet_base',      # 新增 HorNet Base
    'hornet_tiny_gf',   # 新增 HorNet Tiny GF
    'coatnet_0',        # 新增 CoAtNet-0
    'coatnet_1',        # 新增 CoAtNet-1 
    'coatnet_2',        # 新增 CoAtNet-2
    'coatnet_3',        # 新增 CoAtNet-3
    'coatnet_tiny',     # 新增 CoAtNet-Tiny
    'cnn_LSKNet',
    'cnn_SegNeXt',
    'convnext_tiny_gmlp',   # 新增 ConvNeXt Tiny + gMLP
    'convnext_tiny_se',
    'cnn_csp_ghost_resnet50',
    'cnn_csp_ghost_resnet101',
    'cnn_csp_ghost_resnet152',
    'horcoatnet_tiny',  # 新增 HorCoAtNet Tiny
    'horcoatnet_small', # 新增 HorCoAtNet Small
    'horcoatnet_base',  # 新增 HorCoAtNet Base
    'eca_resnet18',
    'eca_resnet34',
    'eca_resnet50',
    'eca_resnet101',
    'eca_resnet152',
    'eca_csp_resnet50',
    'eca_csp_resnet101',
    'eca_csp_resnet152',

    'cnn_LSK_SplitNet',
    'LSKHorNet',        # 新增 LSKHorNet
    'moe_resnet18',
    'LSKHOResNet',
    'resnet_split_hor',
    'resnet_split',     # 新增 Split-Attention ResNet
    'resnet18hornet_tiny',  # 新增 ResNet18-HorNet Tiny
    'resnet18hornet_small', # 新增 ResNet18-HorNet Small
    'resnet18hornet_base',  # 新增 ResNet18-HorNet Base
    'resnet18coatnet_tiny', # 新增 ResNet18-CoAtNet Tiny
    'resnet18coatnet_small',# 新增 ResNet18-CoAtNet Small
    'resnet18coatnet_base', # 新增 ResNet18-CoAtNet Base
    'Dilated_ResNet',   # 新增 Dilated ResNet
    'resnet18_split_coat_tiny',
    'resnet18_split_coat_small',
    'resnet18_split_coat_base',
    'Wide_ResNet',      # 新增 Wide ResNet
    'SE_Wide_ResNet',   # 新增 SE-Wide ResNet
    'wideresnet18hornet_tiny',
    'wideresnet18hornet_small',
    'wideresnet18hornet_base',
    'resnet18hornet_splitattn_tiny',
    'resnet18hornet_splitattn_small',
    'resnet18hornet_splitattn_base',

    # 新增 ResNet18-HorNet-SE 模型
    'resnet18hornetse_tiny',
    'resnet18hornetse_small',
    'resnet18hornetse_base',
    'cnn_resnet18_imporved',  # 新增改进版 ResNet18

    'convnext_isotropic_small',
    'convnext_isotropic_base',
    'convnext_isotropic_large',
    'hor_gmlp_tiny',  # 新增 Hor-gMLP Tiny
]


# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a model on CIFAR100')
parser.add_argument('--model', type=str, default='cnn_base', choices=supported_models, help='which model to use')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained ConvNeXt model (ImageNet-1K or 22K)') # 新增参数，用于加载预训练权重
parser.add_argument('--in_22k', action='store_true', help='If using pretrained, load ImageNet-22K weights (for ConvNeXt Tiny/Small/Base/Large) or only ImageNet-22K (for ConvNeXt XLarge)') # 新增参数
args = parser.parse_args()

# Configuration
DATA_PATH = './data'
BATCH_SIZE = 64
SAVE_PATH = os.path.join('results', args.model)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Data Loading & Preprocessing
# !!! 注意：ConvNeXt 通常在 224x224 或 384x384 图像上训练。CIFAR100 图像是 32x32。
# !!! ConvNeXt 的 stem 层有 4x4 的卷积核和 4 的步长，这意味着它期望更大的输入。
# !!! 为 ConvNeXt 添加图像缩放，并使用 ImageNet 的均值和标准差进行归一化，以匹配预训练模型的输入要求。

# if args.model.startswith('convnext'):
#     # ConvNeXt 推荐的均值和标准差以及输入大小
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     input_size = 224 # ConvNeXt 常用输入尺寸，您可以根据需要调整

#     transform = transforms.Compose([
#         # transforms.Resize((input_size, input_size)), # 缩放到 ConvNeXt 期望的尺寸
#         # transforms.ToTensor(),
#         # transforms.Normalize(mean, std)
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
if args.model.startswith('convnext') or args.model in ['convnext_tiny_gmlp', 'convnext_isotropic_small', 'convnext_isotropic_base', 'convnext_isotropic_large']:
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]
    mean = [0.5071, 0.4866, 0.4409]
    std = [0.2009, 0.1984, 0.2023]
    input_size = 32  # 推荐从 64×64 开始，防止特征图过小
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),  # 保守放大
        transforms.RandomHorizontalFlip(),            # 数据增强
        transforms.RandomCrop(input_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
elif args.model.startswith('hornet'):
    # HorNet也使用ImageNet的均值和标准差
    mean = [0.5071, 0.4866, 0.4409]
    std = [0.2009, 0.1984, 0.2023]
    # 修改stem层后，可以直接使用32x32的图像，也可以适当放大
    input_size = 32
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
elif args.model.startswith('coatnet'):
    # CoAtNet使用ImageNet的均值和标准差
    mean = [0.5071, 0.4866, 0.4409]
    std = [0.2009, 0.1984, 0.2023]
    # CoAtNet已经适配32x32输入
    input_size = 32
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
elif args.model.startswith('resnet18hornet'):
    # ResNet18-HorNet混合模型使用ImageNet的均值和标准差
    mean = [0.5071, 0.4866, 0.4409]
    std = [0.2009, 0.1984, 0.2023]
    input_size = 32
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

elif args.model.startswith('resnet18coatnet'):
    # ResNet18-CoAtNet混合模型使用ImageNet的均值和标准差
    mean = [0.5071, 0.4866, 0.4409]
    std = [0.2009, 0.1984, 0.2023]
    input_size = 32
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
else:
    # 现有模型的 transforms
    mean = [0.5071, 0.4866, 0.4409]
    std = [0.2009, 0.1984, 0.2023]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# 设置日志
log_filename = os.path.join(SAVE_PATH, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO, # 如要调试，可以改为 logging.DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 记录训练配置
logger.info(f"Model: {args.model}")
logger.info(f"Batch Size: {BATCH_SIZE}")
logger.info(f"Device: {device}")
# --- 模型信息记录 ---
# 记录输入尺寸，这是大多数模型都需要的通用信息

if args.model.startswith('convnext'):
    logger.info(f"Input Size: {input_size}x{input_size}")

    logger.info(f"ConvNeXt model variant: {args.model}")
    logger.info(f"Using pretrained: {args.pretrained}")
    if args.pretrained:
        logger.info(f"Using 22K weights: {args.in_22k}")
elif args.model.startswith('hornet'):
    logger.info(f"Input Size: {input_size}x{input_size}")

    logger.info(f"HorNet model variant: {args.model}")
elif args.model.startswith('coatnet'):
    logger.info(f"Input Size: {input_size}x{input_size}")

    logger.info(f"CoAtNet model variant: {args.model}")
elif args.model.startswith('resnet18hornet'):
    logger.info(f"Input Size: {input_size}x{input_size}")

    logger.info(f"ResNet18-HorNet model variant: {args.model}")
elif args.model.startswith('resnet18coatnet'):
    logger.info(f"Input Size: {input_size}x{input_size}")

    logger.info(f"ResNet18-CoAtNet model variant: {args.model}")
else:
    logger.info(f"Using general model: {args.model}")

train_dataset = datasets.CIFAR100(root=DATA_PATH, train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.CIFAR100(root=DATA_PATH, train=False, download=True, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_indices = list(range(10, 20)) + list(range(50, 60)) + list(range(80, 90))

train_indices = [i for i in range(len(train_dataset)) if train_dataset.targets[i] in class_indices]
test_indices = [i for i in range(len(test_dataset)) if test_dataset.targets[i] in class_indices]

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)


classes = train_dataset.classes


# 定义常用常量，避免重复
NUM_CLASSES = 100
BASIC_BLOCK = BasicBlock
BOTTLENECK = Bottleneck

model = None # 初始化 model 变量，确保它总被赋值

# --- ConvNeXt 系列模型（使用 timm.create_model，需要特殊处理） ---
# 由于 ConvNeXt 系列使用 timm 的 create_model 且参数复杂，优先处理
if args.model.startswith('convnext') and args.model != "convnext_tiny_se":
    convnext_configs = {
        'convnext_tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768], 'drop_path_rate': 0.1},
        'convnext_small': {'depths': [3, 3, 27, 3], 'dims': [96, 192, 384, 768], 'drop_path_rate': 0.4},
        'convnext_base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024], 'drop_path_rate': 0.5},
        'convnext_large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536], 'drop_path_rate': 0.5},
        'convnext_xlarge': {'depths': [3, 3, 27, 3], 'dims': [256, 512, 1024, 2048], 'drop_path_rate': 0.2},
    }
    model_name = args.model
    config = convnext_configs.get(model_name)

    if config:
        model = create_model(
            model_name,
            pretrained=args.pretrained,
            in_22k=args.in_22k,
            num_classes=NUM_CLASSES,
            drop_path_rate=config['drop_path_rate'],
            layer_scale_init_value=1e-2,
            head_init_scale=1.0
        )
    else:
        raise ValueError(f"ConvNeXt model variant '{model_name}' not found in configurations.")

# --- 混合/特殊参数模型（这些模型有独特的参数列表或复杂的初始化逻辑） ---
# 独立处理，因为它们的构造函数参数与其他模型差异较大
elif args.model == 'cnn_gmlp':
    model = gMLPVision(image_size=32, patch_size=4, num_classes=NUM_CLASSES, dim=512, depth=12, heads=8, ff_mult=4, channels=3, prob_survival=0.9)
elif args.model == 'hor_gmlp_tiny':
    model = Hor_gMLPNet(
        image_size=32,      # CIFAR-100为例
        patch_size=4,
        in_chans=3,
        num_classes=100,
        dim=64,
        depth=12,           # 层数你可以自定义
        mlp_ratio=4,
        gn_order=5,
        drop_path=0.1
    )
elif args.model == 'convnext_tiny_gmlp':
    model = ConvNeXt_gMLP(
        in_chans=3, num_classes=NUM_CLASSES, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
        drop_path_rate=0.1, layer_scale_init_value=1e-2, head_init_scale=1.0
    )
elif args.model == 'convnext_tiny_se': # 假设这里是 se_convnext_tiny
    model = se_convnext_tiny(num_classes=NUM_CLASSES)
elif args.model == 'cnn_LSKNet':
    model = LSKNet(
        in_channels=3, num_classes=NUM_CLASSES, embed_dims=[64, 128, 256, 512],
        mlp_ratios=[4, 4, 4, 4], depths=[2, 2, 6, 2], drop_path_rate=0.1
    )
elif args.model == 'cnn_LSK_SplitNet':
    model = LSKSplitNet(
        in_channels=3, num_classes=NUM_CLASSES, embed_dims=[64, 128, 256, 512],
        mlp_ratios=[4, 4, 4, 4], depths=[2, 2, 6, 2], drop_path_rate=0.1, groups=2
    )
elif args.model == 'LSKHorNet':
    model = LSKHorNet(
        in_channels=3, num_classes=NUM_CLASSES, embed_dims=[64, 128, 256, 512],
        mlp_ratios=[4, 4, 4, 4], depths=[2, 2, 6, 2], drop_path_rate=0.1, groups=2, orders=[5, 4, 3, 2]
    )
elif args.model == 'moe_resnet18':
    model = MoEResNet18(num_classes=NUM_CLASSES, num_experts=4)
elif args.model == 'resnet_split':
    model = SplitAttentionResNet(num_classes=NUM_CLASSES, groups=2)


# --- ResNet 系列模型（需要指定 ResidualBlock） ---
# 将 ResNet 系列放在一起处理
elif args.model == 'cnn_resnet18':
    model = ResNet_18(num_classes=NUM_CLASSES, ResidualBlock=BASIC_BLOCK)
elif args.model == 'cnn_resnet34':
    model = ResNet_34(num_classes=NUM_CLASSES, ResidualBlock=BASIC_BLOCK)
elif args.model == 'cnn_resnet50':
    model = ResNet_50(num_classes=NUM_CLASSES, ResidualBlock=BOTTLENECK)
elif args.model == 'cnn_resnet101':
    model = ResNet_101(num_classes=NUM_CLASSES, ResidualBlock=BOTTLENECK)
elif args.model == 'cnn_resnet152':
    model = ResNet_152(num_classes=NUM_CLASSES, ResidualBlock=BOTTLENECK)

# --- 其他系列模型（主要参数是 num_classes） ---
# 将大部分只接收 num_classes 作为参数的模型放在一起
elif args.model == 'cnn_base':
    model = BaseCNN(num_of_last_layer=NUM_CLASSES) # 注意这里参数名不同
elif args.model == 'cnn_densenet121':
    model = DenseNet_121(num_classes=NUM_CLASSES)
elif args.model == 'cnn_densenet169':
    model = DenseNet_169(num_classes=NUM_CLASSES)
elif args.model == 'cnn_densenet201':
    model = DenseNet_201(num_classes=NUM_CLASSES)
elif args.model == 'cnn_densenet264':
    model = DenseNet_264(num_classes=NUM_CLASSES)
elif args.model == 'cnn_se_resnet18':
    model = SE_ResNet18(num_classes=NUM_CLASSES)
elif args.model == 'cnn_se_resnet34':
    model = SE_ResNet34(num_classes=NUM_CLASSES)
elif args.model == 'cnn_se_resnet50':
    model = SE_ResNet50(num_classes=NUM_CLASSES)
elif args.model == 'cnn_se_resnet101':
    model = SE_ResNet101(num_classes=NUM_CLASSES)
elif args.model == 'cnn_se_resnet152':
    model = SE_ResNet152(num_classes=100)
elif args.model == 'cnn_gmlp':
    model = gMLPVision(image_size=32,patch_size=4,num_classes=100,dim=512,depth=12,heads=8,ff_mult=4,channels=3,prob_survival=0.9)
elif args.model == 'hor_gmlp_tiny':
    model = Hor_gMLPNet(
        image_size=32,      # CIFAR-100为例
        patch_size=4,
        in_chans=3,
        num_classes=100,
        dim=64,
        depth=12,           # 层数你可以自定义
        mlp_ratio=4,
        gn_order=5,
        drop_path=0.1
    )
elif args.model == 'convnext_tiny_gmlp':
    model = ConvNeXt_gMLP(
        in_chans=3,
        num_classes=100,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.1,
        layer_scale_init_value=1e-2,
        head_init_scale=1.0
    )
elif args.model == 'convnext_tiny_se':
    model = convnext_tiny_se(num_classes=100)


# 新增 ConvNeXt 模型实例化
elif args.model.startswith('convnext'):
    # 获取 ConvNeXt 模型默认参数的字典
    model_configs = {
        'convnext_tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768], 'drop_path_rate': 0.1}, #
        'convnext_small': {'depths': [3, 3, 27, 3], 'dims': [96, 192, 384, 768], 'drop_path_rate': 0.4}, #
        'convnext_base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024], 'drop_path_rate': 0.5}, #
        'convnext_large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536], 'drop_path_rate': 0.5}, #
        'convnext_xlarge': {'depths': [3, 3, 27, 3], 'dims': [256, 512, 1024, 2048], 'drop_path_rate': 0.2}, #
    }
    
    model_name = args.model # 例如 'convnext_tiny'
    config = model_configs.get(model_name)

    NUM_CLASSES=100

    if config:
        model = create_model( # 使用 timm 的 create_model 更方便
            model_name,
            pretrained=args.pretrained,
            in_22k=args.in_22k,
            num_classes=NUM_CLASSES, # 设置为 CIFAR100 的类别数
            drop_path_rate=config['drop_path_rate'],
            layer_scale_init_value=1e-2, # 默认值，可在 args 中自定义
            head_init_scale=1.0 # 默认值，可在 args 中自定义
        )
    else:
        raise ValueError(f"Model {model_name} not found in ConvNeXt configurations.")
elif args.model == 'cnn_csp_resnet50':
    model = csp_resnet50(num_classes=NUM_CLASSES)
elif args.model == 'cnn_csp_resnet101':
    model = csp_resnet101(num_classes=NUM_CLASSES)
elif args.model == 'cnn_csp_resnet152':
    model = csp_resnet152(num_classes=NUM_CLASSES)
elif args.model == 'cnn_ghostnet':
    model = ghostnet(num_classes=NUM_CLASSES)
elif args.model == 'hornet_tiny':
    model = HorNet_T(num_classes=NUM_CLASSES)
elif args.model == 'hornet_small':
    model = HorNet_S(num_classes=NUM_CLASSES)
elif args.model == 'hornet_base':
    model = HorNet_B(num_classes=NUM_CLASSES)
elif args.model == 'hornet_tiny_gf':
    model = HorNet_GF_T(num_classes=NUM_CLASSES)
elif args.model == 'coatnet_0':
    model = CoAtNet_0(num_classes=NUM_CLASSES)
elif args.model == 'coatnet_1':
    model = CoAtNet_1(num_classes=NUM_CLASSES)
elif args.model == 'coatnet_2':
    model = CoAtNet_2(num_classes=NUM_CLASSES)
elif args.model == 'coatnet_3':
    model = CoAtNet_3(num_classes=NUM_CLASSES)
elif args.model == 'coatnet_tiny':
    model = CoAtNet_Tiny(num_classes=NUM_CLASSES)
elif args.model == 'cnn_SegNeXt':
    model = SegNet(input_channels=3, output_channels=NUM_CLASSES) # 注意这里参数名不同
elif args.model == 'eca_resnet18':
    model = eca_resnet18(num_classes=NUM_CLASSES)
elif args.model == 'eca_resnet34':
    model = eca_resnet34(num_classes=NUM_CLASSES)
elif args.model == 'eca_resnet50':
    model = eca_resnet50(num_classes=NUM_CLASSES)
elif args.model == 'eca_resnet101':
    model = eca_resnet101(num_classes=NUM_CLASSES)
elif args.model == 'eca_resnet152':
    model = eca_resnet152(num_classes=NUM_CLASSES)
elif args.model == 'eca_csp_resnet50':
    model = eca_csp_resnet50(num_classes=NUM_CLASSES)
elif args.model == 'eca_csp_resnet101':
    model = eca_csp_resnet101(num_classes=NUM_CLASSES)
elif args.model == 'eca_csp_resnet152':
    model = eca_csp_resnet152(num_classes=NUM_CLASSES)
elif args.model == 'LSKHOResNet':
    model = LSKHOResNet(num_classes=NUM_CLASSES)
elif args.model == 'resnet_split_hor':
    model = HOResNet(num_classes=NUM_CLASSES)
elif args.model == 'resnet18hornet_tiny':
    model = ResNet18HorNet_Tiny(num_classes=NUM_CLASSES)
elif args.model == 'resnet18hornet_small':
    model = ResNet18HorNet_Small(num_classes=NUM_CLASSES)
elif args.model == 'resnet18hornet_base':
    model = ResNet18HorNet_Base(num_classes=NUM_CLASSES)
elif args.model == 'resnet18coatnet_tiny':
    model = ResNet18CoAtNet_Tiny(num_classes=NUM_CLASSES)
elif args.model == 'resnet18coatnet_small':
    model = ResNet18CoAtNet_Small(num_classes=NUM_CLASSES)
elif args.model == 'resnet18coatnet_base':
    model = ResNet18CoAtNet_Base(num_classes=NUM_CLASSES)
# 在模型选择部分添加
elif args.model == 'resnet18_split_coat_tiny':
    model = ResNet18SplitCoAtNet_Tiny(num_classes=100)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200,
        eta_min=1e-6
    )

elif args.model == 'resnet18_split_coat_small':
    model = ResNet18SplitCoAtNet_Small(num_classes=100)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200,
        eta_min=1e-6
    )

elif args.model == 'resnet18_split_coat_base':
    model = ResNet18SplitCoAtNet_Base(num_classes=100)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200,
        eta_min=1e-6
    )
elif args.model == 'Wide_ResNet':
    model = WideResNet(depth=28, widen_factor=10, num_classes=100)
elif args.model == 'SE_Wide_ResNet':
    model = create_enhanced_wideresnet(
        depth=28,            # 网络深度
        width=10,           # 宽度因子
        num_classes=100     # 类别数
    )

elif args.model == 'resnet18hornet_splitattn_tiny':
    model = ResNet18HorNetSplitAttn_Tiny(num_classes=100)
elif args.model == 'resnet18hornet_splitattn_small':
    model = ResNet18HorNetSplitAttn_Small(num_classes=100)
elif args.model == 'resnet18hornet_splitattn_base':
    model = ResNet18HorNetSplitAttn_Base(num_classes=100)

# 在模型选择部分添加
elif args.model == 'wideresnet18hornet_tiny':
    model = WideResNet18HorNet_Tiny(num_classes=100)

elif args.model == 'wideresnet18hornet_small':
    model = WideResNet18HorNet_Small(num_classes=100)

elif args.model == 'wideresnet18hornet_base':
    model = WideResNet18HorNet_Base(num_classes=100)

else:
    # 如果没有匹配的模型，抛出错误
    raise ValueError(f"Unknown model type specified: {args.model}")

# 记录当前使用的模型
logger.info(f"Initialized model: {args.model}")
# 检查模型是否已成功实例化
if model is None:
    raise ValueError(f"Model {args.model} could not be instantiated. Check your --model argument.")





criterion = nn.CrossEntropyLoss()

# ConvNeXt 论文中推荐 AdamW 优化器，这里沿用您现有的 Adam，但可以考虑切换以获得更好性能（可以考虑改一下）
if args.model.startswith('convnext'):
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05) # 示例 AdamW
elif args.model.startswith('hornet'):
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05) # 示例 AdamW
# elif args.model.startswith('hornet'):
    # HorNet论文建议使用AdamW优化器，权重衰减0.05
#    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
# elif args.model.startswith('coatnet'):
    # CoAtNet也使用AdamW优化器
#    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
elif args.model == 'cnn_resnet18_improved':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
else:
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # 




# Training
num_epochs = 50

# 根据模型自动选择是否加调度器


scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
trainer = Trainer(model, criterion, optimizer, device, logger, scheduler=scheduler)
(train_loss_list, train_top1_accuracy_list, train_top5_accuracy_list,
 valid_loss_list, valid_top1_accuracy_list, valid_top5_accuracy_list) = trainer.train(
    train_loader, valid_loader=test_loader, num_epochs=num_epochs
)


# Evaluation
test_loss, test_top1_accuracy, test_top5_accuracy = trainer.evaluate(test_loader)
logger.info(f"Final Test Results:")
logger.info(f"Test Loss: {test_loss:.4f}")
logger.info(f"Top-1 Accuracy: {test_top1_accuracy:.2f}%")
logger.info(f"Top-5 Accuracy: {test_top5_accuracy:.2f}%")



# Save Model
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'model.pth'))
logger.info('Model Saved')


# Visualization
plot_loss_accuracy(
    train_loss_list, valid_loss_list,
    train_top1_accuracy_list, train_top5_accuracy_list,
    valid_top1_accuracy_list, valid_top5_accuracy_list,
    save_path=SAVE_PATH
)