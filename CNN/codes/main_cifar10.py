import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from train import Trainer
from plot import plot_loss_accuracy
import argparse

from model.CNN_Base import BaseCNN
from model.CNN_ResNet import ResNet_18, ResNet_34, ResNet_50, ResNet_101, ResNet_152
from model.CNN_ResNet import BasicBlock, Bottleneck
from model.CNN_DenseNet import DenseNet_121, DenseNet_169, DenseNet_201, DenseNet_264
from model.CNN_DenseNet import DenseBlock
from model.CNN_SE_ResNet import SE_ResNet18, SE_ResNet34, SE_ResNet50, SE_ResNet101, SE_ResNet152

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
    'cnn_se_resnet152'
]


# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a model on CIFAR10')
parser.add_argument('--model', type=str, default='cnn_base', choices=supported_models, help='which model to use')
args = parser.parse_args()

# Configuration
DATA_PATH = './data'
BATCH_SIZE = 64
SAVE_PATH = os.path.join('results', 'CIFAR10', args.model)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Data Loading & Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

train_dataset = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = train_dataset.classes


if args.model == 'cnn_base':
    model = BaseCNN(num_of_last_layer=10)
elif args.model == 'cnn_resnet18':
    model = ResNet_18(num_classes=10, ResidualBlock=BasicBlock)
elif args.model == 'cnn_resnet34':
    model = ResNet_34(num_classes=10, ResidualBlock=BasicBlock)
elif args.model == 'cnn_resnet50':
    model = ResNet_50(num_classes=10, ResidualBlock=Bottleneck)
elif args.model == 'cnn_resnet101':
    model = ResNet_101(num_classes=10, ResidualBlock=Bottleneck)
elif args.model == 'cnn_resnet152':
    model = ResNet_152(num_classes=10, ResidualBlock=Bottleneck)
elif args.model == 'cnn_densenet121':
    model = DenseNet_121(num_classes=10)
elif args.model == 'cnn_densenet169':
    model = DenseNet_169(num_classes=10)
elif args.model == 'cnn_densenet201':
    model = DenseNet_201(num_classes=10)
elif args.model == 'cnn_densenet264':
    model = DenseNet_264(num_classes=10)
elif args.model == 'cnn_se_resnet18':
    model = SE_ResNet18(num_classes=10)
elif args.model == 'cnn_se_resnet34':
    model = SE_ResNet34(num_classes=10)
elif args.model == 'cnn_se_resnet50':
    model = SE_ResNet50(num_classes=10)
elif args.model == 'cnn_se_resnet101':
    model = SE_ResNet101(num_classes=10)
elif args.model == 'cnn_se_resnet152':
    model = SE_ResNet152(num_classes=10)




criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 50
trainer = Trainer(model, criterion, optimizer, device)
train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list = trainer.train(
    train_loader, valid_loader=test_loader, num_epochs=num_epochs
)

# Evaluation
test_loss, test_accuracy = trainer.evaluate(test_loader)

# Save Model
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'model.pth'))
print('Model Saved')

# Visualization
plot_loss_accuracy(train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, save_path=SAVE_PATH)
