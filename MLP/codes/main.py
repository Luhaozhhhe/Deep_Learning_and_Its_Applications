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

from model.MLP_Base import BaseNet
from model.MLP_MySelf import FineTuneNet
from model.MLP_Mixer import MixerNet

supported_models = [
    'mlp_base',
    'mlp_myself',
    'mlp_mixer'
]

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MLP model on MNIST')
parser.add_argument('--model', type=str, default='mlp', choices=supported_models, help='which model to use')
args = parser.parse_args()

# Configuration
DATA_PATH = './data'
BATCH_SIZE = 64
SAVE_PATH = os.path.join('results', args.model)

# Configuration for MLP-Mixer
PATCH_SIZE = 7
NUM_PATCHES = 16
HIDDEN_DIM = 256

# Check for GPU
if torch.cuda.is_available():
    device = 'cuda'


# Data Loading & Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

train_dataset = datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
classes = [str(i) for i in range(10)]
print('Classes: {}'.format(classes))

# Model & Optimization
if args.model == 'mlp_base':
    model = BaseNet()
elif args.model == 'mlp_myself':
    model = FineTuneNet()
elif args.model == 'mlp_mixer':
    model = MixerNet()

else:
    raise NotImplementedError

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(model)

# Training
num_epochs = 20
trainer = Trainer(model, criterion, optimizer, device)
train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list = trainer.train(
    train_loader, valid_loader=test_loader, num_epochs=num_epochs
)

# Evaluation
test_loss, test_accuracy= trainer.evaluate(test_loader)

# Save Model
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'model.pth'))
print('Model Saved!')

# Visualization
plot_loss_accuracy(train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, save_path=SAVE_PATH)

