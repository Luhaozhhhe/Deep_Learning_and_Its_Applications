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

from torch.optim import lr_scheduler as lr_scheduler, swa_utils as swa_utils
from torch.optim._adafactor import Adafactor as Adafactor
from torch.optim.adadelta import Adadelta as Adadelta
from torch.optim.adagrad import Adagrad as Adagrad
from torch.optim.adam import Adam as Adam
from torch.optim.adamax import Adamax as Adamax
from torch.optim.adamw import AdamW as AdamW
from torch.optim.asgd import ASGD as ASGD
from torch.optim.lbfgs import LBFGS as LBFGS
from torch.optim.nadam import NAdam as NAdam
from torch.optim.optimizer import Optimizer as Optimizer
from torch.optim.radam import RAdam as RAdam
from torch.optim.rmsprop import RMSprop as RMSprop
from torch.optim.rprop import Rprop as Rprop
from torch.optim.sgd import SGD as SGD
from torch.optim.sparse_adam import SparseAdam as SparseAdam


from model.MLP_Base import BaseNet
from model.MLP_MySelf import FineTuneNet
from model.MLP_Mixer import MixerNet

supported_models = [
    'mlp_base',
    'mlp_myself',
    'mlp_mixer'
]

# All supported optimizers
supported_optimizers = {
    'Adagrad': optim.Adagrad,
    'Adam': optim.Adam,
    'Rprop': optim.Rprop,
    'SGD': optim.SGD,

}

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MLP model on MNIST with different optimizers')
parser.add_argument('--model', type=str, default='mlp_myself', choices=supported_models, help='which model to use')
args = parser.parse_args()

# Configuration
DATA_PATH = './data'
BATCH_SIZE = 64
SAVE_PATH = os.path.join('results')
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

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

# Initialize model
if args.model == 'mlp_base':
    model_class = BaseNet
elif args.model == 'mlp_myself':
    model_class = FineTuneNet
elif args.model == 'mlp_mixer':
    model_class = MixerNet
else:
    raise NotImplementedError

# Results storage
results = []

# Test each optimizer
for opt_name, opt_class in supported_optimizers.items():
    print(f"\n{'='*50}")
    print(f"Training with optimizer: {opt_name}")
    print(f"{'='*50}")
    
    # Initialize fresh model for each optimizer
    model = model_class().to(device)
    
    # Initialize optimizer
    try:
        optimizer = opt_class(model.parameters(), lr=LEARNING_RATE)
    except TypeError:
        # Some optimizers might require additional parameters
        optimizer = opt_class(model.parameters())
    
    criterion = nn.CrossEntropyLoss()
    
    # Training
    trainer = Trainer(model, criterion, optimizer, device)
    train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list = trainer.train(
        train_loader, valid_loader=test_loader, num_epochs=NUM_EPOCHS
    )
    
    # Get final epoch results
    final_train_acc = train_accuracy_list[-1]
    final_valid_acc = valid_accuracy_list[-1]
    final_train_loss = train_loss_list[-1]
    final_valid_loss = valid_loss_list[-1]
    
    # Store results
    results.append({
        'optimizer': opt_name,
        'train_accuracy': final_train_acc,
        'valid_accuracy': final_valid_acc,
        'train_loss': final_train_loss,
        'valid_loss': final_valid_loss
    })
    
    # Print summary
    print(f"\nOptimizer: {opt_name}")
    print(f"Final Train Accuracy: {final_train_acc:.4f}")
    print(f"Final Valid Accuracy: {final_valid_acc:.4f}")
    print(f"Final Train Loss: {final_train_loss:.4f}")
    print(f"Final Valid Loss: {final_valid_loss:.4f}")

# Print all results in a table format
print("\n\nSummary of Results:")
print("="*80)
print(f"{'Optimizer':<10} | {'Train Acc':>10} | {'Valid Acc':>10} | {'Train Loss':>10} | {'Valid Loss':>10}")
print("-"*80)
for res in results:
    print(f"{res['optimizer']:<10} | {res['train_accuracy']:>10.2f}% | {res['valid_accuracy']:>10.2f}% | {res['train_loss']:>10.4f} | {res['valid_loss']:>10.4f}")
print("="*80)

# Save results to file
results_dir = os.path.join('results', 'optimizer_comparison')
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'optim_optimizer_results.txt'), 'w') as f:
    f.write("Optimizer | Train Acc | Valid Acc | Train Loss | Valid Loss\n")
    f.write("-"*80 + "\n")
    for res in results:
        f.write(f"{res['optimizer']:<10} | {res['train_accuracy']:>10.2f}% | {res['valid_accuracy']:>10.2f}% | {res['train_loss']:>10.4f} | {res['valid_loss']:>10.4f}\n")

print(f"\nResults saved to {os.path.join(results_dir, f'{args.model}_optimizer_results.txt')}")