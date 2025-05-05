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
parser = argparse.ArgumentParser(description='Train MLP model on MNIST with different learning rates')
parser.add_argument('--model', type=str, default='mlp_myself', choices=supported_models, help='which model to use')
args = parser.parse_args()

# Configuration
DATA_PATH = './data'
BATCH_SIZE = 64
SAVE_PATH = os.path.join('results')
NUM_EPOCHS = 20
LEARNING_RATES = [0.001, 0.01, 0.1, 0.2, 0.5]  # 要测试的学习率列表

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

# Test each learning rate
for lr in LEARNING_RATES:
    print(f"\n{'='*50}")
    print(f"Training with Adam optimizer, learning rate: {lr}")
    print(f"{'='*50}")
    
    # Initialize fresh model for each learning rate
    model = model_class().to(device)
    
    # Initialize Adam optimizer with current learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
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
        'learning_rate': lr,
        'train_accuracy': final_train_acc,
        'valid_accuracy': final_valid_acc,
        'train_loss': final_train_loss,
        'valid_loss': final_valid_loss
    })
    
    # Print summary
    print(f"\nLearning Rate: {lr}")
    print(f"Final Train Accuracy: {final_train_acc:.4f}")
    print(f"Final Valid Accuracy: {final_valid_acc:.4f}")
    print(f"Final Train Loss: {final_train_loss:.4f}")
    print(f"Final Valid Loss: {final_valid_loss:.4f}")

# Print all results in a table format
print("\n\nSummary of Results:")
print("="*80)
print(f"{'Learning Rate':<15} | {'Train Acc':>10} | {'Valid Acc':>10} | {'Train Loss':>10} | {'Valid Loss':>10}")
print("-"*80)
for res in results:
    print(f"{res['learning_rate']:<15.4f} | {res['train_accuracy']:>10.4f} | {res['valid_accuracy']:>10.4f} | {res['train_loss']:>10.4f} | {res['valid_loss']:>10.4f}")
print("="*80)

# Save results to file
results_dir = os.path.join('results', 'optimizer_comparison')
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'lr_optimizer_results.txt'), 'w') as f:
    f.write("Learning Rate | Train Acc | Valid Acc | Train Loss | Valid Loss\n")
    f.write("-"*80 + "\n")
    for res in results:
        f.write(f"{res['learning_rate']:<15.4f} | {res['train_accuracy']:>10.4f} | {res['valid_accuracy']:>10.4f} | {res['train_loss']:>10.4f} | {res['valid_loss']:>10.4f}\n")

print(f"\nResults saved to {os.path.join(results_dir, 'lr_optimizer_results.txt')}")