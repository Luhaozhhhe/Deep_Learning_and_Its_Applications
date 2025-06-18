from pre import NameDataset, collate_fn
from train import Trainer
from plot import plot_loss_accuracy, plot_confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os
import argparse
from model.LSTM_OurSelves import CustomLSTM


supported_models = [
    'rnn',
    'lstm',
    'lstm_ourselves',
]

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train RNN Model on Name-Language Classification')
parser.add_argument('--model', type=str, default='lstm', choices=supported_models, help='which model to use')
args = parser.parse_args()


# Configuration
DATA_PATH = './data/names'
TRAIN_PROP = 0.8
BATCH_SIZE = 64
SAVE_PATH = os.path.join('results', args.model)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


dataset = NameDataset(DATA_PATH)
train_size = int(TRAIN_PROP * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])  # Train/Valid Dataset Split
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Model
input_size = len(dataset.all_letters)
hidden_size = 128
num_layers = 1
num_classes = len(dataset.all_categories)

if args.model == 'rnn':
    from model.RNN import RNN as Model
elif args.model == 'lstm':
    from model.LSTM import LSTM as Model
elif args.model == 'lstm_ourselves':
    from model.LSTM_OurSelves import CustomLSTM as Model

else:
    raise NotImplementedError

model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
print(model)

# Training Scheduler
num_epochs = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
trainer = Trainer(model, criterion, optimizer, device)
train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list = trainer.train(train_loader, valid_loader, num_epochs)

# Evaluation
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loss, test_accuracy, conf_mat = trainer.evaluate(test_loader, verbose=True)

# Save Model
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'model.pth'))
print('Model Saved')

# Visualization
plot_loss_accuracy(train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, save_path=SAVE_PATH)
plot_confusion_matrix(conf_mat, dataset.all_categories, normalize=True, save_path=SAVE_PATH)
print('Confusion Matrix Saved')