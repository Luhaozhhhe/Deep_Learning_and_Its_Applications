import torch.nn as nn
import torch.nn.functional as F
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class FineTuneNet(nn.Module):
    def __init__(self):
        super(FineTuneNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_drop = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 128)
        self.fc3_drop = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = F.relu(self.fc3(x))
        x = self.fc3_drop(x)
        return self.fc4(x)

model = FineTuneNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

print(model)