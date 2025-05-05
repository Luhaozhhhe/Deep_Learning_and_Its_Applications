import torch.nn as nn
import torch.nn.functional as F
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class MixerNet(nn.Module):
    def __init__(self, image_size=28, channels=1, patch_size=7, hidden_dim=512, num_classes=10):
        super(MixerNet, self).__init__()

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(channels * patch_size * patch_size, hidden_dim)
        self.fc2 = nn.Linear(self.num_patches, self.num_patches)
        self.fc3 = nn.Linear(self.num_patches, self.num_patches)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        N, C, H, W = x.size()

        patch_size = self.patch_size
        unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        x = unfold(x)
        x = x.transpose(1, 2)

        x = self.fc1(x)

        y = x.transpose(1, 2)
        y = F.gelu(self.fc2(y))
        y = F.gelu(self.fc3(y))
        y = y.transpose(1, 2)
        x = x + y

        y = F.gelu(self.fc4(x))
        y = F.gelu(self.fc5(y))
        x = x + y

        x = x.mean(dim=1)

        x = self.fc6(x)
        return x

model = MixerNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

print(model)