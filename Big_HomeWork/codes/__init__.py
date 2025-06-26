import torch
import torchvision
import torchvision.transforms as transforms

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Is GPU available?: ', torch.cuda.is_available()) #查看GPU是否可用
print('Using PyTorch version:', torch.__version__, ' Device:', device) #查看PyTorch版本和设备
print('Current GPU device id:', torch.cuda.current_device()) #查看当前GPU设备id
print('Current GPU device name:', torch.cuda.get_device_name(0)) #查看当前GPU设备名称