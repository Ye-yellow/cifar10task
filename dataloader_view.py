from models import *
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torch.utils.tensorboard import SummaryWriter
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, drop_last=True)

writer = SummaryWriter('./tensorboard')

step = 0
for data in train_loader:
    images, labels = data
    writer.add_images('batch_img', images, step)
    step += 1

writer.close()
