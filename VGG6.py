import torch
import torch.nn as nn
import torch.nn.functional as functonal
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#VGG6
#We will be creating 3 Conv layers and 3 Fully connected layers
#
class VGG6(nn.Module):
  def __init__(self, activationFunction):
    super().__init__()
    class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    self.activationFunction = activationFunction
    #create first convilution layer
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1)
    self.pool  = nn.MaxPool2d(kernel_size=(2,2))
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1)
    self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1)

    self.fc1 = nn.Linear(4*4*256, 512)
    self.fc2 = nn.Linear(512, 512)
    self.fc3 = nn.Linear(512, len(class_names)) #map it to class which we are providing
  def forward(self, x):
    match self.activationFunction:
      case 'Relu':
        x = self.pool(functonal.relu(self.conv1(x)))
        x = self.pool(functonal.relu(self.conv2(x)))
        x = self.pool(functonal.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = functonal.relu(self.fc1(x))
        x = functonal.relu(self.fc2(x))
        x = self.fc3(x)

      case 'Sigmoid':
        x = self.pool(functonal.sigmoid(self.conv1(x)))
        x = self.pool(functonal.sigmoid(self.conv2(x)))
        x = self.pool(functonal.sigmoid(self.conv3(x)))
        #x = torch.flatten(x, 1)
        x = torch.flatten(x, start_dim=1)
        x = functonal.sigmoid(self.fc1(x))
        x = functonal.sigmoid(self.fc2(x))
        x = self.fc3(x)

      case 'Tanh':
        x = self.pool(functonal.tanh(self.conv1(x)))
        x = self.pool(functonal.tanh(self.conv2(x)))
        x = self.pool(functonal.tanh(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = functonal.tanh(self.fc1(x))
        x = functonal.tanh(self.fc2(x))
        x = self.fc3(x)

      case 'SiLU':
        x = self.pool(functonal.silu(self.conv1(x)))
        x = self.pool(functonal.silu(self.conv2(x)))
        x = self.pool(functonal.silu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = functonal.silu(self.fc1(x))
        x = functonal.silu(self.fc2(x))
        x = self.fc3(x)

      case 'GELU':
        x = self.pool(functonal.gelu(self.conv1(x)))
        x = self.pool(functonal.gelu(self.conv2(x)))
        x = self.pool(functonal.gelu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = functonal.gelu(self.fc1(x))
        x = functonal.gelu(self.fc2(x))
        x = self.fc3(x)
    return x