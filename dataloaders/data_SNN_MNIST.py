from torchvision import datasets
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, transforms
from torch.utils.data import DataLoader, Subset
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import utils.misc as misc
import pandas as pd
import torch
transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x))
            ])
train_dataset = MNIST('/home/datasets/SNN/', train=True, download=False, transform=transform)
test_dataset = MNIST('/home/datasets/SNN/', train=False, download=False, transform=transform) 

print(train_dataset)