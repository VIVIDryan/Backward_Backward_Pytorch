from torchvision import datasets
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import utils.misc as misc
import pandas as pd
import torch
import random
import numpy as np

def ESC50_loaders(path='/home/datasets/SNN/ESC50/', batch_size=64, num_subsets=1):
    '''
    num_subsets：训练集划分数量
    description: 用于生成ESC-50的train_loader, test_loader。每个音频样本的有220500点，一共50类
    return {*}: train_loader, test_loader
    '''    
    path_audio = path + 'audio/audio/'
    data = pd.read_csv(path + 'esc50.csv')
    
    train_data = misc.DataGenerator(path_audio, kind='train')
    test_data = misc.DataGenerator(path_audio, kind='test')
    
    # Calculate the size of each subset
    subset_size = len(train_data) // num_subsets
    
    train_loaders = []
    for i in range(num_subsets):
        # Calculate the starting and ending indices for each subset
        start_idx = i * subset_size
        end_idx = start_idx + subset_size
        
        # Create a subset of the train_data using the indices
        subset = Subset(train_data, list(range(start_idx, end_idx)))
        
        # Create a DataLoader for each subset
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return train_loaders, test_loader


def MSD_loaders(path='/home/datasets/SNN/MSD/', batch_size=64, num_subsets=1):
    '''
    num_subsets：训练集划分数量
    description: 生成MSD的dataloader, 一共6类
    return: 
    '''
    train_dir = path + 'train'
    test_dir = path + 'test'
    valid_dir = path + 'valid'
    
    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    
    data = {
        'train': datasets.ImageFolder(root=train_dir, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_dir, transform=image_transforms['valid'])
    }
    
    # Calculate the size of each subset
    subset_size = len(data['train']) // num_subsets
    
    train_loaders = []
    for i in range(num_subsets):
        # Calculate the starting and ending indices for each subset
        start_idx = i * subset_size
        end_idx = start_idx + subset_size
        
        # Create a subset of the train dataset using the indices
        subset = Subset(data['train'], list(range(start_idx, end_idx)))
        
        # Create a DataLoader for each subset
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    
    valid_loader = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    
    return train_loaders, test_loader, valid_loader
class CreateMNISTSNNDataset(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, index):
        image, label = self.mnist_dataset[index]

        # 将图像放置在100x100的黑色像素中
        image = self.place_mnist_in_black_background(image)

        return image, label

    def place_mnist_in_black_background(self, image):
        # 创建一个100x100的黑色像素画布
        background = np.zeros((200, 200))

        # 随机选择图像的位置
        x = random.randint(0, 200 - 28)  # 28是MNIST图像的宽度
        y = random.randint(0, 200 - 28)  # 28是MNIST图像的高度

        # 将图像复制到黑色背景中的随机位置
        background[y:y+28, x:x+28] = image

        return background


def MNIST_loaders(batch_size=50000, num_subsets=1, transform=None, fixed_number = False, amount = 10000, SNN = False):
    '''
    num_subsets: 训练集划分数量
    description: 输入batch_size和需要划分的数量
    fixed_number: 是否固定每个client中的数据集数量
    amount: 数据集数量
    return {*}
    ''' 
    if transform is None:  
        transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x))
            ])

    train_dataset = MNIST('/home/datasets/SNN/', train=True, download=False, transform=transform)
    test_dataset = MNIST('/home/datasets/SNN/', train=False, download=False, transform=transform)
    if SNN:
            train_dataset =CreateMNISTSNNDataset( MNIST('/home/datasets/SNN/', train=True, download=False, transform=transform))
            test_dataset = CreateMNISTSNNDataset(MNIST('/home/datasets/SNN/', train=False, download=False, transform=transform))
    train_loaders = []
    # Calculate the size of each subset
    if fixed_number:
        subset_size =amount
        num_samples = len(train_dataset)
        if num_subsets > 1:
            step = (num_samples - num_subsets * subset_size) // (num_subsets - 1)
        else:
            step = 0  # 当 num_subsets 为 1 时，无需间隔步长
        train_loaders = []
        start_idx = 0
        for i in range(num_subsets):
            # Calculate the ending index for each subset
            end_idx = min(start_idx + subset_size, num_samples)

            # Create a subset of the train_dataset using the indices
            subset = Subset(train_dataset, list(range(start_idx, end_idx)))

            # Create a DataLoader for each subset
            train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)
            train_loaders.append(train_loader)

            # Update the starting index for the next subset
            start_idx = end_idx + step
    else:
        subset_size = len(train_dataset) // num_subsets
        for i in range(num_subsets):
            # Calculate the starting and ending indices for each subset
            start_idx = i * subset_size
            end_idx = start_idx + subset_size
            # Create a subset of the train_dataset using the indices
            subset = Subset(train_dataset, list(range(start_idx, end_idx)))
            # Create a DataLoader for each subset
            train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)
            train_loaders.append(train_loader)

    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=True, drop_last=True)

    return train_loaders, test_loader




def debug_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])
    ## transform处就展平了

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


# def MNIST_SNN_loaders(batch_size=50000, num_subsets=1, transform=None, fixed_number = False, amount = 10000):
#     '''
#     num_subsets: 训练集划分数量
#     description: 输入batch_size和需要划分的数量
#     fixed_number: 是否固定每个client中的数据集数量
#     amount: 数据集数量
#     return {*}
#     ''' 
#     if transform is None:  
#         transform = Compose([
#             ToTensor(),
#             Normalize((0.1307,), (0.3081,)),
#             Lambda(lambda x: torch.flatten(x))
#             ])

#     train_dataset =CreateMNISTSNNDataset( MNIST('/home/datasets/SNN/', train=True, download=False, transform=transform))
#     test_dataset = CreateMNISTSNNDataset(MNIST('/home/datasets/SNN/', train=False, download=False, transform=transform))
#     train_loaders = []
#     # Calculate the size of each subset
#     if fixed_number:
#         subset_size =amount
#         num_samples = len(train_dataset)
#         if num_subsets > 1:
#             step = (num_samples - num_subsets * subset_size) // (num_subsets - 1)
#         else:
#             step = 0  # 当 num_subsets 为 1 时，无需间隔步长
#         train_loaders = []
#         start_idx = 0
#         for i in range(num_subsets):
#             # Calculate the ending index for each subset
#             end_idx = min(start_idx + subset_size, num_samples)

#             # Create a subset of the train_dataset using the indices
#             subset = Subset(train_dataset, list(range(start_idx, end_idx)))

#             # Create a DataLoader for each subset
#             train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)
#             train_loaders.append(train_loader)

#             # Update the starting index for the next subset
#             start_idx = end_idx + step
#     else:
#         subset_size = len(train_dataset) // num_subsets
#         for i in range(num_subsets):
#             # Calculate the starting and ending indices for each subset
#             start_idx = i * subset_size
#             end_idx = start_idx + subset_size
#             # Create a subset of the train_dataset using the indices
#             subset = Subset(train_dataset, list(range(start_idx, end_idx)))
#             # Create a DataLoader for each subset
#             train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)
#             train_loaders.append(train_loader)

#     test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=True, drop_last=True)

#     return train_loaders, test_loader




if __name__ == '__main__':
    train, test = MNIST_loaders(batch_size=10000, num_subsets=1, transform=None, fixed_number = True, amount = 20000)
    print(len(test))
