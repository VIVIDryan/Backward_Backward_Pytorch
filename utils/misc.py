from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchaudio
import torch
from collections.abc import Iterable

import numpy as np


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    首先，函数创建了输入数据 x 的一个副本 x_,以确保在不改变原始数据的情况下进行修改。
    接着，函数将 x_ 的前10个像素位置(第0列至第9列)全部设置为0.0,意味着将前10个像素清零。
    然后，函数使用标签 y 来创建一个 one-hot 编码表示，将 y 对应的位置设为 x 的最大值。例如，如果标签 y 为 3,则会将 x_ 中第3个像素位置(索引为 3)设置为 x 的最大值。
    最后，函数返回修改后的数据 x_,其中标签信息已被叠加在前10个像素位置上,其余像素与输入数据 x 保持不变。
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

def Conv_overlay_y_on_x(x, y):
    """
    Replace the first 10 pixels of image data [x] with one-hot-encoded label [y]
    """
    
    # 复制 x 以保持输入不变
    x_ = x.clone()
    
    if isinstance(y, Iterable):
    
        for i,j in zip(x_, y):
            i[0,0,:10] =0
            i[0,0, j] = x_.max()
    else:
        for i in x_:
            i[0,0,:10] =0
            i[0,0, y] = x_.max() 
          
    return x_

def create_hybrid_image(image1, image2):
    # Create a mask with large regions of ones and zeros
    mask = np.zeros_like(image1)
    mask[10:18, 10:18] = 1
    mask[20:25, 20:25] = 1

    # Blur the mask with a filter of the form [1/4, 1/2, 1/4] in both directions
    filter = np.array([1/4, 1/2, 1/4])
    for i in range(10):
        mask = convolve(mask, filter[np.newaxis, :],
                        mode='constant')  # Note the np.newaxis
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

    # Create hybrid images for negative data
    negative_data = image1 * mask + image2 * (1 - mask)
    return negative_data

class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        # 定义一个包含n个数据的列表
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def avg(self):
        return [sum(a)/len(a) for a in self.data]

    def __getitem__(self, idx):
        return self.data[idx]

def snn_accuracy(data, targets, net, batchsize):
    """
    return the acc of a batch in snn
    """
    output, _ = net(data.view(batchsize, -1 ))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())
    return acc

def accuracy(y_hat, y):  # @save

    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:

        y_hat = y_hat.argmax(axis=1)

    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class DataGenerator(Dataset):
 
    def __init__(self, path, kind='train'):
        if kind=='train':
            files = Path(path).glob('[1-3]-*')
        if kind=='val':
            files = Path(path).glob('4-*')
        if kind=='test':
            files = Path(path).glob('[4-5]-*')
        
        self.items = [(str(file), file.name.split('-')[-1].replace('.wav', '')) for file in files]
        self.length = len(self.items)
        
    def __getitem__(self, index):
        filename, label = self.items[index]
        data_tensor, rate = torchaudio.load(filename)
        return (data_tensor, int(label))
    
    def __len__(self):
        return self.length
    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
def visualize_negative(data, name='', idx=0):
    reshaped = data.reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()

def split_non_iid_data(dataset, num_subsets):
    """
    Split a dataset into Non-IID subsets.

    Args:
    - dataset: The PyTorch dataset to be split.
    - num_subsets: The number of Non-IID subsets to create.

    Returns:
    - A list of datasets, each containing a Non-IID subset.
    """
    num_samples = len(dataset)
    samples_per_subset = num_samples // num_subsets


    non_iid_subsets = []
    start_idx = 0

    for i in range(num_subsets):
        end_idx = start_idx + samples_per_subset
        subset = torch.utils.data.Subset(dataset, list(range(start_idx, end_idx)))
        non_iid_subsets.append(subset)

        start_idx = end_idx

    return non_iid_subsets

def plot_loss(loss):
    # plot the loss over epochs
    fig = plt.figure()
    plt.plot(list(range(len(loss))), loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Plot")
    plt.savefig("Loss Plot.png")
    plt.show()


if __name__ == "__main__":
    import numpy as np
    from scipy.ndimage import convolve

    # Create a random bit image
    random_image = np.random.randint(0, 2, size=(28, 28))
    visualize_negative(random_image)
    # Create a mask with large regions of ones and zeros
    mask = np.zeros_like(random_image)
    mask[10:18, 10:18] = 1
    mask[20:25, 20:25] = 1

    # Blur the mask with a filter of the form [1/4, 1/2, 1/4] in both directions
    filter = np.array([1/4, 1/2, 1/4])
    for i in range(10):
        mask = convolve(mask, filter[np.newaxis, :],
                        mode='constant')  # Note the np.newaxis
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

    # Create hybrid images for negative data
    digit1 = np.random.randint(0, 10, size=(28, 28))
    digit2 = np.random.randint(0, 10, size=(28, 28))
 