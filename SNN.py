from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import time
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt
import numpy as np

from dataloaders.dataset import MNIST_loaders
import os
import sys
import logging
import utils.misc as misc

# dataloader arguments
batch_size = 4096
data_path = '/home/datasets/SNN/'

noise_visualize = True


dtype = torch.float
DEVICE = torch.device('cuda:3')
excu_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

logging.basicConfig(
    level=logging.WARN,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def print_batch_accuracy(data, targets, net, train=False):
    output, _ = net(data.view(batch_size, -1).float())
    ## 这种做法会返回output根据第一维来选出最大值以及他们的索引
    _, idx = output.sum(dim=0).max(1)
    ## 会返回True的比例，即正确的比例 
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")
    return acc

def snnacc(out, targets):
    _, idx = out.sum(dim=0).max(1) 
    acc = np.mean((targets == idx).detach().cpu().numpy())
    return acc

def train_printer(
    data, targets, epoch,
    counter, iter_counter,
        loss_hist, test_loss_hist, test_data, test_targets, net):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, net, train=True)
    print_batch_accuracy(test_data, test_targets, net, train=False)
    print("\n")

class MLP(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        
        self.netwotk = nn.Sequential(
           nn.Linear(num_inputs, num_hidden),
           nn.ReLU(),
           nn.Linear(num_hidden, num_outputs) 
        )
    
    def forward(self, x):
        y = self.netwotk(x)
        return y 
        



# Define Network
class SNNNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta, num_steps):
        super().__init__()       
        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs, bias=False)
        self.lif2 = snn.Leaky(beta=beta)
        self.num_steps = num_steps

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            # 数据集中的所有输入像素应用线性变换;
            cur1 = self.fc1(x)
            # 随着时间的推移对加权输入进行集成，如果满足阈值条件，则发出脉冲;
            spk1, mem1 = self.lif1(cur1, mem1)
            # `fc2`对`lif1`的输出峰值进行线性变换;
            cur2 = self.fc2(spk1)
            #  脉冲神经元层，随着时间的推移整合加权脉冲
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


# 准确度计算
def compute_accuracy(data_loader, net):
    total = 0
    correct = 0

    with torch.no_grad():
        net.eval()
        for data, targets in data_loader:
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            test_spk, test_mem = net(data.view(data.size(0), -1).float())
            _, predicted = test_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def compute_accuracy_mlp(data_iter, net):
    metric = misc.Accumulator(2)
    net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
             # BERT微调所需的(之后将介绍)
                X = [x.to(DEVICE) for x in X]
            else:
                X = X.to(DEVICE)
            y = y.to(DEVICE)
            metric.add(misc.accuracy(net(X.float()), y), y.numel())
    return metric[0] / metric[1]

# def plot_snn(loss_hist, test_loss_hist, train_accuracy_hist,test_accuracy_hist, train_loss_hist, savename = None):
#     base_figpath = 'figures/SNN/'
#     # Plot 
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     axes[0].plot(loss_hist)
#     axes[0].plot(test_loss_hist)
#     axes[0].set_title("Loss Curves")
#     axes[0].legend(["Train Loss", "Test Loss"])
#     axes[0].set_xlabel("Iteration")
#     axes[0].set_ylabel("Loss")
    
#     axes[1].plot(train_accuracy_hist)
#     axes[1].plot(test_accuracy_hist)
#     axes[1].set_title("Accuracy Curves")
#     axes[1].legend(["Train Accuracy", "Test Accuracy"])
#     axes[1].set_xlabel("Epoch")
#     axes[1].set_ylabel("Accuracy (%)")
#     axes[1].annotate(f'best test acc:{max(train_accuracy_hist)};',xy=(1,4),  xytext=(3,6), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='red')
    
    
#     axes[2].plot(train_loss_hist)
#     axes[2].set_title("Train Loss")
#     axes[2].set_xlabel("Iteration")
#     axes[2].set_ylabel("Loss")
#     if savename is not None:
#         path = os.path.join(base_figpath, savename)
#         plt.savefig(path)
#     else:
#         plt.show()

def plot_snn(test_acc_hist,train_acc_hist, train_loss_hist, savename = None):
    base_figpath = 'figures/SNN/'
    style = dict(size=10, color='gray')
    # Plot 
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    axes[0].plot(train_acc_hist)
    axes[0].plot(test_acc_hist)
    axes[0].set_title("Accuracy")
    axes[0].legend(["Train Acc", "Test Acc"])
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Accuracy")
    axes[0].text(10,10,s='best test acc:{:.2f}%;'.format(max(test_acc_hist)), ha='center', **style)
    
    # axes[0].annotate(f'best test acc:{max(train_acc_hist)};',xy=(1,4),  xytext=(3,6), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='red')
    
    axes[1].plot(train_loss_hist)
    axes[1].set_title("Train Loss")
    axes[1].legend(["Train Loss"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    
    if savename is not None:
        path = os.path.join(base_figpath, savename)
        plt.savefig(path)
    else:
        plt.show()
        
        
def snn_experiment():
        # Define a transform
    transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    # mnist_train = datasets.MNIST(data_path, train=True, download=False, transform=transform)
    # mnist_test = datasets.MNIST(data_path, train=False, download=False, transform=transform)

    # train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

    train_loader,test_loader = MNIST_loaders(batch_size=batch_size, transform=transform, SNN =False)
    
    train_loader = train_loader[0]
    
    
    
    num_steps = 25
    # Load the network onto CUDA if available
    net = SNNNet(
            num_inputs = 28*28,
            num_hidden = 3000,
            num_outputs = 10,
            num_steps = 25,
            beta = 0.95,
        ).to(DEVICE)
    # net = net.double()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=6e-4, betas=(0.9, 0.999))

    num_epochs = 40
    train_accuracy_hist = []  # 用于存储训练精度
    test_accuracy_hist = []   # 用于存储测试精度
    train_loss_hist = []      # 用于存储训练损失

    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)
        metrics = misc.Accumulator(3)
        # Minibatch training loop
        for data, targets in train_batch:
            optimizer.zero_grad()
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # forward pass
            net.train()
            spk_rec, mem_rec = net(data.view(batch_size, -1).float())

            # 初始化损失并在时间上累加
            loss_val = torch.zeros((1), dtype=dtype, device=DEVICE)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)

            # 梯度计算 + 权重更新
            
            loss_val.backward()
            optimizer.step()

            # 测试集
            with torch.no_grad():
                metrics.add(
                    loss_val * data.shape[0],
                    snnacc(spk_rec, targets),
                    data.shape[0]
                )
                
                # net.eval()
                # test_data, test_targets = next(iter(test_loader))
                # test_data = test_data.to(DEVICE)
                # test_targets = test_targets.to(DEVICE)

                # # 测试集前向传播
                # test_spk, test_mem = net(test_data.view(batch_size, -1).float())

                # # 测试集损失
                # test_loss = torch.zeros((1), dtype=dtype, device=DEVICE)
                # for step in range(num_steps):
                #     test_loss += loss(test_mem[step], test_targets)
                # test_loss_hist.append(test_loss.item())

                # # 打印训练/测试损失/精度
                # if iter_counter % 10 == 0:
                #     train_printer(
                #         data, targets, epoch,
                #         iter_counter, iter_counter,
                #         loss_hist, test_loss_hist,
                #         test_data, test_targets, net)
            train_l = metrics[0]/metrics[2]
            train_acc = metrics[1]/metrics[2]
            iter_counter += 1

        
        # 在每个epoch结束后计算并存储训练和测试精度
        test_accuracy = compute_accuracy(test_loader, net)
        
        train_accuracy_hist.append(train_acc)
        train_loss_hist.append(train_l)
        test_accuracy_hist.append(test_accuracy)
        logging.warning(f"epoch:{epoch} | train_loss:{train_l} | train_acc:{train_acc} | test_acc:{test_accuracy}\n") 
        
        
    plot_snn(test_accuracy_hist, train_accuracy_hist,train_loss_hist, savename = f'snn_teststatic{excu_time}.png') 

    # 打印最终的测试精度
    final_test_accuracy = test_accuracy
    print(f"Final Test Set Accuracy: {final_test_accuracy:.2f}%")

def mlp_experiment():
        # Define a transform
    transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])    

    train_loader,test_loader = MNIST_loaders(batch_size=batch_size, transform=transform, SNN =True)
    train_loader = train_loader[0]
    
    # Define the network
    net = MLP(
            num_inputs = 40000,
            num_hidden = 1000,
            num_outputs = 10
        ).to(DEVICE) 
    
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = 5e-4 
                                 )
    num_epochs = 100
    train_accuracy_hist = []  # 用于存储训练精度
    test_accuracy_hist = []   # 用于存储测试精度
    train_loss_hist = []      # 用于存储训练损失
    
    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)
        metrics = misc.Accumulator(3)
        # Minibatch training loop
        for data, targets in train_batch:
            optimizer.zero_grad()
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # forward pass
            net.train()
            out = net(data.view(batch_size, -1).float())

            loss_val = loss(out, targets)
        
            # 梯度计算 + 权重更新   
            loss_val.backward()
            optimizer.step()
            
            # 测试集
            with torch.no_grad():
                metrics.add(loss_val * data.shape[0], misc.accuracy(out, targets), data.shape[0])
            train_l = metrics[0] / metrics[2]
            train_acc = metrics[1] / metrics[2]
            
            iter_counter += 1   
        
        test_accuracy = compute_accuracy_mlp(test_loader, net)
        train_loss_hist.append(train_l)
        train_accuracy_hist.append(train_acc)
        test_accuracy_hist.append(test_accuracy)
        logging.warning(f"epoch:{epoch} | train_loss:{train_l} | train_acc:{train_acc} | test_acc:{test_accuracy}\n")

    plot_snn(test_acc_hist=test_accuracy_hist, train_acc_hist=train_accuracy_hist, train_loss_hist=train_loss_hist, savename= f"mlpexp_{excu_time}.png")
    

def snn_fewshotexperiment():
        # Define a transform
    transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    # mnist_train = datasets.MNIST(data_path, train=True, download=False, transform=transform)
    # mnist_test = datasets.MNIST(data_path, train=False, download=False, transform=transform)

    # train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

    train_loader,test_loader = MNIST_loaders(batch_size=batch_size, transform=transform, SNN =True, fixed_number=True, num_subsets=2, amount=20000)
    
    train_loader = train_loader[0]
    
    
    
    num_steps = 25
    # Load the network onto CUDA if available
    net = SNNNet(
            num_inputs = 200*200,
            num_hidden = 1000,
            num_outputs = 10,
            num_steps = 25,
            beta = 0.95,
        ).to(DEVICE)
    # net = net.double()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=6e-4, betas=(0.9, 0.999))

    num_epochs = 100 
    train_accuracy_hist = []  # 用于存储训练精度
    test_accuracy_hist = []   # 用于存储测试精度
    train_loss_hist = []      # 用于存储训练损失

    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)
        metrics = misc.Accumulator(3)
        # Minibatch training loop
        for data, targets in train_batch:
            optimizer.zero_grad()
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # forward pass
            net.train()
            spk_rec, mem_rec = net(data.view(batch_size, -1).float())

            # 初始化损失并在时间上累加
            loss_val = torch.zeros((1), dtype=dtype, device=DEVICE)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)

            # 梯度计算 + 权重更新
            
            loss_val.backward()
            optimizer.step()

            # 测试集
            with torch.no_grad():
                metrics.add(
                    loss_val * data.shape[0],
                    snnacc(spk_rec, targets),
                    data.shape[0]
                )
                
                # net.eval()
                # test_data, test_targets = next(iter(test_loader))
                # test_data = test_data.to(DEVICE)
                # test_targets = test_targets.to(DEVICE)

                # # 测试集前向传播
                # test_spk, test_mem = net(test_data.view(batch_size, -1).float())

                # # 测试集损失
                # test_loss = torch.zeros((1), dtype=dtype, device=DEVICE)
                # for step in range(num_steps):
                #     test_loss += loss(test_mem[step], test_targets)
                # test_loss_hist.append(test_loss.item())

                # # 打印训练/测试损失/精度
                # if iter_counter % 10 == 0:
                #     train_printer(
                #         data, targets, epoch,
                #         iter_counter, iter_counter,
                #         loss_hist, test_loss_hist,
                #         test_data, test_targets, net)
            train_l = metrics[0]/metrics[2]
            train_acc = metrics[1]/metrics[2]
            iter_counter += 1

        
        # 在每个epoch结束后计算并存储训练和测试精度
        test_accuracy = compute_accuracy(test_loader, net)
        
        train_accuracy_hist.append(train_acc)
        train_loss_hist.append(train_l)
        test_accuracy_hist.append(test_accuracy)
        logging.warning(f"epoch:{epoch} | train_loss:{train_l} | train_acc:{train_acc} | test_acc:{test_accuracy}\n") 
        
        
    plot_snn(test_accuracy_hist, train_accuracy_hist,train_loss_hist, savename = f'snn_teststatic_fewshotexperiment{excu_time}.png') 

    # 打印最终的测试精度
    final_test_accuracy = test_accuracy
    print(f"Final Test Set Accuracy: {final_test_accuracy:.2f}%")
   


def mlp_fewshotexperiment():
        # Define a transform
    transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])    

    train_loader,test_loader = MNIST_loaders(batch_size=batch_size, transform=transform, fixed_number=True,SNN=True,num_subsets=2, amount=20000)

    train_loader = train_loader[0]
    
    # Define the network
    net = MLP(
            num_inputs = 40000,
            num_hidden = 1000,
            num_outputs = 10
        ).to(DEVICE) 
    
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = 5e-4 
                                 )
    num_epochs = 100
    train_accuracy_hist = []  # 用于存储训练精度
    test_accuracy_hist = []   # 用于存储测试精度
    train_loss_hist = []      # 用于存储训练损失
    
    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)
        metrics = misc.Accumulator(3)
        # Minibatch training loop
        for data, targets in train_batch:
            optimizer.zero_grad()
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # forward pass
            net.train()
            out = net(data.view(batch_size, -1).float())

            loss_val = loss(out, targets)
        
            # 梯度计算 + 权重更新   
            loss_val.backward()
            optimizer.step()
            
            # 测试集
            with torch.no_grad():
                metrics.add(loss_val * data.shape[0], misc.accuracy(out, targets), data.shape[0])
            train_l = metrics[0] / metrics[2]
            train_acc = metrics[1] / metrics[2]
            
            iter_counter += 1   
        
        test_accuracy = compute_accuracy_mlp(test_loader, net)
        train_loss_hist.append(train_l)
        train_accuracy_hist.append(train_acc)
        test_accuracy_hist.append(test_accuracy)
        logging.warning(f"epoch:{epoch} | train_loss:{train_l} | train_acc:{train_acc} | test_acc:{test_accuracy}\n")

    plot_snn(test_acc_hist=test_accuracy_hist, train_acc_hist=train_accuracy_hist, train_loss_hist=train_loss_hist, savename= f"mlpexpfewshot_{excu_time}.png")


def snn_noisyexperiment():
    def add_gaussian_noise(img, mean=0.0, std=1.0):
        noise = torch.randn(img.size()) * std + mean
        return img + noise
        
    noise_mean = 0 
    noise_std = 0.2
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: add_gaussian_noise(x, mean=noise_mean, std=noise_std)),
        transforms.Normalize((0,), (1,)),
    ])


    train_loader,test_loader = MNIST_loaders(batch_size=batch_size, transform=transform, SNN =True)
    
    train_loader = train_loader[0]
    
    
    
    num_steps = 25
    # Load the network onto CUDA if available
    net = SNNNet(
            num_inputs = 200*200,
            num_hidden = 1000,
            num_outputs = 10,
            num_steps = 25,
            beta = 0.95,
        ).to(DEVICE)
    # net = net.double()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=6e-4, betas=(0.9, 0.999))

    num_epochs = 100 
    train_accuracy_hist = []  # 用于存储训练精度
    test_accuracy_hist = []   # 用于存储测试精度
    train_loss_hist = []      # 用于存储训练损失

    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)
        metrics = misc.Accumulator(3)
        # Minibatch training loop
        for data, targets in train_batch:
            optimizer.zero_grad()
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # forward pass
            net.train()
            spk_rec, mem_rec = net(data.view(batch_size, -1).float())

            # 初始化损失并在时间上累加
            loss_val = torch.zeros((1), dtype=dtype, device=DEVICE)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)

            # 梯度计算 + 权重更新
            
            loss_val.backward()
            optimizer.step()

            # 测试集
            with torch.no_grad():
                metrics.add(
                    loss_val * data.shape[0],
                    snnacc(spk_rec, targets),
                    data.shape[0]
                )
            train_l = metrics[0]/metrics[2]
            train_acc = metrics[1]/metrics[2]
            iter_counter += 1

        
        # 在每个epoch结束后计算并存储训练和测试精度
        test_accuracy = compute_accuracy(test_loader, net)
        
        train_accuracy_hist.append(train_acc)
        train_loss_hist.append(train_l)
        test_accuracy_hist.append(test_accuracy)
        logging.warning(f"epoch:{epoch} | train_loss:{train_l} | train_acc:{train_acc} | test_acc:{test_accuracy}\n") 
        
        
    plot_snn(test_accuracy_hist, train_accuracy_hist,train_loss_hist, savename = f'snn_teststatic_noisyexperiment{excu_time}.png') 

    # 打印最终的测试精度
    final_test_accuracy = test_accuracy
    print(f"Final Test Set Accuracy: {final_test_accuracy:.2f}%")



def Modify_noisy(mean, std):
    def add_gaussian_noise(img, mean=0.0, std=1.0):
        noise = torch.randn(img.size()) * std + mean
        return img + noise

    def visualize_images(clean_images, n_images=3, mean=0.0, std=1.0):
        
        plt.figure(figsize=(10, 4))
        for i in range(n_images):
            # 绘制原始图像
            plt.subplot(2, n_images, i + 1)
            plt.imshow(clean_images[i].reshape(200,200), cmap='gray')
            plt.title("Original")
            plt.axis('off')

            # 绘制加噪声后的图像
            plt.subplot(2, n_images, n_images + i + 1)
            
            noisy_images = add_gaussian_noise(clean_images[i], mean=mean, std=std)
            plt.imshow(noisy_images.reshape(200, 200), cmap='gray')
            plt.title("Noisy")
            plt.axis('off')
        save_path = f'figures/SNN/noiseVisualization/noisevisualization_Mean_{mean}_Std_{std}_{excu_time}.png'
        plt.savefig(save_path)
        plt.close()

    

    noise_mean = mean 
    noise_std = std
   
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])  
    
    train_loader, test_loader = MNIST_loaders(batch_size=batch_size, transform=transform, SNN=True) 


    clean_images, _ = next(iter(train_loader[0]))
    visualize_images(clean_images,n_images=3,mean=noise_mean, std=noise_std)  
    print(f"Finish visualizing the image with noise (mean:{noise_mean}, std:{noise_std})")
    
    



def mlp_noisyexperiment():

    def add_gaussian_noise(img, mean=0.0, std=1.0):
        noise = torch.randn(img.size()) * std + mean
        return img + noise
    
    def visualize_images(clean_images, noisy_images, n_images=3):
        
        plt.figure(figsize=(10, 4))
        for i in range(n_images):
            # 绘制原始图像
            plt.subplot(2, n_images, i + 1)
            plt.imshow(clean_images[i].reshape(200,200), cmap='gray')
            plt.title("Original")
            plt.axis('off')

            # 绘制加噪声后的图像
            plt.subplot(2, n_images, n_images + i + 1)
            plt.imshow(noisy_images[i].reshape(200, 200), cmap='gray')
            plt.title("Noisy")
            plt.axis('off')
        save_path = f'figures/SNN/noiseVisualization/visualization_{excu_time}.png'
        plt.savefig(save_path)
        plt.close()
    
    def create_transform(noise=False, noise_mean=0.0, noise_std=1.0):
        if noise:
            return transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: add_gaussian_noise(x, mean=noise_mean, std=noise_std)),
                transforms.Normalize((0,), (1,))
            ])
        else:
            return transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))
            ])


    
    
    # Define a transform       
    noise_mean = 0 
    noise_std = 1.0
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: add_gaussian_noise(x, mean=noise_mean, std=noise_std)),
        transforms.Normalize((0,), (1,)),
    ])

    transform_noisy = create_transform(noise=True, noise_mean=noise_mean, noise_std=noise_std)
    transform_clean = create_transform(noise=False)

    train_loader_noisy, test_loader_noisy = MNIST_loaders(batch_size=batch_size, transform=transform_noisy, SNN=True)
    train_loader_clean, test_loader_clean = MNIST_loaders(batch_size=batch_size, transform=transform_clean, SNN=True)

    # 取出三对图像进行可视化
    noisy_images, _ = next(iter(train_loader_noisy[0]))
    clean_images, _ = next(iter(train_loader_clean[0]))
    visualize_images(clean_images, noisy_images)

    if noise_visualize:
        return
    train_loader_noisy, test_loader_noisy = MNIST_loaders(batch_size=batch_size, transform=transform_noisy, SNN=True)
    train_loader_clean, test_loader_clean = MNIST_loaders(batch_size=batch_size, transform=transform_clean, SNN=True)




    train_loader,test_loader = MNIST_loaders(batch_size=batch_size, transform=transform, SNN =True)

    train_loader = train_loader_noisy[0]
    test_loader = test_loader_noisy
    
    # Define the network
    net = MLP(
            num_inputs = 40000,
            num_hidden = 1000,
            num_outputs = 10
        ).to(DEVICE) 
    
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = 5e-4 
                                 )
    num_epochs = 100
    train_accuracy_hist = []  # 用于存储训练精度
    test_accuracy_hist = []   # 用于存储测试精度
    train_loss_hist = []      # 用于存储训练损失
    
    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)
        metrics = misc.Accumulator(3)
        # Minibatch training loop
        for data, targets in train_batch:
            optimizer.zero_grad()
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # forward pass
            net.train()
            out = net(data.view(batch_size, -1).float())

            loss_val = loss(out, targets)
        
            # 梯度计算 + 权重更新   
            loss_val.backward()
            optimizer.step()
            
            # 测试集
            with torch.no_grad():
                metrics.add(loss_val * data.shape[0], misc.accuracy(out, targets), data.shape[0])
            train_l = metrics[0] / metrics[2]
            train_acc = metrics[1] / metrics[2]
            
            iter_counter += 1   
        
        test_accuracy = compute_accuracy_mlp(test_loader, net)
        train_loss_hist.append(train_l)
        train_accuracy_hist.append(train_acc)
        test_accuracy_hist.append(test_accuracy)
        logging.warning(f"epoch:{epoch} | train_loss:{train_l} | train_acc:{train_acc} | test_acc:{test_accuracy}\n")

    plot_snn(test_acc_hist=test_accuracy_hist, train_acc_hist=train_accuracy_hist, train_loss_hist=train_loss_hist, savename= f"mlpexpnoisy_{excu_time}.png")


if __name__  == "__main__":
    
    ## Noise Modification
    # Modify_noisy(mean=0, std=0.2)
    
    
   snn_experiment() 
    #snn_fewshotexperiment()
    # snn_noisyexperiment()
    # mlp_experiment()
    #mlp_fewshotexperiment()
    # mlp_noisyexperiment()