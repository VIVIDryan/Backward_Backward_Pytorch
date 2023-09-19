from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt
import numpy as np

# dataloader arguments
batch_size = 256
data_path = '/home/datasets/SNN/'

dtype = torch.float
DEVICE = torch.device('cuda:2')

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=False, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=False, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

# Network Architecture
num_inputs = 28 * 28
num_hidden = 1000
num_outputs = 10

# Temporal Dynamics
num_steps = 25
beta = 0.95

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(
    data, targets, epoch,
    counter, iter_counter,
        loss_hist, test_loss_hist, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

# Define Network
class SNNNet(nn.Module):
    def __init__(self):
        super().__init__()       
        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# 准确度计算
def compute_accuracy(data_loader):
    total = 0
    correct = 0

    with torch.no_grad():
        net.eval()
        for data, targets in data_loader:
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            test_spk, _ = net(data.view(data.size(0), -1))
            _, predicted = test_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Load the network onto CUDA if available
net = SNNNet().to(DEVICE)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

num_epochs = 50 
loss_hist = []
test_loss_hist = []
train_accuracy_hist = []  # 用于存储训练精度
test_accuracy_hist = []   # 用于存储测试精度
train_loss_hist = []      # 用于存储训练损失

for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data.view(batch_size, -1))

        # 初始化损失并在时间上累加
        loss_val = torch.zeros((1), dtype=dtype, device=DEVICE)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        # 梯度计算 + 权重更新
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # 存储损失历史以供后续绘图
        loss_hist.append(loss_val.item())
        train_loss_hist.append(loss_val.item())  # 记录训练损失

        # 测试集
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(DEVICE)
            test_targets = test_targets.to(DEVICE)

            # 测试集前向传播
            test_spk, test_mem = net(test_data.view(batch_size, -1))

            # 测试集损失
            test_loss = torch.zeros((1), dtype=dtype, device=DEVICE)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # 打印训练/测试损失/精度
            if iter_counter % 50 == 0:
                train_printer(
                    data, targets, epoch,
                    iter_counter, iter_counter,
                    loss_hist, test_loss_hist,
                    test_data, test_targets)
            iter_counter += 1

    # 在每个epoch结束后计算并存储训练和测试精度
    train_accuracy = compute_accuracy(train_loader)
    test_accuracy = compute_accuracy(test_loader)
    train_accuracy_hist.append(train_accuracy)
    test_accuracy_hist.append(test_accuracy)
    print(f"Epoch {epoch}, Train Set Accuracy: {train_accuracy:.2f}%")
    print(f"Epoch {epoch}, Test Set Accuracy: {test_accuracy:.2f}%")
    print("\n")

# 绘制损失曲线和精度曲线
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(loss_hist)
axes[0].plot(test_loss_hist)
axes[0].set_title("Loss Curves")
axes[0].legend(["Train Loss", "Test Loss"])
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Loss")

axes[1].plot(train_accuracy_hist)
axes[1].plot(test_accuracy_hist)
axes[1].set_title("Accuracy Curves")
axes[1].legend(["Train Accuracy", "Test Accuracy"])
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")

axes[2].plot(train_loss_hist)
axes[2].set_title("Train Loss")
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel("Loss")

plt.savefig('imgs/snn.png')

# 打印最终的测试精度
final_test_accuracy = compute_accuracy(test_loader)
print(f"Final Test Set Accuracy: {final_test_accuracy:.2f}%")
