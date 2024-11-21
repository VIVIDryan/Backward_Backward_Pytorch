import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
import time
import logging
import copy
import sys
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as tortra
import numpy as np

from networks.Model import CNNNet, SNNv1, SNNv2
from dataloaders.dataset import MNIST_loaders, debug_loaders
from utils import misc
from utils import federated

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.WARN,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)




DEVICE = torch.device('cuda:3')
config = {
    'lr': 5e-3,
    'epoch': 1,
    'globalepoch': 50,
    'batchsize': 1024,
    'num_steps':25
    }


def FederatedCNN_experiment(num_of_clients):
    """
    ANN implement 
    """
    config = {
    'lr': 5e-2,
    'epoch': 3,
    'globalepoch': 50,
    'batchsize': 1024,
    }
    num_of_clients = num_of_clients
    
    transform = tortra.Compose([
            tortra.ToTensor(),
            tortra.Normalize((0.1307,), (0.3081,))
            ])
    train_loader, test_loader = MNIST_loaders(batch_size=config['batchsize'], transform=transform, num_subsets=num_of_clients)
    client_step = [iter(_) for _ in train_loader]
    
    server_model = CNNNet(10)
    client_models = [copy.copy(server_model) for _ in range(num_of_clients)]
    client_optims = [torch.optim.Adam(client_models[i].parameters(), lr = config['lr'], weight_decay=0.01) for i in range(len(client_models))]
    loss = nn.CrossEntropyLoss()
    
    
    FF_start_time = time.time()
    train_acc = []
    epoch = 0
    
    outertqdm = tqdm(range(config['globalepoch']), desc=f"Global Epoch", position=0)
    
    train_accuracy_hist = []  # 用于存储训练精度
    test_accuracy_hist = []   # 用于存储测试精度
    train_loss_hist = []      # 用于存储训练损失
    client_accuracies = [[] for _ in range(num_of_clients)]
    
    for epoch in outertqdm:
        print(f"Global epoch {epoch+1}")
        inertqdm = tqdm(train_loader, desc=f"Local client", position=1, leave=False)
        
        local_avg_acc = []
        for i, iterator in enumerate(inertqdm):
            client_models[i].train()
            metric = misc.Accumulator(3)
            client_models[i] = client_models[i].to(DEVICE)
            train_acc_list = []
            
            for data in iterator:
                x, y = data
                x, y = x.to(DEVICE), y.to(DEVICE)
                client_optims[i].zero_grad()
                y_hat = client_models[i](x)
                l = loss(y_hat, y)
                l.backward()
                client_optims[i].step()
                
                with torch.no_grad():
                    metric.add(l * x.shape[0], misc.accuracy(y_hat, y), x.shape[0])
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                train_acc_list.append(train_acc)
                
            client_accuracies[i].append(sum(train_acc_list)/len(train_acc_list))
            
            local_avg_acc.append(sum(train_acc_list)/len(train_acc_list))
            
            print(f"client {i}: trainacc {sum(train_acc_list)/len(train_acc_list)}")         

        
        client_models, server_model = federated.FedAvg(client_models, server_model)
        
        test_acc = []
        testmetrics = misc.Accumulator(2) 
        server_model = server_model.to(DEVICE)
        for data in test_loader:
            
            x, y = data
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                testmetrics.add(misc.accuracy(server_model(x), y), x.shape[0]) 
            
            acc = metric[0] / metric[1] 
            test_acc.append(acc)
        
        avg_testacc = sum(test_acc)/len(test_acc)
        test_accuracy_hist.append(avg_testacc)    
        print(f"GLobal epoch: test acc {avg_testacc}\n")
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, config['globalepoch'] + 1), test_accuracy_hist, marker='o', linestyle='-')
    plt.title("Avg Test Accuracy vs. Global Epoch")
    plt.xlabel("Global Epoch")
    plt.ylabel("Avg Test Accuracy")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for i in range(num_of_clients):
        plt.plot(range(1, config['globalepoch'] + 1), client_accuracies[i], label=f"Client {i}", marker='o', linestyle='-')
    plt.title("Local Avg Accuracy vs. Global Epoch (Per Client)")
    plt.xlabel("Global Epoch")
    plt.ylabel("Local Avg Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # 确保子图不重叠
    plt.savefig('figures/FederatedCNN/FederatedSNN_G_{0}_L_{1}_lr_{2}_Client_{3}.png'.format(config['globalepoch'], config['epoch'], config['lr'], num_of_clients))    
    
    
def FederatedSNNv2_experiment(num_of_clients):
    
    config = {
    'lr': 5e-2,
    'epoch': 1,
    'globalepoch': 50,
    'batchsize': 1024,
    'num_steps':20
    }
    # config['globalepoch'], config['epoch'], config['lr'], num_of_clients
    print('Global epoch:{0} \
          Local epoch:{1}\
          Learning rate:{2}'
          .format(config['globalepoch'], config['epoch'], config['lr'], num_of_clients))
     
    
    num_of_clients = num_of_clients
    
    transform = tortra.Compose([
                tortra.Resize((28, 28)),
                tortra.Grayscale(),
                tortra.ToTensor(),
                tortra.Normalize((0,), (1,))])
    
    server_model = SNNv2([28*28, 1000, 10])
    client_models = [copy.copy(server_model) for _ in range(num_of_clients)]
    client_optims = [torch.optim.Adam(client_models[i].parameters(), lr = config['lr'], betas=(0.9, 0.999)) for i in range(len(client_models))]
    loss = nn.CrossEntropyLoss()
    
    train_acc = []
    epoch = 0
    # outertqdm = tqdm(range(config['globalepoch']), desc=f"Global Epoch", position=0)
    
    global_test_acc_list = []  # 用于存储每个globalepoch的avg_testacc
    local_avg_acc_lists = [[] for _ in range(num_of_clients)]  # 每个客户端的local_avg_acc随着globalepoch的变化
    
    
    for epoch in range(config['globalepoch']):
        train_loader, test_loader = MNIST_loaders(batch_size=config['batchsize'], transform=transform, num_subsets=num_of_clients)

        logging.warning(f"Global epoch {epoch+1}")
        for i , iterator in enumerate(train_loader):
            loss_hist = []
            train_loss_hist = []      # 用于存储训练损失
            client_models[i] = client_models[i].to(DEVICE)
            train_acc_list = [] # save the average acc of local epochs
            
            
            for epoch in range(config['epoch']):
                train_batch = iter(iterator)

                metric = misc.Accumulator(3)
                
                
                # Minibatch training loop
                for data, targets in train_batch:
                    data = data.to(DEVICE)
                    targets = targets.to(DEVICE)
                    
                    # mem_rec: [time_step, batchsize, classnum]
                    # forward pass
                    spk_rec, mem_rec = client_models[i](data.view(config['batchsize'], -1))

                    # 初始化损失并在时间上累加
                    loss_val = torch.zeros((1), dtype=torch.float, device=DEVICE)
                    for step in range(config['num_steps']):
                        loss_val += loss(mem_rec[step], targets)

                    # 梯度计算 + 权重更新
                    client_optims[i].zero_grad()
                    loss_val.backward()
                    client_optims[i].step()

                    # 存储损失历史以供后续绘图
                    loss_hist.append(loss_val.item())
                    train_loss_hist.append(loss_val.item())  # 记录训练损失

                    # 测试集
                    with torch.no_grad():
                        client_models[i].eval()
                        metric.add(loss_val.sum(), 
                                   misc.snn_accuracy(data, targets, client_models[i],config['batchsize']),
                                  1 
                                   )
                
                train_acc = metric[1]/metric[2]
                train_acc_list.append(train_acc)
                
            local_avg_acc = sum(train_acc_list)/len(train_acc_list)
            local_avg_acc_lists[i].append(local_avg_acc)
            logging.warning(f"client {i}: trainacc {local_avg_acc}\n")
                             

        
        client_models, server_model = federated.FedAvg(client_models, server_model)
        
        server_model = server_model.to(DEVICE)
        testmetrics = misc.Accumulator(2)
        with torch.no_grad():
            server_model.eval()
            for data in test_loader:
                x, y = data
                x, y = x.to(DEVICE), y.to(DEVICE)
                testmetrics.add(misc.snn_accuracy(x, y, server_model, config['batchsize']), 1)
            test_acc = testmetrics[0]/testmetrics[1]
        global_test_acc_list.append(test_acc)    
        writer.add_scalar('FederatedSNN/globaltestacc', test_acc, epoch) 
        logging.warning(f"GLobal epoch: test acc {test_acc}\n")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, config['globalepoch'] + 1), global_test_acc_list, marker='o', linestyle='-')
    plt.title("Avg Test Accuracy vs. Global Epoch")
    plt.xlabel("Global Epoch")
    plt.ylabel("Avg Test Accuracy")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for i in range(num_of_clients):
        plt.plot(range(1, config['globalepoch'] + 1), local_avg_acc_lists[i], label=f"Client {i}", marker='o', linestyle='-')
    plt.title("Local Avg Accuracy vs. Global Epoch (Per Client)")
    plt.xlabel("Global Epoch")
    plt.ylabel("Local Avg Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # 确保子图不重叠
    plt.savefig('figures/FederatedSNN/FederatedSNN_G_{0}_L_{1}_lr_{2}_Client_{3}.png'.format(config['globalepoch'], config['epoch'], config['lr'], num_of_clients)) 


def FederatedSNN_experiment(num_of_clients):
    """
    SNN implemented in LFL
    """
    num_of_clients = num_of_clients
    
    transform = tortra.Compose([
            tortra.ToTensor(),
            tortra.Normalize((0.1307,), (0.3081,))
            ])
    train_loader, test_loader = MNIST_loaders(batch_size=config['batchsize'], transform=transform, num_subsets=num_of_clients)
    client_step = [iter(_) for _ in train_loader]
    
    server_model = SNNv1([28*28*1, 500, 10],config['batchsize'])
    client_models = [copy.copy(server_model) for _ in range(num_of_clients)]
    client_optims = [torch.optim.Adam(client_models[i].parameters(), lr = config['lr'], weight_decay=0.01) for i in range(len(client_models))]
    loss = nn.CrossEntropyLoss()
    
    
    FF_start_time = time.time()
    train_acc = []
    epoch = 0
    
    outertqdm = tqdm(range(config['globalepoch']), desc=f"Global Epoch", position=0)
    
    for epoch in outertqdm:
        print(f"Global epoch {epoch+1}")
        inertqdm = tqdm(train_loader, desc=f"Local client", position=1, leave=False)
        
        local_avg_acc = []
        for i, iterator in enumerate(inertqdm):
            client_models[i].train()
            metric = misc.Accumulator(3)
            client_models[i] = client_models[i].to(DEVICE)
            train_acc_list = []
            
            for data in iterator:
                x, y = data
                x, y = x.to(DEVICE), y.to(DEVICE)
                client_optims[i].zero_grad()
                y_hat = client_models[i](x)
                l = loss(y_hat, y)
                l.backward()
                client_optims[i].step()
                
                with torch.no_grad():
                    metric.add(l * x.shape[0], misc.accuracy(y_hat, y), x.shape[0])
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                train_acc_list.append(train_acc)
            local_avg_acc.append(sum(train_acc_list)/len(train_acc_list))
            print(f"client {i}: trainacc {sum(train_acc_list)/len(train_acc_list)}")
        
        
        writer.add_scalars('FederatedSNN/localtrainacc',{f"client{i}": j for i, j in enumerate(local_avg_acc)} , epoch)              

        
        client_models, server_model = federated.FedAvg(client_models, server_model)
        
        test_acc = []
        testmetrics = misc.Accumulator(2) 
        server_model = server_model.to(DEVICE)
        for data in test_loader:
            
            x, y = data
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                testmetrics.add(misc.accuracy(server_model(x), y), x.shape[0]) 
            
            acc = metric[0] / metric[1] 
            test_acc.append(acc)
        
        avg_testacc = sum(test_acc)/len(test_acc)    
        writer.add_scalar('FederatedSNN/globaltrainacc', avg_testacc, epoch)  
        print(f"GLobal epoch: test acc {avg_testacc}\n")
        
        
if __name__ == "__main__":
    
    
    writer = SummaryWriter(comment=f"LR_{config['lr']}_EPOCH_{config['epoch']}_FederatedSNNv2_{5}")
    FederatedSNN_experiment(5) 
    # FederatedCNN_experiment(4)