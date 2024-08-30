import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import time
import logging
import copy
import sys
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as tortra
import numpy as np

from networks.Model import CNNNet, SNNv1, SNNv2
from dataloaders.dataset import MNIST_loaders, MNIST_jaxdataset
from utils import misc
from utils import federated


from jax import numpy as jnp, random, nn, jit
import sys, getopt as gopt, optparse, time

from networks.csdp_model import CSDP_SNN
## bring in ngc-learn analysis tools
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL

from utils.loadargs import load_arg



torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.WARN,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


DEVICE = torch.device('cuda:3')



def FederatedSNNFF_experiment(args):
 
    print('################\n \
          Global epoch:{0}\n \
          Local epoch:{1}\n\
          Learning rate:{2}\n'
          .format(args.globalepoch, args.epoch, args.lr))   
    
    dkey = random.PRNGKey(args.seed)
    dkey, *subkeys = random.split(dkey, 10)    
    num_of_clients = args.nc
    
    
    server_model = CSDP_SNN(subkeys[1], in_dim=784, out_dim=10, hid_dim=3000, hid_dim2=600,
              batch_size=args.batchsize, eta_w=0.002, T=50, dt=3, algo_type=args.algo_type,
              exp_dir='exp_supervised_mnist', learn_recon= True) 

    client_models = [copy.copy(server_model) for _ in range(num_of_clients)]

    # outertqdm = tqdm(range(config['globalepoch']), desc=f"Global Epoch", position=0)
    
    global_test_acc_list = []  # 用于存储每个globalepoch的avg_testacc
    local_avg_acc_lists = []  # 用于存储每个globalepoch的avg_trainacc
    local_avg_nll_lists = []
    client_datasets, test_data = MNIST_jaxdataset(args.nc)
    Xdev, Ydev = test_data

    for epoch in range(args.globalepoch):

        logging.warning(f"Global epoch {epoch+1}")
        
        global_metric = misc.Accumulator(2) ## local_avg_train_acc, local_avg_train_nll
        for i , iterator in enumerate(client_datasets):
            loss_hist = []
            train_loss_hist = []      # 用于存储训练损失
            client_models[i] = client_models[i]
            train_acc_list = [] # save the average acc of local epochs
            X , Y = iterator
            metric = misc.Accumulator(2) ## avg_train_acc, avg_train_nll
                     
            for epoch in range(args.epoch):
                dkey, *subkeys = random.split(dkey, 3)
                ptrs = random.permutation(subkeys[0], X.shape[0])
                X = X[ptrs, :]
                Y = Y[ptrs, :]
                n_batches = int(X.shape[0]/args.batchsize)
                ## begin a single epoch/iteration
                n_samp_seen = 0
                tr_nll = 0.
                tr_acc = 0.             
                for j in range(n_batches):
                    dkey, *subkeys = random.split(dkey, 2)
                    ## sample mini-batch of patterns
                    idx = j * args.batchsize 
                    s_ptr = idx
                    e_ptr = idx + args.batchsize 
                    if e_ptr > X.shape[0]:
                        e_ptr = X.shape[0]
                    Xb = X[s_ptr: e_ptr, :]
                    Yb = Y[s_ptr: e_ptr, :]

                    ## perform a step of inference/learning
                    yMu, yCnt, _, _, _, x_mu = client_models[i].process(
                        Xb, Yb, dkey=dkey, adapt_synapses=True, collect_rate_codes=True
                    )
                    ## track "online" training log likelihood and accuracy
                    _tr_acc, _tr_nll = misc.measure_acc_nll(yMu, Yb) # compute masked scores
                    tr_nll += _tr_nll * Yb.shape[0] ## un-normalize score
                    tr_acc += _tr_acc * Yb.shape[0] ## un-normalize score
                    n_samp_seen += Yb.shape[0]

                    train_msg = "\r Client{} Epoch {} (batch {}/{}): Train.NLL = {:.5f} ACC = {:.3f}".format(
                        i, epoch, (j+1), n_batches, (tr_nll / n_samp_seen), (tr_acc / n_samp_seen) * 100.
                    )
                    if args.verbosity > 0:
                        train_msg = "{} (Online training estimate)".format(train_msg)
                    print(
                        train_msg, end=""
                    )
                tr_acc = (tr_acc/n_samp_seen)
                tr_nll = (tr_nll/n_samp_seen)  
                metric.add(tr_acc, tr_nll)
            avg_metric = metric.avg()
            global_metric.add(avg_metric[0], avg_metric[1])
        local_avg_acc_lists.append(global_metric[0])    
        local_avg_nll_lists.append(global_metric[1])    
        client_models, server_model = federated.SNNavg_aggregation(client_models, server_model)
        test_nll, test_acc, bce, mse = misc.eval_model(
            server_model, Xdev, Ydev, args.batchsize, dkey, verbosity=args.verbosity)

        global_test_acc_list.append(test_acc)    
        logging.warning(f"GLobal epoch: test acc {test_acc}\n")

    logging.warning(f"Finsh Training")
    logging.warning("Begin Ploting!")
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, args.globalepoch + 1), global_test_acc_list, marker='o', linestyle='-')
    plt.title("Global Test Accuracy")
    plt.xlabel("Global Epoch")
    plt.ylabel("Acc")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(1, args.globalepoch + 1), local_avg_acc_lists, marker='o', linestyle='-')
    plt.title("Local Train Accuracy")
    plt.xlabel("Global Epoch")
    plt.ylabel("Acc")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(1, args.globalepoch + 1), local_avg_acc_lists, marker='o', linestyle='-')
    plt.title("Local Train NLL")
    plt.xlabel("Global Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # 确保子图不重叠
    plt.savefig('figures/FederatedSNNFF/FederatedSNNFF_G_{0}_L_{1}_Client_{3}.png'.format(args.globalepoch, args.epoch, num_of_clients)) 
    logging.warning("End Ploting!") 




      
def FederatedSNNv2_experiment(num_of_clients):
    
    config = {
    'lr': 5e-4,
    'epoch': 3,
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
    
    args = load_arg()
    
    # writer = SummaryWriter(comment=f"LR_{config['lr']}_EPOCH_{config['epoch']}_FederatedSNNv2_{5}")
    FederatedSNNFF_experiment(args) 
    # FederatedCNN_experiment(4)