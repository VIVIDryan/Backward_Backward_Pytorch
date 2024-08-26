import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as tortra
import snntorch as snn


from networks.Model import FFNet, BPNet, FFNet_Unsupervised
from dataloaders.dataset import MNIST_loaders, debug_loaders
from utils import misc

import sys
import logging


DEVICE = torch.device('cuda:3')

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

from networks.Model import FFLayer, goodness_score

class FFLeaky(snn.Leaky):
    def __init__(self, device, para, *args, **kwargs):
        super(FFLeaky, self).__init__(*args, **kwargs)
        self.opt = Adam(para, lr=0.03)
        self.device = device
        self.to(device)
        self.goodness = goodness_score
        self.threshold = torch.tensor(2.0)
        self.relu = torch.nn.ReLU()
        self.mem = False 
        
    def ftrain(self, x_pos, x_neg, num_steps):
    
        self.opt.zero_grad()
        
        goodness = torch.zeros((1), dtype = torch.float, device = self.device)
        for step in range(num_steps):
            goodness += self.goodness(x_pos[step], x_neg[step], self.threshold)
            
        goodness.backward(retain_graph=True)
        self.opt.step()
        

class FF_SNN(nn.Module):
    """
    SNN trianing using FF on the basis of SNN v2  
    """
    def __init__(self, layersize, n_epochs, device):
        super().__init__()       
        # Initialize layers
        self.layersize = layersize
        self.device = device
        self.beta = 0.80
        self.num_steps = 25

        self.fc1 = FFLayer(self.layersize[0], self.layersize[1], device=device)
        self.lif1 = FFLeaky(beta=self.beta, device=device, para = self.fc1.parameters())
        self.fc2 = FFLayer(self.layersize[1], self.layersize[2], device=device)
        self.lif2 = FFLeaky(beta=self.beta, device=device, para = self.fc2.parameters())
        self.layers = nn.ModuleList()
        self.layers.extend([
            self.fc1,
            self.lif1,
            self.fc2,
            self.lif1
        ])
        self.n_epochs = n_epochs 
        
    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

    def ftrain(self, pos_dataloader, neg_dataloader):
        """
        forward train SNN using the pos dataloader and neg dataloader
        """
        self.train()
        outer_tqdm = tqdm(range(self.n_epochs), desc="Training FF Layers", position=0)
        for epoch in outer_tqdm:
            inner_tqdm = tqdm(zip(pos_dataloader, neg_dataloader), desc=f"Training FF Layers | Epoch {epoch}",
                              leave=False, position=1)
            for pos_data, neg_imgs in inner_tqdm:
                
                ## Don't need to flatten in the data transpose
                pos_imgs, _ = pos_data
                ### squeeze the data
                # pos_imgs.shape  torch.Size([64, 1, 28, 28])
                pos_imgs = torch.squeeze(pos_imgs)
                
                # pos_acts.shape  torch.Size([64, 1, 784])
                pos_acts = torch.reshape(pos_imgs, (pos_imgs.shape[0], -1)).to(self.device)
                # neg_acts.shape torch.Size([64, 1, 784])
                neg_acts = torch.reshape(neg_imgs, (neg_imgs.shape[0], -1)).to(self.device)

                
                for idx, layer in enumerate(self.layers):
                    if isinstance(layer, FFLeaky):
                        mem_pos = layer.init_leaky()
                        neg_pos = layer.init_leaky()
                        
                        pos_acts, pos_mem = layer(pos_acts, mem_pos)
                        neg_acts, neg_mem = layer(neg_acts, neg_pos) 
                        
                        ## calculate forward loss
                    else:
                        pos_acts = layer(pos_acts)
                        neg_acts = layer(neg_acts)
                        ## dont calculate the forward loss just forward the output
                        

                    
                    if isinstance(layer, FFLeaky):
                        layer.ftrain(pos_mem, neg_mem, self.num_steps)    
                    else:
                        layer.ftrain(pos_acts, neg_acts)
                
                
                
                
                # for idx, layer in enumerate(self.layers):
                    
                    
                #     # for step in range(self.num_steps):
                #     #     loss_val = 
                    
                    
                    
                #     if isinstance(layer, FFLeaky):
                        
                #         mem_pos = layer.init_leaky()
                #         neg_pos = layer.init_leaky()
                        
                #         pos_acts, _ = layer(pos_acts, mem_pos)
                #         neg_acts, _ = layer(neg_acts, neg_pos)
                #     else:
                #         pos_acts = layer(pos_acts)
                #         neg_acts = layer(neg_acts)
    
                #     if not isinstance(layer, FFLeaky):
                #         layer.ftrain(pos_acts, neg_acts)
        
    def evaluate(self, dataloder):
        total = 0
        correct = 0
        
        with torch.no_grad():
            self.eval()
            for data, targets in dataloder:
                data = data.to(self.device)
                targets = targets.to(self.device)
                test_spk, _ = self.forward(data.view(data.size(0), -1))
                _, predicted = test_spk.sum(dim=0).max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        
        logging.warning(f'Test Accuracy: {accuracy}%')
        return accuracy
        

def SNNFF_experiment_withNN():
    """
    SNN with Forward to Forward algorithm
    """
    from torch.utils.data import DataLoader
    
    batchsize = 50000
    epochs = 100
    transformss = tortra.Compose([
            tortra.ToTensor(),
            tortra.Normalize((0.1307,), (0.3081,)),
            ])
    pos_train_loader, test_loader = MNIST_loaders(batch_size=batchsize ,transform=transformss)
    pos_train_loader = pos_train_loader[0]
    
    
    neg_dataset = torch.load('/home/datasets/SNN/transformed_flattendataset.pt')
    neg_train_loader = DataLoader(neg_dataset, batch_size=batchsize)
    
    unsuperviesd_ff = FF_SNN([28*28, 1000, 10], n_epochs = epochs, device = DEVICE)
    
    loss_list = unsuperviesd_ff.ftrain(pos_train_loader, neg_train_loader)

    # misc.plot_loss(loss_list)
    unsuperviesd_ff.evaluate(pos_train_loader)
    # unsuperviesd_ff.evaluate(pos_train_loader, dataset_type="Train")
    # unsuperviesd_ff.evaluate(test_loader, dataset_type="Test")
    

if __name__ == "__main__":
   SNNFF_experiment_withNN() 