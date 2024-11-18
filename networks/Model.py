import os

from typing import Union
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from torch.optim import Adam
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
import snntorch as snn
import numpy as np
from tqdm import trange, tqdm
from sklearn.metrics import accuracy_score


from utils.misc import overlay_y_on_x, Conv_overlay_y_on_x
from networks.block import FC_block


DEVICE = torch.device('cuda:3')
writer = SummaryWriter(comment=f"FFLayer")

def goodness_score(pos_acts, neg_acts, threshold=2):
    """
    Compute the goodness score for a given set of positive and negative activations.

    Parameters:

    pos_acts (torch.Tensor): Numpy array of positive activations.
    neg_acts (torch.Tensor): Numpy array of negative activations.
    threshold (int, optional): Threshold value used to compute the score. Default is 2.

    Returns:

    goodness (torch.Tensor): Goodness score computed as the sum of positive and negative goodness values. Note that this
    score is actually the quantity that is optimized and not the goodness itself. The goodness itself is the same
    quantity but without the threshold subtraction
    """

    pos_goodness = -torch.sum(torch.pow(pos_acts, 2)) + threshold
    neg_goodness = torch.sum(torch.pow(neg_acts, 2)) - threshold
    return torch.add(pos_goodness, neg_goodness)

def get_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    return dict(accuracy_score=acc)

class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def ftrain(self, x_pos, x_neg):
        # for i in tqdm(range(self.num_epochs)):
        for i in range(self.num_epochs):
        # for i in trange(self.num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            writer.add_scalar("FFLoss/Layer", loss, i)
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()



class FFLayer(nn.Linear):
    """
    Unsupervised Version of FFLayer 
    """
    
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.to(device)
        self.goodness = goodness_score
        self.ln_layer = nn.LayerNorm(normalized_shape=[1, out_features]).to(device)

    def forward(self, input):
        input = super().forward(input)
        # input = self.ln_layer(input.detach())
        return input

    def ftrain(self, x_pos, x_neg):
        """
        Train the FF Linear Layer using pos_activation and neg_activation (Notes that it is not the data but the output of the ff layer).
        """
        self.opt.zero_grad()
        goodness = self.goodness(x_pos, x_neg, self.threshold)
        goodness.backward(retain_graph=True)
        self.opt.step()

class FFNet(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).to(DEVICE)]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        # goodness_per_label [50000, 10]
        # goodness_per_label.argmax(1) [50000, 1] 
        return goodness_per_label.argmax(1)

    def ftrain(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            # print('training layer', i, '...')
            h_pos, h_neg = layer.ftrain(h_pos, h_neg)

class FFNet_Unsupervised(torch.nn.Module):

    def __init__(self, dims, n_epochs, device):
        super().__init__()
        self.device = device
        self.layers = []
        
        self.n_epochs = n_epochs
        for d in range(len(dims) - 2):
            self.layers += [FFLayer(dims[d], dims[d + 1]).to(DEVICE)]
        ## Only the final layer is the classifier
        self.n_hid_to_log = len(self.layers) - 1
        self.last_layer = nn.Linear(in_features= sum(dims[1:-2]), out_features= dims[-1])
        self.to(device)
        self.opt = torch.optim.Adam(self.last_layer.parameters())
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
        
    def forward(self, image: torch.Tensor):
        image = image.to(self.device)
        image = torch.reshape(image, (image.shape[0], 1, -1))
        concat_output = []
        for idx, layer in enumerate(self.layers):
            image = layer(image)
            if idx > len(self.layers) - self.n_hid_to_log - 1:
                concat_output.append(image)
        concat_output = torch.concat(concat_output, 2)
        logits = self.last_layer(concat_output)
        return logits.squeeze()

    def ftrain(self, pos_dataloader, neg_dataloader):
        """
        forward train ForwardNet using the pos dataloader and neg dataloader
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
                pos_acts = torch.reshape(pos_imgs, (pos_imgs.shape[0], 1, -1)).to(self.device)
                # neg_acts.shape torch.Size([64, 1, 784])
                neg_acts = torch.reshape(neg_imgs, (neg_imgs.shape[0], 1, -1)).to(self.device)

                for idx, layer in enumerate(self.layers):
                    pos_acts = layer(pos_acts)
                    neg_acts = layer(neg_acts)
                    layer.ftrain(pos_acts, neg_acts)
        
        ## Training the last layer
        num_examples = len(pos_dataloader)
        outer_tqdm = tqdm(range(self.n_epochs), desc="Training Last Layer", position=0)
        loss_list = []
        for epoch in outer_tqdm:
            epoch_loss = 0
            inner_tqdm = tqdm(pos_dataloader, desc=f"Training Last Layer | Epoch {epoch}", leave=False, position=1)
            for images, labels in inner_tqdm:
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.opt.zero_grad()
                preds = self(images)
                loss = self.loss(preds, labels)
                epoch_loss += loss
                loss.backward()
                self.opt.step()
            loss_list.append(epoch_loss / num_examples)
            # Update progress bar with current loss
        
        return [l.detach().cpu().numpy() for l in loss_list]
    
    def evaluate(self, dataloader: DataLoader, dataset_type: str = "train"):
        self.eval()
        inner_tqdm = tqdm(dataloader, desc=f"Evaluating model", leave=False, position=1)
        all_labels = []
        all_preds = []
        for images, labels in inner_tqdm:
            images = images.to(self.device)
            labels = labels.to(self.device)
            preds = self(images)
            preds = torch.argmax(preds, 1)
            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())
        all_labels = torch.concat(all_labels, 0).numpy()
        all_preds = torch.concat(all_preds, 0).numpy()
        metrics_dict = get_metrics(all_preds, all_labels)
        print(f"{dataset_type} dataset scores: ", "\n".join([f"{key}: {value}" for key, value in metrics_dict.items()]))
      
        
        
class BPNet(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []

        self.model = nn.Sequential()
        for d in range(len(dims) - 1):
            self.model.append(nn.Linear(dims[d], dims[d + 1]))
            self.model.append(nn.ReLU())


    def forward(self, x):
        y = self.model(x)
        return y

    # def trainnet(self, x):
    #     y = self.forward(x)
    #     return y

    def predict(self, x):
        y = self.forward(x)

        return y

class FFConvLayer(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride= 1, padding = 0, dilation= 1, groups= 1, bias= True, padding_mode= 'zeros', device=None, dtype=None, isrelu=True, ismaxpool=True) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.relu = torch.nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.opt = Adam(self.parameters(), lr=0.05)
        self.threshold = 2.0
        self.num_epochs = 1000 
        self.ismaxpool = ismaxpool
        self.isrelu = isrelu

        
        
    def add_compute(self, func):
        pass
        
    def forward(self, input):
        
        output = self._conv_forward(input, self.weight, self.bias)
        if self.isrelu:
            output = self.relu(output)
        if self.ismaxpool:
            output = self.maxpool(output)
        return output
    
    def ftrain(self, x_pos, x_neg):
        
        for i in range(self.num_epochs):
        # for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            T = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean(dim=(1,2))
            mask = torch.isinf(T)
            
            T = T[~mask]
            loss = T.mean()
            writer.add_scalar("FFLoss/Layer", loss, i)
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

class FlattenLayer(nn.Flatten):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__(start_dim, end_dim)
    
    def ftrain(self, x_pos, x_neg):

        return x_pos.flatten(self.start_dim, self.end_dim).detach(), x_neg.flatten(self.start_dim, self.end_dim).detach()      


class FFAlexNet(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = [
            FFConvLayer(1, 32, kernel_size=3, padding=1).to(DEVICE),
            FFConvLayer(32, 64, kernel_size=3, stride=1, padding=1).to(DEVICE),
            FFConvLayer(64, 128, kernel_size=3, padding=1, ismaxpool=False).to(DEVICE),
            FFConvLayer(128, 256, kernel_size=3, padding =1, ismaxpool=False).to(DEVICE),
            FlattenLayer().to(DEVICE),
            FFLayer(12544, 1024).to(DEVICE),
            FFLayer(1024, 512).to(DEVICE), 
            FFLayer(512, 10).to(DEVICE)
        ]
         
    def ftrain(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            layer.train()
            h_pos, h_neg = layer.ftrain(h_pos, h_neg)        

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = Conv_overlay_y_on_x(x, label)
            goodness = []
            h = h.to(DEVICE)
            for layer in self.layers:
                h = layer(h)
                if isinstance(layer, FlattenLayer):
                    pass
                else:
                    goodness += [h.pow(2).mean(dim=[di for di in range(1, len(h.shape))])]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        # goodness_per_label [50000, 10]
        # goodness_per_label.argmax(1) [50000, 1] 
        return goodness_per_label.argmax(1)

    
    
hyperparams= [16, 1728, 3, 0.005, 20, 'vehicleimage', 0.001, 'C1']


class SNN(nn.Module):
    def __init__(self, layersize, batchsize):
        """
        layersize is the size of each layer like [24*24*3, 500, 3]
        stepsize is the batchsize.
        """
        super(SNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers_size = layersize
        self.lastlayer_size = layersize[-1]
        self.len = len(self.layers_size) - 1
        self.error = None
        self.stepsize = batchsize
        self.time_windows = 20

        for i in range(self.len):
            self.layers.append(FC_block(self.stepsize,self.time_windows, self.lastlayer_size, self.layers_size[i], self.layers_size[i + 1]))

    def forward(self, input):
        for step in range(self.stepsize):

            x = input > torch.rand(input.size()).to(DEVICE)

            x = x.float().to(DEVICE)
            x = x.view(self.stepsize, -1)
            y = x
            for i in range(self.len):
                y = self.layers[i](y)
#        print('x',x)
#        print('x.shape',x.shape)
        outputs = self.layers[-1].sumspike / self.time_windows 

        return outputs


    def predict(self, x):
        goodness_per_label = []
        _out = torch.zeros(10, x.shape[0], 500).to(DEVICE)
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
            
            _out[label] = h
        goodness_per_label = torch.cat(goodness_per_label, 1)
        result = [_out, goodness_per_label]
        return result
    def train_in_shallow(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer in shallow', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)
        return h_pos, h_neg

# class FFNet_deep(torch.nn.Module):
#     '''
#     description: 深层模型
#     param {*} self
#     param {*} x
#     return {*}
#     '''
#     def __init__(self, dims):
#         super().__init__()
#         self.layers = []
#         for d in range(len(dims) - 1):
#             self.layers += [Layer(dims[d], dims[d + 1]).to(DEVICE)]

#     def predict(self, x):
#         goodness_per_label = []
#         for label in range(10):
#             h = x[0][label]
#             goodness = []
#             for layer in self.layers:
#                 h = layer(h)
#                 goodness += [h.pow(2).mean(1)]
#             goodness_per_label += [sum(goodness).unsqueeze(1)]
#         goodness_per_label = torch.cat(goodness_per_label, 1)
#         goodness_per_label_mean = (goodness_per_label + x[1]) / 2.0
#         return goodness_per_label_mean.argmax(1)

#     def train_in_deep(self, x_pos, x_neg):
#         h_pos, h_neg = x_pos, x_neg
#         for i, layer in enumerate(self.layers):
#             print('training layer in deep', i, '...')
#             h_pos, h_neg = layer.train(h_pos, h_neg)

class BPNet_split(torch.nn.Module):
    '''
    description: 
    param {*} self
    param {*} dims
    return {*}
    '''    
    def __init__(self, dims):
        super().__init__()
        self.shallow_model = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU()
        )
        self.deep_model = nn.Sequential()
        self.dims = dims
        for d in range(1, len(dims) - 1):
            self.deep_model.append(nn.Linear(dims[d], dims[d + 1]))
            self.deep_model.append(nn.ReLU())

    def forward(self, x):
        shallow_output = self.shallow_model(x)
        deep_output = self.deep_model(shallow_output)
        return deep_output
    
class FFNet_shallow(torch.nn.Module):
    '''
    description: 浅层模型
    param {*} h_pos, h_neg: 浅层模型最后输出的一层特征,直接输入给深层网络
    param {*} result: _out浅层模型最后输出的一层特征,直接输入给深层网络
               goodness_per_label:浅层模型的预测精度     
    return {*}
    '''
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        self.dims = dims
        for d in range(len(dims) - 1):
            self.layers += [FFLayer(dims[d], dims[d + 1]).to(DEVICE)]


    def predict(self, x):
        goodness_per_label = []
        # _out = torch.zeros(10, x.shape[0], 500).to(DEVICE)
        _out = torch.zeros(10, x.shape[0], self.dims[-1]).to(DEVICE)
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
            
            _out[label] = h
        goodness_per_label = torch.cat(goodness_per_label, 1)
        result = [_out, goodness_per_label]
        return result
    def train_in_shallow(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            # print('training layer in shallow', i, '...')
            h_pos, h_neg = layer.ftrain(h_pos, h_neg)
        return h_pos, h_neg

class FFNet_deep(torch.nn.Module):
    '''
    description: 深层模型
    param {*} self
    param {*} x
    return {*}
    '''
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [FFLayer(dims[d], dims[d + 1]).to(DEVICE)]
    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = x[0][label]
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        goodness_per_label_mean = (goodness_per_label + x[1]) / 2.0
        return goodness_per_label_mean.argmax(1)
    def train_in_deep(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            # print('training layer in deep', i, '...')
            h_pos, h_neg = layer.ftrain(h_pos, h_neg)
    
    
hyperparams= [16, 1728, 3, 0.005, 20, 'vehicleimage', 0.001, 'C1']


class SNNv1(nn.Module):
    """
    SNN v1 implement from LFL
    """
    def __init__(self, layersize, batchsize):
        """
        layersize is the size of each layer like [24*24*3, 500, 3]
        stepsize is the batchsize.
        """
        super(SNNv1, self).__init__()
        self.layers = nn.ModuleList()
        self.layers_size = layersize
        self.lastlayer_size = layersize[-1]
        self.len = len(self.layers_size) - 1
        self.error = None
        self.stepsize = batchsize
        self.time_windows = 20

        for i in range(self.len):
            self.layers.append(FC_block(self.stepsize,self.time_windows, self.lastlayer_size, self.layers_size[i], self.layers_size[i + 1]))

    def forward(self, input):
        for step in range(self.stepsize):

            x = input > torch.rand(input.size()).to(DEVICE)

            x = x.float().to(DEVICE)
            x = x.view(self.stepsize, -1)
            y = x
            for i in range(self.len):
                y = self.layers[i](y)
#        print('x',x)
#        print('x.shape',x.shape)
        outputs = self.layers[-1].sumspike / self.time_windows 

        return outputs
    
    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = x[0][label]
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        goodness_per_label_mean = (goodness_per_label + x[1]) / 2.0
        return goodness_per_label_mean.argmax(1)

    def train_in_deep(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer in deep', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)
            
            
            
class SNNv2(nn.Module):
    """
    SNN v2 implement from snntorch 
    """
    def __init__(self, layersize):
        super().__init__()       
        # Initialize layers
        self.layersize = layersize
        self.fc1 = nn.Linear(self.layersize[0], self.layersize[1])
        self.beta = 0.80
        self.num_steps = 20
        
        self.lif1 = snn.Leaky(beta=self.beta)
        self.fc2 = nn.Linear(self.layersize[1], self.layersize[2])
        self.lif2 = snn.Leaky(beta=self.beta)

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
    
    
    
class CNNNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = []

        self.feature = nn.Sequential(
            #1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),   # 28*28->32*32-->28*28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
            
            #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes),
        )
        
    def forward(self, x):
        return self.classifier(self.feature(x))

    # def trainnet(self, x):
    #     y = self.forward(x)
    #     return y

    def predict(self, x):
        y = self.forward(x)

        return y
