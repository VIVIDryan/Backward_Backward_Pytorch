import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as tortra

from networks.Model import FFNet, BPNet, FFNet_Unsupervised
from dataloaders.dataset import MNIST_loaders, debug_loaders
from utils import misc

import sys
import logging


DEVICE = torch.device('cuda:3')
config = {
    'lr': 0.001,
    'epoch': 50,

}
writer = SummaryWriter(
    comment=f"LR_{config['lr']}_EPOCH_{config['epoch']}_rewriteFF")

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)



def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()


def FF_experiment():
    train_loader, test_loader = MNIST_loaders()

    net = FFNet([784, 1000, 1000]).to(DEVICE)

    FF_start_time = time.time()
    train_acc = []
    for i, (x, y) in enumerate(train_loader[0]):

        x, y = x.to(DEVICE), y.to(DEVICE)
        x_pos = misc.overlay_y_on_x(x, y)
        rnd = torch.randperm(x.size(0))
        x_neg = misc.overlay_y_on_x(x, y[rnd])

        # for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
        #     visualize_sample(data, name)

        net.ftrain(x_pos, x_neg)
        train_acc.append(net.predict(x).eq(y).float().mean().item())
        
        break
    
    logging.warning(f'train acc of FF:{sum(train_acc)/len(train_acc)}')  
    
         
    
    FF_end_time = time.time()
    journey = FF_end_time - FF_start_time
    writer.add_scalar('FFAccuracy/train', sum(train_acc)/len(train_acc))
    writer.add_scalar('Time/FFtime', journey)

    acc_list = []
    for x_te, y_te in test_loader:

        x_te, y_te = x_te.to(DEVICE), y_te.to(DEVICE)
        acc = net.predict(x_te).eq(y_te).float().mean().item()
        acc_list.append(acc)
        break

    logging.warning(f'test acc of FF: {sum(acc_list)/len(acc_list)}')


def FF_experiment_withNN():
    """
    Forward Experiment with unsupervised approach
    """
    
    from torch.utils.data import DataLoader
    
    batchsize = 50000
    epochs = 80
    transformss = tortra.Compose([
            tortra.ToTensor(),
            tortra.Normalize((0.1307,), (0.3081,)),
            ])
    pos_train_loader, test_loader = MNIST_loaders(batch_size=batchsize ,transform=transformss)
    pos_train_loader = pos_train_loader[0]
    
    
    neg_dataset = torch.load('/home/datasets/SNN/transformed_flattendataset.pt')
    neg_train_loader = DataLoader(neg_dataset, batch_size=batchsize)
    
    unsuperviesd_ff = FFNet_Unsupervised([28*28, 500, 500, 500, 10], n_epochs = epochs, device = DEVICE)
    
    loss_list = unsuperviesd_ff.ftrain(pos_train_loader, neg_train_loader)

    misc.plot_loss(loss_list)
    
    unsuperviesd_ff.evaluate(pos_train_loader, dataset_type="Train")
    unsuperviesd_ff.evaluate(test_loader, dataset_type="Test")
    
    

def BP_experiment():
    BPtrain_loader, BPtest_loader = MNIST_loaders(512)

    BP_net = BPNet([784, 500, 10]).to(DEVICE)
    BP_loss = nn.CrossEntropyLoss(reduction='none')
    BP_optim = torch.optim.Adam(BP_net.parameters(), lr=config['lr'])
    bestacc = 0.0
    minloss = 1.0
    journeylist = []
    for epoch in range(config['epoch']):
        BP_start_time = time.time()
        BP_metric = misc.Accumulator(3)
        for x, y in tqdm(BPtrain_loader[0]):
            if isinstance(BP_net, torch.nn.Module):
                BP_net.train()
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_hat = BP_net(x)
            loss = BP_loss(y_hat, y)
            BP_optim.zero_grad()
            loss.mean().backward()
            BP_optim.step()

            BP_metric.add(float(loss.sum()),
                          misc.accuracy(y_hat, y), y.numel())
        BP_end_time = time.time()
        journeylist.append(BP_end_time-BP_start_time)

        avg_loss, avg_acc = BP_metric[0] / \
            BP_metric[2], BP_metric[1] / BP_metric[2]
        writer.add_scalar('BPLoss/train', avg_loss, epoch)
        writer.add_scalar('BPAccuracy/train', avg_acc, epoch)
        print(f"Epoch {epoch}: loss {avg_loss}, acc {avg_acc}\n")
        if avg_loss <= minloss:
            minloss = avg_loss
        if avg_acc >= bestacc:
            bestacc = avg_acc

        BP_net.eval()
        testmetric = misc.Accumulator(2)
        with torch.no_grad():
            for X, y in BPtest_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                testmetric.add(misc.accuracy(BP_net(X), y), y.numel())

        testacc = testmetric[0] / testmetric[1]
        writer.add_scalar('BPAccuracy/test', testacc, epoch)
        print(f"Epoch {epoch}: test acc: {testacc}\n")

    writer.add_scalar('Time/BPtime', sum(journeylist))

    print(f"BP trainacc:{bestacc} \n BP trainloss:{minloss}")
    print(f"BP testacc:{testacc}\n")


if __name__ == "__main__":
    torch.manual_seed(1234)


    # FF_experiment()
    
    FF_experiment_withNN()
    # BP
    # BP_experiment()
    print(f"Done")
