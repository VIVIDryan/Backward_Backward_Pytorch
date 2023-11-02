import torch
import time
from networks.Model import FFNet_shallow, FFNet_deep
from dataloaders.dataset import MNIST_loaders
from utils.misc import *
from torchvision import transforms as tortra
from tqdm import trange, tqdm

DEVICE = torch.device("cuda:3")
from torch.utils.tensorboard import SummaryWriter

config = {"lr": 0.001, "epoch": 50, "batchsize": 64}
writer = SummaryWriter(comment=f"LR_{config['lr']}_EPOCH_{config['epoch']}_rewriteFF")


# ---------------------------------- 设置网络结构 ---------------------------------- #
def splitFF_experiment():
    net_shallow = FFNet_shallow([784, 500]).to(DEVICE)  # 浅层模型
    net_deep = FFNet_deep([500, 500]).to(DEVICE)  # 深层模型

    # ----------------------------------- 读取数据 ----------------------------------- #
    train_loader, test_loader = MNIST_loaders(batch_size=64)
    x, y = next(iter(train_loader[0]))
    # ----------------------------------- 开始训练 ----------------------------------- #
    
    outertqdm = trange(config["epoch"], desc=f"Global epoch==>", position=2)
    for epoch in outertqdm:
        train_acc = []
        for i, (x, y) in enumerate(train_loader[0]):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_pos = overlay_y_on_x(x, y)
            rnd = torch.randperm(x.size(0))
            x_neg = overlay_y_on_x(x, y[rnd])
            x_te, y_te = next(iter(test_loader))
            x_te, y_te = x_te.to(DEVICE), y_te.to(DEVICE)
            out_pos, out_neg = net_shallow.train_in_shallow(x_pos, x_neg)
            out_pos, out_neg = out_pos.to(DEVICE), out_neg.to(DEVICE)
            net_deep.train_in_deep(out_pos, out_neg)
            out_x = net_shallow.predict(x)
            train_acc.append(net_deep.predict(out_x).eq(y).float().mean().item())
            break
        print(f"Epoch {i} train acc of FF:", sum(train_acc) / len(train_acc))
        writer.add_scalar("FFAccuracy/train", sum(train_acc) / len(train_acc), epoch)

        acc_list = []
        for x_te, y_te in test_loader:
            x_te, y_te = x_te.to(DEVICE), y_te.to(DEVICE)
            out_x_te = net_shallow.predict(x_te)
            acc = net_deep.predict(out_x_te).eq(y_te).float().mean().item()
            acc_list.append(acc)
            break

        print("test acc of FF:", sum(acc_list) / len(acc_list))
        writer.add_scalar("FFAccuracy/test", sum(acc_list) / len(acc_list), epoch)


def splitFF_with_new_neg():
    net_shallow = FFNet_shallow([784, 500]).to(DEVICE)  # 浅层模型
    net_deep = FFNet_deep([500, 500]).to(DEVICE)  # 深层模型

    # ----------------------------------- 读取数据 ----------------------------------- #
    train_loader, test_loader = MNIST_loaders(batch_size=64)
    x, y = next(iter(train_loader[0]))
    neg_dataset = torch.load("./data/transformed_dataset.pt")
    neg_dataloader = DataLoader(
            neg_dataset, batch_size=config["batchsize"], shuffle=True, num_workers=4
        )
    # ----------------------------------- 开始训练 ----------------------------------- #
    train_acc = []
    outertqdm = trange(config["epoch"], desc=f"Global epoch==>", position=2)
    for epoch in outertqdm:
        datatqdm = tqdm(zip(train_loader[0], neg_dataloader), desc= f"Local Iteration {epoch}", leave=False, position=1)
        acclist = []
        for pos_data, neg_imgs in datatqdm:
            x_pos, y = pos_data
            x_neg = neg_imgs.view(64, -1)
            x_pos = x_pos.to(DEVICE)
            x_neg = x_neg.to(DEVICE)
            y = y.to(DEVICE)
            out_pos, out_neg = net_shallow.train_in_shallow(x_pos, x_neg)
            out_pos, out_neg = out_pos.to(DEVICE), out_neg.to(DEVICE)
            net_deep.train_in_deep(out_pos, out_neg)
            out_x = net_shallow.predict(x_pos)
            acc = net_deep.predict(out_x).eq(y).float().mean().item()
            acclist.append(acc)
            print(f"acc:{acc}\n")
            break
        a = sum(acclist) /len(acclist)   
        print(f"epoch{epoch}: train acc, {a}")
        # writer.add_scalar("FFAccuracy/train", sum(train_acc) / len(train_acc), epoch)

        acc_list = []
        for x_te, y_te in test_loader:
            x_te, y_te = x_te.to(DEVICE), y_te.to(DEVICE)
            out_x_te = net_shallow.predict(x_te)
            acc = net_deep.predict(out_x_te).eq(y_te).float().mean().item()
            acc_list.append(acc)
            # break

        # print("test acc of FF:", sum(acc_list) / len(acc_list))
        # writer.add_scalar("FFAccuracy/test", sum(acc_list) / len(acc_list), epoch)

def SplitFF_multi_clients_experiment(num_of_clients):
    num_of_clients = num_of_clients

    transform = tortra.Compose(
        [
            tortra.ToTensor(),
            tortra.Normalize((0.1307,), (0.3081,)),
            tortra.Lambda(lambda x: torch.flatten(x)),
        ]
    )

    pos_dataloaders, _ = MNIST_loaders(
        batch_size=config["batchsize"], transform=transform, num_subsets=num_of_clients
    )
    neg_dataset = torch.load("./data/transformed_dataset.pt")
    neg_datasets = split_non_iid_data(neg_dataset, num_of_clients)
    neg_dataloaders = [
        DataLoader(
            neg_datasets[i], batch_size=config["batchsize"], shuffle=True, num_workers=4
        )
        for i in range(len(neg_datasets))
    ]

    clients_models = [
        FFNet_shallow([784, 1200]).to(DEVICE) for _ in range(num_of_clients)
    ]
    server_model = FFNet_deep([1200, 500, 300, 64, 16]).to(DEVICE)

    train_acc = []

    print("Start!")
    outertqdm = trange(config["epoch"], desc=f"Global epoch==>", position=2)
    for epoch in outertqdm:
        
        clientstqdm = trange(
            num_of_clients, position=1, desc=f"Local Train==>", leave=False
        )
        for client in clientstqdm:
            datatqdm = tqdm(
                zip(pos_dataloaders[client], neg_dataloaders[client]),
                desc=f"Local Iteration {epoch} in Client_{client}",
                leave=False,
                position=1,
            )
            train_acc = []
            for pos_data, neg_imgs in datatqdm:
                x_pos, y = pos_data
                # x_neg = neg_imgs.unsqueeze(1)
                x_neg = neg_imgs.view(64, -1)
                x_pos = x_pos.to(DEVICE)
                x_neg = x_neg.to(DEVICE)
                y = y.to(DEVICE)
                out_pos, out_neg = clients_models[client].train_in_shallow(x_pos, x_neg)
                out_pos, out_neg = out_pos.to(DEVICE), out_neg.to(DEVICE)
                server_model.train_in_deep(out_pos, out_neg)
                out_x = clients_models[client].predict(x_pos)
                train_acc.append(
                    server_model.predict(out_x).eq(y).float().mean().item()
                )
                print(f"\n Epoch {epoch} train acc of FF:{sum(train_acc)/len(train_acc)}")
                break
        


if __name__ == "__main__":
    torch.manual_seed(1234)
    SplitFF_multi_clients_experiment(5)
    splitFF_experiment()
    # splitFF_with_new_neg()
    # print("Done!")
