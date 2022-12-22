from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import torch
from model import regressor
from dataset import mydata_set
import os, glob, argparse, wandb
import pandas as pd


def train(opt):
    os.makedirs('ckpt', exist_ok = True)
    model = regressor(opt.attribute_num).to(opt.device).train()
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = opt.lr , momentum = opt.moment)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)
    
    data_set = mydata_set(opt.data_path)
  
    my_data_loader = DataLoader(data_set, batch_size = opt.batch_size, shuffle = True)
    cnt = 0
    lossfunc = nn.BCELoss()
    for epoch in range(opt.max_epoch):
        torch.save(model.state_dict(), f'./ckpt/learning_{epoch}_epoch.pt')
        for batch_idx, sample in enumerate(my_data_loader):
            input_tensor ,y = sample
            y = y.to(opt.device)
            input_tensor = input_tensor.to(opt.device)
            y_hat = model(input_tensor)
            loss = lossfunc( y_hat, y.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt = cnt + 1

            print(f'epoch:{epoch},iter:{batch_idx}||loss:{loss}')

    torch.save(model.state_dict(), './ckpt/learning_result.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type = str, default='cuda')
    parser.add_argument('--optimizer', type = str, default='Adam')
    parser.add_argument('--max_epoch', type = int, default= 40)
    parser.add_argument('--slope', type = float, default = 0.1)
    parser.add_argument('--lr', type = float, default = 0.00001)
    parser.add_argument('--moment', type = int, default= 0.1)
    parser.add_argument('--batch_size',type = int, default = 16 )
    parser.add_argument('--attribute_num', type = int, default = 1)
    #data_forder which have two forder named 'F' and 'M'
    parser.add_argument('--data_path', type = str)
    opt = parser.parse_args()
    train(opt)







