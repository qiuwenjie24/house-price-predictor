# utils/train.py 模型训练函数与数据加载器

import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F
from utils.loss import ElasticNetLoss
from utils.metrics import log_rmse

def load_data_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def train(net, train_features, train_labels, valid_features, valid_labels,
          num_epochs, learning_rate, regula_str, l1_ratio, batch_size, is_optuna=False):
    train_ls, valid_ls = [], []
    train_iter = load_data_array((train_features, train_labels), batch_size, is_train=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                    base_lr=learning_rate / 400,
                    max_lr=learning_rate,
                    step_size_up=1000,
                    mode="exp_range")
    loss = ElasticNetLoss(regula_str=regula_str, l1_ratio=l1_ratio)

    '''早停参数'''
    patience = 10
    best_loss = float('inf')
    best_epoch = 0
    no_improve = 0

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y, net)
            l.backward()
            optimizer.step()
            scheduler.step()

        train_ls.append(log_rmse(net, train_features, train_labels))
        if valid_labels is not None:
            current_val_loss = log_rmse(net, valid_features, valid_labels)
            valid_ls.append(current_val_loss)
            '''记录最小验证损失'''
            if current_val_loss < best_loss:
                best_loss = current_val_loss
                best_epoch = epoch
                no_improve = 0
            elif is_optuna:
                '''自动调参时才启动早停'''
                no_improve += 1
                if no_improve >= patience:
                    break

    return train_ls, valid_ls, best_epoch, best_loss