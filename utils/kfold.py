# utils/kfold.py K折交叉验证

import torch
from utils.train import train

def get_k_fold_data(k, i, X, y):
    '''获取当前折的训练数据和验证数据'''
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate,
           regula_str, l1_ratio, batch_size, get_net, is_optuna=False):
    train_l_epochs, valid_l_epochs = [0] * (num_epochs + 1), [0] * (num_epochs + 1)
    all_best_epochs = []
    all_best_loss = []
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls, best_epoch, best_loss = train(
            net, *data, num_epochs, learning_rate, regula_str, l1_ratio, batch_size, is_optuna)

        all_best_epochs.append(best_epoch)
        all_best_loss.append(best_loss)
        if not is_optuna:
            train_l_epochs = [x + y / k for x, y in zip(train_l_epochs, train_ls)]
            valid_l_epochs = [x + y / k for x, y in zip(valid_l_epochs, valid_ls)]

    return train_l_epochs, valid_l_epochs, all_best_epochs, all_best_loss