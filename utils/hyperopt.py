# utils/hyperopt.py Optuna超参数优化逻辑

import optuna
import numpy as np
from utils.kfold import k_fold

def objective(trial, train_features, train_labels, get_net):
    k = trial.suggest_int("k", 5, 10)
    num_epochs = 300
    lr = trial.suggest_float("lr", 1e-1, 100, log=True)
    regula_str = trial.suggest_categorical("regula_str", [i * 0.05 for i in range(10)])
    l1_ratio = trial.suggest_categorical("l1_ratio", [i * 0.1 for i in range(5)])
    exponent  = trial.suggest_int("batch_size_exponent", 5, 8)
    batch_size = 2 ** exponent   # 2^4=16, 2^8=256

    _, _, best_epochs, best_losses = k_fold(k, train_features, train_labels,
                                            num_epochs, lr, regula_str,
                                            l1_ratio, batch_size, get_net,
                                            is_optuna=True)

    final_best_epoch = int(np.mean(best_epochs))
    final_best_loss = np.mean(best_losses)
    trial.set_user_attr("best_epoch", final_best_epoch)
    trial.set_user_attr("max_epoch", num_epochs)
    return final_best_loss

def run_optimization(train_features, train_labels, get_net, n_trials=20):
    '''创建 Study 用于记录所有试验的结果，并启动自动调参'''
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_features, train_labels, get_net),
                   n_trials=n_trials)
    return study