# utils/metrics.py log-RMSE与特征重要性分析

import torch
import torch.nn.functional as F

# 计算对数均方根误差（log RMSE）
def log_rmse(net, features, labels):
    preds = torch.clamp(net(features), min=1e-6)
    labels = torch.clamp(labels, min=1e-6)
    rmse = torch.sqrt(F.mse_loss(torch.log(preds), torch.log(labels)))
    return rmse.item()

# 基于梯度的特征重要性分析
def feature_importance(net, features):
    inputs = features.clone().requires_grad_(True)
    outputs = net(inputs)
    grads = torch.autograd.grad(outputs, inputs,
                                grad_outputs=torch.ones_like(outputs))[0]
    return torch.mean(torch.abs(grads), dim=0)