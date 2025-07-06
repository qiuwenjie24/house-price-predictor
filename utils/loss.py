# utils/loss.py 自定义弹性网络损失

import torch
import torch.nn as nn
import torch.nn.functional as F

class ElasticNetLoss(nn.Module):
    def __init__(self, regula_str=0.0, l1_ratio=0.5):
        super().__init__()
        self.regula_str = regula_str
        self.l1_ratio = l1_ratio

    def forward(self, pred, target, model):
        mse = F.mse_loss(pred, target)
        l1_reg = 0.0
        l2_reg = 0.0
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_reg += torch.norm(param, 1)
                l2_reg += torch.norm(param, 2)
        return mse + self.regula_str * (self.l1_ratio * l1_reg + (1 - self.l1_ratio) * l2_reg)