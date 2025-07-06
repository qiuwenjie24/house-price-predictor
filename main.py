# main.py 项目主入口脚本

import os
import joblib
import torch
import matplotlib.pyplot as plt
import pandas as pd

from data.download import download, DATA_HUB
from data.load_data import load_and_preprocess
from models import predictor
from utils.train import train
from utils.kfold import k_fold
from utils.metrics import feature_importance
from utils.hyperopt import run_optimization

# 下载数据
train_path = download('kaggle_house_train')
test_path = download('kaggle_house_test')

# 预处理
train_features, train_labels, test_features, feature_names = load_and_preprocess(train_path, test_path)

# 获取模型（使模型函数化）
get_net = lambda: predictor.MLP(train_features.shape[1])


# 超参数调优和保存
study_path = 'saved_models/MLP_study.pkl'
if os.path.exists(study_path):  
    '''若已经保存过调参结果，则直接加载；否则运行自动调参函数'''
    study = joblib.load(study_path)
else:
    study = run_optimization(train_features, train_labels, get_net, n_trials=200)
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(study, study_path) # 保存调参记录

best_trial = study.best_trial
params = best_trial.params
batch_size = 2 ** params['batch_size_exponent']
num_epochs = best_trial.user_attrs['best_epoch']

# k折交叉验证和绘图（最优超参数下训练，并绘制训练损失和验证损失）
train_l, valid_l, _, _ = k_fold(
    k = params['k'],
    X_train = train_features,
    y_train = train_labels,
    num_epochs = num_epochs,
    learning_rate = params['lr'],
    regula_str = params['regula_str'],
    l1_ratio = params['l1_ratio'],
    batch_size = batch_size,
    get_net = get_net,
    is_optuna = False
)

plt.plot(range(1, num_epochs+1), train_l, label='train')
plt.plot(range(1, num_epochs+1), valid_l, label='valid')
plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('log RMSE')
plt.grid(True)
plt.legend()
plt.title('K-fold average loss')
plt.savefig('train_valid_curve.png')
print(f"train loss: {train_l[-1]:.2f}, valid loss: {valid_l[-1]:.2f}")

# 最终模型训练并保存
net = get_net()
train_ls, _, _, _ = train(
    net, train_features, train_labels, None, None,
    num_epochs=num_epochs, learning_rate=params['lr'],
    regula_str=params['regula_str'], l1_ratio=params['l1_ratio'],
    batch_size=batch_size
)
print(f"final train loss: {train_ls[-1]:.2f}")
torch.save(net.state_dict(), 'saved_models/MLP_model.pth')

# 特征重要性分析（取前100个样本）
imp = feature_importance(net, train_features[:100])
topk = torch.topk(imp, 10)
plt.figure(figsize=(10,6))
plt.barh(feature_names[topk.indices], imp[topk.indices])
plt.title('Gradient-based Feature Importance')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png')

# 生成预测文件
net.eval()
preds = net(test_features).detach().numpy().reshape(-1)
test_df = pd.read_csv(test_path)
submission = pd.DataFrame({"Id": test_df['Id'], "SalePrice": preds})
submission.to_csv('submission.csv', index=False)
print("预测完成，结果保存在 submission.csv")

