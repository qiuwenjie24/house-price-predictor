# config.py
"""
项目配置文件：集中管理超参数、路径和随机种子等配置
"""

# 数据相关路径
TRAIN_DATA_PATH = './house_data/kaggle_house_pred_train.csv'
TEST_DATA_PATH = './house_data/kaggle_house_pred_test.csv'
MODEL_PATH = './MLP_model.pth'
SUBMISSION_PATH = './MLP_submission.csv'

# 随机种子
RANDOM_SEED = 42

# 训练超参数（可被 main.py 动态覆盖）
DEFAULT_CONFIG = {
    'k': 5,
    'num_epochs': 300,
    'learning_rate': 0.1,
    'regula_str': 0.0,
    'l1_ratio': 0.0,
    'batch_size': 64
}

# 学习率调度器参数
CYCLIC_LR_BASE_DIVISOR = 400  # base_lr = lr / 400

# 可选调参范围（用于 Optuna）
SEARCH_SPACE = {
    'k': (5, 10),
    'lr': (1e-1, 100),
    'regula_str': [i*0.05 for i in range(10)],
    'l1_ratio': [i*0.1 for i in range(5)],
    'batch_size_exponent': (5, 8)  # 对应 batch_size=32~256
}