# data/load_data.py 数据加载与预处理

import pandas as pd
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import cauchy


def detect_outliers(data):
    """使用柯西分布检测异常值"""
    params = cauchy.fit(data['SalePrice'])
    threshold = params[0] + 3*params[1]   # params[0]：位置参数，类似均值   params[1]：尺度参数，类似标准差
    return data[data['SalePrice'] > threshold].index


# 加载和预处理数据
def load_and_preprocess(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 异常值检测（柯西分布）
    outlier_idx = detect_outliers(train_data)
    train_data = train_data.drop(outlier_idx)  # 删除异常样本

    train_features = train_data.iloc[:, 1:-1]
    train_labels = train_data.iloc[:, -1]
    test_features = test_data.iloc[:, 1:]

    num_features = train_features.select_dtypes(exclude='object').columns
    cat_features = train_features.select_dtypes(include='object').columns

    # 缺失值处理
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    train_features[num_features] = num_imputer.fit_transform(train_features[num_features])
    test_features[num_features] = num_imputer.transform(test_features[num_features])

    train_features[cat_features] = cat_imputer.fit_transform(train_features[cat_features])
    test_features[cat_features] = cat_imputer.transform(test_features[cat_features])

    # 特征工程（年建造周期）
    # for df in [train_features, test_features]:
    #     df['YearSin'] = np.sin(2 * np.pi * df['YearBuilt'] / 100)
    #     df['YearCos'] = np.cos(2 * np.pi * df['YearBuilt'] / 100)

    # 标准化数值特征
    # num_features = num_features.drop(['YearSin', 'YearCos'], errors='ignore')
    scaler = StandardScaler()
    train_features[num_features] = scaler.fit_transform(train_features[num_features])
    test_features[num_features] = scaler.transform(test_features[num_features])

    # 类别特征独热编码
    train_features = pd.get_dummies(train_features, dummy_na=True)
    test_features = pd.get_dummies(test_features, dummy_na=True)
    test_features = test_features.reindex(columns=train_features.columns, fill_value=0)

    train_features = train_features.astype('float32')
    test_features = test_features.astype('float32')

    feature_names = train_features.columns.to_numpy()

    # 一定要指定为浮点数的张量 dtype=torch.float32
    return (torch.tensor(train_features.values, dtype=torch.float32),
            torch.tensor(train_labels.values.reshape(-1, 1), dtype=torch.float32),
            torch.tensor(test_features.values, dtype=torch.float32),
            feature_names)