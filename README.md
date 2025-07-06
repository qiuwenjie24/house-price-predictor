#  House Price Prediction 

## 介绍

该项目使用 PyTorch 构建多层感知机（MLP）模型，通过 Optuna 自动调参预测房价，并支持特征重要性分析。

数据来源：https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

## 项目结构

house-price-predictor/
├── data/
│   ├── download.py             # 下载数据
│   └── load_data.py             # 加载并预处理数据
├── models/
│   └── predictor.py             # 模型定义（MLP）
├── utils/
│   ├── loss.py                    # 自定义 ElasticNetLoss
│   ├── metrics.py               # log_rmse, feature_importance
│   ├── train.py                   # 训练函数
│   ├── kfold.py                   # K折交叉验证逻辑
│   └── hyperopt.py             # Optuna 调参逻辑
├── main.py                        # 主程序入口：加载数据、调参、训练、预测、可视化化
├── predict.py                     # 预测
├── config.py                      # 配置超参数、路径等
├── requirements.txt            # 依赖文件
├── README.md                 # 项目说明文档
└── saved_models/
    └── MLP_model.pth           # 保存的模型权重



## 快速开始

```bash
pip install -r requirements.txt
python main.py
```



输出：

- `MLP_model.pth`: 模型权重
- `MLP_submission.csv`: 提交文件
- `loss_curve.png`: 损失曲线图



## 项目流程

1. **下载和加载数据集**
2. **数据预处理**
3. **K折交叉验证**
4. **模型选择和超参数调优**
5. **训练并保存模型**
6. **特征重要性分析**
7. **预测**



## 数据预处理

1. **异常值检测**

   因为房价分布属于重尾分布，所以使用柯西分布检测异常值。并删除这些异常的样本。对于其他的特征，不太好检测异常值，故暂不进行处理。

2. **缺失值填充**

   可以根据缺失值不同的比例采取不同的措施处理缺失值。这了为了简单，数值型特征用均值填充，类别型特征用众数填充。

3. **物理特征工程**

   将会有助于模型识别这些因素的影响。因此，根据`YearBuilt`特征构建两个新的特征，`YearSin`和`YearCos`。

   由于没有当地的经济周期或政策周期的相关信息，所以这里笼统地选择100年为一个周期，使得足够覆盖大多数房屋的建造年份范围，同时特征变化也比较平滑，避免周期过短导致特征变化过于频繁。



## 训练

**损失函数**

使用弹性网络正则化作为损失函数，即均方误差（MSE）+ L1+ L2正则，避免模型过拟合。

评估时，使用对数均方根误差(log RMSE)作为损失函数，原因是房价是重尾分布，我们关心的是相对偏差，而不是绝对偏差。



**优化器**

Adam优化算法和动态循环的学习率（CyclicLR学习率调度器），它能够接受比较大范围的学习率，通过调节最小学习率和迭代步数，减小初始学习率变化对损失的影响，更好找到最优解。



**超参数调优**

使用K折交叉验证和网络搜索，粗略寻找最优的超参数。

为了提高调参效率，使用Optuna工具实现自动化超参搜索，代替手动的网络搜索。



## 特征重要性分析

随机选取100个训练样本，通过计算每个特征的平均梯度，给出前10个对模型影响最大的特征。



## 后续改进

考虑其他模型的效果