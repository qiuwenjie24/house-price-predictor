# predict.py 加载模型 + 预测

import torch
import pandas as pd
from models import predictor
from data.load_data import load_and_preprocess
import os

def predict():
    # ---------- 参数设置 ----------
    model_path = 'saved_models/MLP_model.pth'
    train_path = 'house_data/kaggle_house_pred_train.csv'
    test_path = 'house_data/kaggle_house_pred_test.csv'
    submission_path = 'submission.csv'

    # ---------- 加载数据 ----------
    print("加载数据并进行预处理...")
    train_features, _, test_features, _ = load_and_preprocess(train_path, test_path)

    # ---------- 加载模型 ----------
    print("加载模型参数...")
    model = predictor.MLP(train_features.shape[1])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # ---------- 模型预测 ----------
    print("进行预测...")
    with torch.no_grad():
        preds = model(test_features).squeeze().numpy()

    # ---------- 保存预测结果 ----------
    print("保存预测结果...")
    test_df = pd.read_csv(test_path)
    submission = pd.DataFrame({"Id": test_df['Id'], "SalePrice": preds})
    submission.to_csv(submission_path, index=False)
    print(f"预测完成，结果已保存到 {submission_path}")


# 作为模块导入时不会执行下面代码
if __name__ == '__main__':
    predict()

