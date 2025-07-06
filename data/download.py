# data/download.py 数据下载

import os
import requests

DATA_HUB = {
    'kaggle_house_train': ("http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv",
                            '585e9cc93e70b39160e7921475f9bcd7d31219ce'),
    'kaggle_house_test': ("http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv",
                           'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
}

def download(name, cache_dir='./house_data'):
    """下载一个DATA_HUB中的文件，返回路径+本地文件名"""
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        return fname
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname