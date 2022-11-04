# -*- coding:utf-8 -*-
"""
作者：sunli
日期：2022年03月07日20:52
"""
 
# 使用train_test_split()将选取的固定大小的数据集，按照一定的比例，如9：1随机划分为训练集，测试集。
# 并分别将划分好的数据集进行写入到固定目录下
 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import pandas as pd
dataset_root = '/data2/liguanlin/Datasets/iHarmony4/HAdobe5k_1024/'
train_file = dataset_root + 'HAdobe5k_train.txt'
all_data=pd.read_csv(train_file)
train_data, test_data = train_test_split(all_data, train_size=0.9, test_size=0.1)
print(train_data)
print(test_data)
train_data.to_csv(dataset_root + "HAdobe5k_train_large.txt",index=False)
test_data.to_csv(dataset_root + "HAdobe5k_train_small.txt",index=False)
 