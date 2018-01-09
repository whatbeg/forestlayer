# -*- coding:utf-8 -*-
"""
Tianchi AI contest. Intelligent Manufacturing.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.ensemble import GradientBoostingRegressor
from forestlayer.backend.backend import set_base_dir
from forestlayer.datasets.dataset import get_data_base
import os.path as osp

set_base_dir(osp.expanduser(osp.join('~', 'forestlayer')))

train = pd.read_excel(osp.join(get_data_base(), 'tianchi-intellijmanufacturing/train.xlsx'))
testA = pd.read_excel(osp.join(get_data_base(), 'tianchi-intellijmanufacturing/test_A.xlsx'))

train_test = pd.concat([train, testA], axis=0, ignore_index=True)


def func(x):
    try:
        return float(x)
    except:
        if x is None:
            return 0
        else:
            return x


# 获取标签
label = pd.DataFrame(train_test)['Y'][:500]
print(label)
train_test = train_test.fillna(0)
train_test.applymap(func)
feat_columns = list(train_test.columns.values)
feat_columns.remove("ID")
feat_columns.remove("Y")
# data 为除去ID和label后所有的特征
data = train_test[feat_columns]


# # 类别型特征
# cate_columns = data.select_dtypes(include=["object"]).columns
# # 数值型特征
# num_columns = data.select_dtypes(exclude=["object"]).columns
# print("cate feat num: ", len(cate_columns),"num feat num: ", len(num_columns))
# feat_cate = data[cate_columns]
# feat_nume = data[num_columns]
# # 类别型特征独热编码
# feat_cate_dummies = pd.get_dummies(feat_cate)
# # 数值型特征最值归一化
# feat_nume_scale = pd.DataFrame(MinMaxScaler().fit_transform(feat_nume))
# # 拼接上述两种特征
# feat_all = pd.concat([feat_nume_scale, feat_cate_dummies], axis=1)
#
# # 数据切分
# x_train, x_test, y_train, y_test = train_test_split(feat_all, feat_all, test_size=0.2, random_state=42)
# # 自编码维度
# encoding_dim = 100
# # 输入层
# input_ = Input(shape=(8051,))
# # 编码层
# encoded = Dense(encoding_dim, activation='relu')(input_)
# # 解码层
# decoded = Dense(8051, activation='sigmoid')(encoded)
# # 自编码器模型
# autoencoder = Model(input=input_, output=decoded)
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# autoencoder.fit(x_train,
#                 x_train,
#                 nb_epoch=500,
#                 batch_size=10,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))
#
# # 根据上面我们训练的自编码器，截取其中编码部分定义为编码模型
# encoder = Model(input=input_, output=encoded)
# # 对特征进行编码降维
# feat_dim_100 = encoder.predict(feat_all)
#
# # 数据划分，训练集，验证集，测试集为后100条（测试集用来生成提交结果）
# x_tr, x_te, y_tr, y_te = train_test_split(feat_dim_100[:500], label, test_size=0.2, random_state=42)
#
# # 这里用的参数没有调过，都是随便取的，结果还有上升空间
# gbdt = GradientBoostingRegressor(
#           loss='ls'
#             , learning_rate=0.01
#             , n_estimators=50
#             , subsample=1
#             , min_samples_split=2
#             , min_samples_leaf=1
#             , max_depth=4
#             , init=None
#             , random_state=None
#             , max_features=None
#             , alpha=0.9
#             , verbose=0
#             , max_leaf_nodes=None
#             , warm_start=False
#     )
# gbdt.fit(x_tr, y_tr)
#
#
# # 懒得找MSE的API，自己随便写了一个，反正只有500条不用管效率
# def loss(list1, list2):
#     import math
#     _sum = 0
#     for k, v in zip(list1, list2):
#      _sum += math.pow(k-v, 2)
#     return _sum / len(list1)
#
#
# # 预测验证集label
# label_test = gbdt.predict(x_te)
# # 计算线下误差,线下误差0.04192
# loss(label_test, y_te)
#
# # 测试集特征为feat_all的后一百条
# result = gbdt.predict(feat_dim_100[500:])
# ret = pd.DataFrame()
# ret["ID"] = testA["ID"]
# ret["Y"] = result
# ret.to_csv("result.csv", index=False, header=False)
#
#
#
#









