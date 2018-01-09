# -*- coding:utf-8 -*-
"""
Tianchi AI contest. Intelligent Manufacturing.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Input
from keras.models import Model
from forestlayer.layers.layer import AutoGrowingCascadeLayer
from forestlayer.estimators.arguments import CompletelyRandomForest, RandomForest
from forestlayer.backend.backend import set_base_dir
from forestlayer.datasets.dataset import get_data_base
from forestlayer.utils.storage_utils import get_data_save_base, get_model_save_base
import os.path as osp
import pickle

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


label = train_test['Y'][:500]
train_test = train_test.fillna(0)
train_test.applymap(func)
feat_columns = list(train_test.columns.values)
feat_columns.remove("ID")
feat_columns.remove("Y")
# data 为除去ID和label后所有的特征
data = train_test[feat_columns]
label_id = train_test['ID']

cate_columns = data.select_dtypes(include=["object"]).columns
num_columns = data.select_dtypes(exclude=["object"]).columns
print("categorical feat num: ", len(cate_columns), "number feat num: ", len(num_columns))

feat_cate = data[cate_columns]
feat_nume = data[num_columns]

# categorical features: One-hot
feat_categorical_dummies = pd.get_dummies(feat_cate)
# number features: MinMax
feat_number_scale = pd.DataFrame(MinMaxScaler().fit_transform(feat_nume))
# Concatenate
feat_all = pd.concat([feat_number_scale, feat_categorical_dummies], axis=1)

if not osp.exists(osp.join(get_model_save_base(), 'feat_dim_100.pkl')):
    x_train, x_test, y_train, y_test = train_test_split(feat_all, feat_all, test_size=0.2, random_state=42)

    encoding_dim = 100
    input_ = Input(shape=(8051,))
    encoded = Dense(encoding_dim, activation='relu')(input_)
    decoded = Dense(8051, activation='sigmoid')(encoded)
    autoencoder = Model(input=input_, output=decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(x_train,
                    x_train,
                    nb_epoch=3,
                    batch_size=10,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    # 根据上面我们训练的自编码器，截取其中编码部分定义为编码模型
    encoder = Model(input=input_, output=encoded)
    # 对特征进行编码降维
    feat_dim_100 = encoder.predict(feat_all)

    with open(osp.join(get_model_save_base(), 'feat_dim_100.pkl'), 'wb') as f:
        pickle.dump(feat_dim_100, f)
else:
    with open(osp.join(get_model_save_base(), 'feat_dim_100.pkl'), 'rb') as f:
        feat_dim_100 = pickle.load(f)


est_configs = [
    CompletelyRandomForest(),
    CompletelyRandomForest(),
    RandomForest(),
    RandomForest()
]

data_save_dir = osp.join(get_data_save_base(), 'tianchi-intellijmanu')

agc = AutoGrowingCascadeLayer(task='regression',
                              est_configs=est_configs,
                              max_layers=1,
                              data_save_dir=data_save_dir,
                              keep_test_result=True)

agc.fit_transform(feat_dim_100[:500], label, feat_dim_100[500:])
result = agc.test_results

ret = pd.DataFrame()
ret["ID"] = testA["ID"]
ret["Y"] = result
ret.to_csv(osp.join(data_save_dir, "result.csv"), index=False, header=False)


print("Application end!")




