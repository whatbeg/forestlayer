# -*- coding:utf-8 -*-
"""
Test Suites of layers.layer.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import numpy as np
from forestflow.layers.window import Window, Pooling
from forestflow.layers.layer import MultiGrainScanLayer, PoolingLayer, ConcatLayer
from forestflow.estimators import get_estimator_kfold
from keras.datasets import mnist


# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = np.reshape(x_train, (60000, -1, 28, 28))
X = X[:200, :, :, :]
y_train = y_train[:200]

print('X: ', X.shape, 'y: ', y_train.shape)

windows = [Window(7, 7, 2, 2, 0, 0), Window(11, 11, 2, 2, 0, 0)]

args = {
    'n_estimators': 500,
    'max_depth': 100,
    'n_jobs': -1,
    'min_samples_leaf': 10
}

rf1 = get_estimator_kfold('rf1', 3, 'RandomForestClassifier', est_args=args)
rf2 = get_estimator_kfold('rf2', 3, 'CompletelyRandomForestClassifier', est_args=args)

est_for_windows = [[rf1, rf2], [rf1.copy(), rf2.copy()]]

mgs = MultiGrainScanLayer(input_shape=X.shape,
                          batch_size=None,
                          dtype=np.float32,
                          windows=windows,
                          est_for_windows=est_for_windows,
                          n_class=10)

res = mgs.fit_transform(X, y_train)

# print('mgs result: ', )
# for i, r in enumerate(res):
#     print('mgs result {}: '.format(i))
#     for j in r:
#         print(j.shape)

pools = [[Pooling(2, 2, "max"), Pooling(2, 2, "max")], [Pooling(2, 2, "max"), Pooling(2, 2, "max")]]

poolayer = PoolingLayer(pools=pools)

res = poolayer.fit_transform(res)

# for i, r in enumerate(res):
#     print('mgs result {}: '.format(i))
#     for j in r:
#         print(j.shape)

concat_layer = ConcatLayer()

res = concat_layer.fit_transform(res)

# for i, r in enumerate(res):
#     print('mgs result {}: '.format(i))
#     print(r.shape)



