# -*- coding:utf-8 -*-
"""
Test Suites of layers.graph.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import numpy as np
import os.path as osp
from forestlayer.layers.window import Window, Pooling
from forestlayer.layers.layer import MultiGrainScanLayer, PoolingLayer, ConcatLayer, CascadeLayer, AutoGrowingCascadeLayer
from forestlayer.layers.graph import Graph
from forestlayer.estimators import get_estimator_kfold
from forestlayer.utils.storage_utils import get_data_save_base
from keras.datasets import mnist


# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = np.reshape(x_train, (60000, -1, 28, 28))
X = X[:200, :, :, :]
y_train = y_train[:200]
X_test = np.reshape(x_test[:100], (100, -1, 28, 28))
y_test = y_test[:100]

print('X: ', X.shape, 'y: ', y_train.shape)
print('X_test: ', X_test.shape, 'y: ', y_test.shape)

windows = [Window(7, 7, 2, 2, 0, 0), Window(11, 11, 2, 2, 0, 0)]

args = {
    'n_estimators': 500,
    'max_depth': 100,
    'n_jobs': -1,
    'min_samples_leaf': 10
}

rf1 = get_estimator_kfold('rf1', 3, 'RF', est_args=args)
rf2 = get_estimator_kfold('rf2', 3, 'CRF', est_args=args)

est_for_windows = [[rf1, rf2], [rf1.copy(), rf2.copy()]]

mgs = MultiGrainScanLayer(batch_size=None,
                          dtype=np.float32,
                          windows=windows,
                          est_for_windows=est_for_windows,
                          n_class=10)

pools = [[Pooling(2, 2, "max"), Pooling(2, 2, "max")], [Pooling(2, 2, "max"), Pooling(2, 2, "max")]]

poolayer = PoolingLayer(pools=pools)

concat_layer = ConcatLayer()


def get_est_args(est_type):
    est_args = {
        'est_type': est_type,
        'n_folds': 3,
        'n_estimators': 500,
        'max_depth': 100,
        'n_jobs': -1,
        'min_samples_leaf': 10
    }
    return est_args


est_configs = [
    get_est_args('CRF'),
    get_est_args('CRF'),
    get_est_args('RF'),
    get_est_args('RF')
]

cascade_kwargs = {
    'n_classes': 10,
    'data_save_dir': osp.join(get_data_save_base(), 'test_graph', 'cascade'),
    'layer_id': 1,
    'keep_in_mem': True,
    'dtype': np.float32,
}

auto_cascade_kwargs = {
    'early_stop_rounds': 4,
    'max_layers': 0,
    'stop_by_test': False,
    'n_classes': 10,
    'data_save_rounds': 4,
    'data_save_dir': osp.join(get_data_save_base(), 'test_graph', 'auto_cascade'),
    'keep_in_mem': True,
    'dtype': np.float32,
}

auto_cascade = AutoGrowingCascadeLayer(est_configs=est_configs, kwargs=auto_cascade_kwargs)

model = Graph()
model.add(mgs)
model.add(poolayer)
model.add(concat_layer)
model.add(auto_cascade)
# model.fit_transform(X, y_train, X_test)
model = model.fit(X, y_train)
