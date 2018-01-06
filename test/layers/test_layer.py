# -*- coding:utf-8 -*-
"""
Test Suites of layers.layer.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import numpy as np
import os.path as osp
from forestlayer.layers.window import Window, Pooling
from forestlayer.layers.layer import MultiGrainScanLayer, PoolingLayer, ConcatLayer, CascadeLayer, AutoGrowingCascadeLayer
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


def test_fit_transform(X, y_train, X_test, y_test):
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

    res_train, res_test = mgs.fit_transform(X, y_train, X_test, y_test)

    pools = [[Pooling(2, 2, "max"), Pooling(2, 2, "max")], [Pooling(2, 2, "max"), Pooling(2, 2, "max")]]

    poolayer = PoolingLayer(pools=pools)

    res_train, res_test = poolayer.fit_transform(res_train, None, res_test, None)

    # for i, r in enumerate(res):
    #     print('mgs result {}: '.format(i))
    #     for j in r:
    #         print(j.shape)

    concat_layer = ConcatLayer()

    res_train, res_test = concat_layer.fit_transform(res_train, None, res_test)


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
        'data_save_dir': osp.join(get_data_save_base(), 'test_layer', 'cascade'),
        'layer_id': 1,
        'keep_in_mem': True,
        'dtype': np.float32,
    }

    # cascade = CascadeLayer(est_configs=est_configs, kwargs=kwargs)
    #
    # res_train, res_test = cascade.fit_transform(res_train[0], y_train, res_test[0], y_test)
    #
    # print(res_train.shape, res_test.shape)

    auto_cascade_kwargs = {
        'early_stop_rounds': 4,
        'max_layers': 0,
        'stop_by_test': False,
        'n_classes': 10,
        'data_save_rounds': 4,
        'data_save_dir': osp.join(get_data_save_base(), 'test_layer', 'auto_cascade'),
        'keep_in_mem': True,
        'dtype': np.float32,
    }

    auto_cascade = AutoGrowingCascadeLayer(est_configs=est_configs, kwargs=auto_cascade_kwargs)

    res_train, res_test = auto_cascade.fit_transform(res_train, y_train, res_test)

    print(res_train.shape, res_test.shape)


def test_fit(X, y_train, X_test, y_test):
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

    res_train = mgs.fit(X, y_train)

    pools = [[Pooling(2, 2, "max"), Pooling(2, 2, "max")], [Pooling(2, 2, "max"), Pooling(2, 2, "max")]]

    poolayer = PoolingLayer(pools=pools)

    res_train = poolayer.fit(res_train, y_train)

    concat_layer = ConcatLayer()

    res_train = concat_layer.fit(res_train, None)

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
        'data_save_dir': osp.join(get_data_save_base(), 'test_layer', 'cascade'),
        'layer_id': 1,
        'keep_in_mem': True,
        'dtype': np.float32,
    }

    # cascade = CascadeLayer(est_configs=est_configs, kwargs=cascade_kwargs)
    #
    # res_train = cascade.fit(res_train[0], y_train)
    #
    # print(res_train.shape)

    auto_cascade_kwargs = {
        'early_stop_rounds': 4,
        'max_layers': 0,
        'stop_by_test': False,
        'n_classes': 10,
        'data_save_rounds': 4,
        'data_save_dir': osp.join(get_data_save_base(), 'test_layer', 'auto_cascade'),
        'keep_in_mem': True,
        'dtype': np.float32,
    }

    auto_cascade = AutoGrowingCascadeLayer(est_configs=est_configs, kwargs=auto_cascade_kwargs)

    res_train = auto_cascade.fit(res_train, y_train)

    #
    # print(res_train.shape, res_test.shape)


def test_predict(X, y_train, X_test, y_test):
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

    res_train = mgs.fit(X, y_train)
    predicted = mgs.predict(X_test)

    pools = [[Pooling(2, 2, "max"), Pooling(2, 2, "max")], [Pooling(2, 2, "max"), Pooling(2, 2, "max")]]

    poolayer = PoolingLayer(pools=pools)

    res_train = poolayer.fit(res_train, y_train)

    predicted = poolayer.predict(predicted)

    concat_layer = ConcatLayer()

    res_train = concat_layer.fit(res_train, None)

    predicted = concat_layer.predict(predicted)

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
        'data_save_dir': osp.join(get_data_save_base(), 'test_layer', 'cascade'),
        'layer_id': 1,
        'keep_in_mem': True,
        'dtype': np.float32,
    }

    # cascade = CascadeLayer(est_configs=est_configs, kwargs=cascade_kwargs)
    #
    # res_train = cascade.fit(res_train[0], y_train)
    #
    # predicted = cascade.predict(predicted[0])
    #
    # print(predicted[:125])

    auto_cascade_kwargs = {
        'early_stop_rounds': 4,
        'max_layers': 0,
        'stop_by_test': False,
        'n_classes': 10,
        'data_save_rounds': 4,
        'data_save_dir': osp.join(get_data_save_base(), 'test_layer', 'auto_cascade'),
        'keep_in_mem': True,
        'dtype': np.float32,
    }

    auto_cascade = AutoGrowingCascadeLayer(est_configs=est_configs, kwargs=auto_cascade_kwargs)

    res_train = auto_cascade.fit(res_train, y_train)
    predicted = auto_cascade.predict(predicted)
    print(predicted[:200])


# test_fit(X, y_train, X_test, y_test)


test_predict(X, y_train, X_test, y_test)


