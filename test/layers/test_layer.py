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
import time
from forestlayer.layers.window import Window, Pooling
from forestlayer.layers.layer import MultiGrainScanLayer, PoolingLayer, ConcatLayer, CascadeLayer, AutoGrowingCascadeLayer
from forestlayer.estimators import get_estimator_kfold
from forestlayer.layers.graph import Graph
from forestlayer.utils.storage_utils import get_data_save_base
from keras.datasets import mnist
from forestlayer.datasets import uci_adult


def MNIST_based_test():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = np.reshape(x_train, (60000, -1, 28, 28))
    X = X[:200, :, :, :]
    y_train = y_train[:200]
    X_test = np.reshape(x_test[:100], (100, -1, 28, 28))
    y_test = y_test[:100]

    print('X: ', X.shape, 'y: ', y_train.shape)
    print('X_test: ', X_test.shape, 'y: ', y_test.shape)

    def _init():
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
            'data_save_dir': osp.join(get_data_save_base(), 'test_layer', 'cascade'),
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
            'data_save_dir': osp.join(get_data_save_base(), 'test_layer', 'auto_cascade'),
            'keep_in_mem': True,
            'dtype': np.float32,
        }

        cascade = CascadeLayer(est_configs=est_configs, kwargs=cascade_kwargs)

        auto_cascade = AutoGrowingCascadeLayer(est_configs=est_configs, kwargs=auto_cascade_kwargs)

        return mgs, poolayer, concat_layer, auto_cascade

    def test_fit_transform():
        mgs, poolayer, concat_layer, auto_cascade = _init()

        res_train, res_test = mgs.fit_transform(X, y_train, X_test, y_test)

        res_train, res_test = poolayer.fit_transform(res_train, None, res_test, None)

        res_train, res_test = concat_layer.fit_transform(res_train, None, res_test)

        res_train, res_test = auto_cascade.fit_transform(res_train, y_train, res_test)

        print(res_train.shape, res_test.shape)

    def test_fit():
        mgs, poolayer, concat_layer, auto_cascade = _init()
        res_train = mgs.fit(X, y_train)

        res_train = poolayer.fit(res_train, y_train)

        res_train = concat_layer.fit(res_train, None)

        res_train = auto_cascade.fit(res_train, y_train)

    def test_predict():
        mgs, poolayer, concat_layer, auto_cascade = _init()
        res_train = mgs.fit(X, y_train)
        predicted = mgs.predict(X_test)

        res_train = poolayer.fit(res_train, y_train)
        predicted = poolayer.predict(predicted)

        res_train = concat_layer.fit(res_train, None)
        predicted = concat_layer.predict(predicted)

        res_train, _ = auto_cascade.fit(res_train, y_train)
        auto_cascade.evaluate(predicted, y_test)

    test_fit_transform()

    test_fit()

    test_predict()


def UCI_ADULT_based_test():
    start_time = time.time()
    (x_train, y_train) = uci_adult.load_data("adult.data")
    (x_test, y_test) = uci_adult.load_data("adult.test")

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(x_train.shape[1], 'features')

    end_time = time.time()
    print('time cost: {}'.format(end_time - start_time))

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
        'n_classes': 2,
        'data_save_dir': osp.join(get_data_save_base(), 'test_layer', 'cascade'),
        'layer_id': 1,
        'keep_in_mem': True,
        'dtype': np.float32,
    }

    auto_cascade_kwargs = {
        'early_stop_rounds': 2,
        'max_layers': 0,
        'stop_by_test': False,
        'n_classes': 2,
        'data_save_rounds': 4,
        'data_save_dir': osp.join(get_data_save_base(), 'uci_adult', 'auto_cascade'),
        'keep_in_mem': True,
        'dtype': np.float32,
    }

    gc = CascadeLayer(est_configs=est_configs, kwargs=cascade_kwargs)

    agc = AutoGrowingCascadeLayer(est_configs=est_configs, kwargs=auto_cascade_kwargs)

    def test_uci_graph():
        model = Graph()
        model.add(agc)
        model.fit_transform(x_train, y_train, x_test, y_test)

    def test_fit_predict():
        agc.fit(x_train, y_train)
        agc.evaluate(x_test, y_test)

    def test_graph_fit_evaluate():
        model = Graph()
        model.add(agc)
        model.fit(x_train, y_train)
        model.evaluate(x_test, y_test)

    def test_graph_transform():
        model = Graph()
        model.add(agc)
        model.fit(x_train, y_train)
        model.transform(x_test)

    # test_uci_graph()
    # test_fit_predict()
    # test_graph_fit_evaluate()
    test_graph_transform()


# UCI_ADULT_based_test()


