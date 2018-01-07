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
from forestlayer.utils.log_utils import get_logger
from keras.datasets import mnist

LOGGER = get_logger('test.layer.graph')


def MNIST_test_graph():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = np.reshape(x_train, (60000, -1, 28, 28))
    X_train = X[:200, :, :, :]
    y_train = y_train[:200]
    X_test = np.reshape(x_test[:100], (100, -1, 28, 28))
    y_test = y_test[:100]

    print('X: ', X.shape, 'y: ', y_train.shape)
    print('X_test: ', X_test.shape, 'y: ', y_test.shape)

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
        'early_stop_rounds': 2,
        'max_layers': 0,
        'stop_by_test': False,
        'n_classes': 10,
        'data_save_rounds': 4,
        'data_save_dir': osp.join(get_data_save_base(), 'test_graph', 'auto_cascade'),
        'keep_in_mem': True,
        'dtype': np.float32,
    }

    args = {
        'n_estimators': 500,
        'max_depth': 100,
        'n_jobs': -1,
        'min_samples_leaf': 10
    }

    def _init():
        windows = [Window(7, 7, 2, 2, 0, 0), Window(11, 11, 2, 2, 0, 0)]

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

        auto_cascade = AutoGrowingCascadeLayer(est_configs=est_configs, kwargs=auto_cascade_kwargs)
        return mgs, poolayer, concat_layer, auto_cascade

    def test_fit():
        mgs, poolayer, concat_layer, auto_cascade = _init()
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit(X_train, y_train)
        LOGGER.info('test fit passed!')

    def test_fit_transform():
        mgs, poolayer, concat_layer, auto_cascade = _init()
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit_transform(X_train, y_train, X_test)
        LOGGER.info('test fit_transform(x, y, x_test) passed!')

    def test_fit_transform2():
        mgs, poolayer, concat_layer, auto_cascade = _init()
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit_transform(X_train, y_train, X_test, y_test)
        LOGGER.info('test fit_transform(x, y, x_test, y_test) passed!')

    def test_transform():
        mgs, poolayer, concat_layer, auto_cascade = _init()
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit(X_train, y_train)
        model.transform(X_test)
        LOGGER.info('test transform passed!')

    def test_fit_predict():
        mgs, poolayer, concat_layer, auto_cascade = _init()
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit(X_train, y_train)
        model.predict(X_test)
        LOGGER.info('test fit_predict passed!')

    def test_fit_evaluate():
        mgs, poolayer, concat_layer, auto_cascade = _init()
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit(X_train, y_train)
        model.evaluate(X_test, y_test)
        LOGGER.info('test fit_evaluate passed!')

    test_fit_transform()
    test_fit_transform2()
    test_fit()
    test_transform()
    test_fit_predict()
    test_fit_evaluate()


MNIST_test_graph()


