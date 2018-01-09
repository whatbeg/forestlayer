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
from forestlayer.layers.layer import MultiGrainScanLayer, PoolingLayer, ConcatLayer, AutoGrowingCascadeLayer
from forestlayer.layers.graph import Graph
from forestlayer.estimators.arguments import RandomForest, CompletelyRandomForest, GBDT
from forestlayer.utils.storage_utils import get_data_save_base
from forestlayer.utils.log_utils import get_logger
from keras.datasets import mnist

LOGGER = get_logger('test.layer.graph')


def mnist_test_graph():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = np.reshape(x_train, (60000, -1, 28, 28))
    X_train = X[:200, :, :, :]
    y_train = y_train[:200]
    X_test = np.reshape(x_test[:100], (100, -1, 28, 28))
    y_test = y_test[:100]

    print('X_train: ', X_train.shape, 'y: ', y_train.shape)
    print('X_test: ', X_test.shape, 'y: ', y_test.shape)

    est_configs = [
        CompletelyRandomForest(n_estimators=40),
        CompletelyRandomForest(n_estimators=40),
        RandomForest(n_estimators=40),
        RandomForest(n_estimators=40)
    ]

    def _init():
        windows = [Window(7, 7, 2, 2, 0, 0), Window(11, 11, 2, 2, 0, 0)]

        rf1 = CompletelyRandomForest(min_samples_leaf=10)
        rf2 = RandomForest(min_samples_leaf=10)

        est_for_windows = [[rf1, rf2], [rf1, rf2]]

        mgs = MultiGrainScanLayer(batch_size=None,
                                  dtype=np.float32,
                                  windows=windows,
                                  est_for_windows=est_for_windows,
                                  n_class=10)

        pools = [[Pooling(2, 2, "max"), Pooling(2, 2, "max")], [Pooling(2, 2, "max"), Pooling(2, 2, "max")]]

        poolayer = PoolingLayer(pools=pools)

        concat_layer = ConcatLayer()

        auto_cascade = AutoGrowingCascadeLayer(est_configs=est_configs,
                                               early_stopping_rounds=2,
                                               stop_by_test=False,
                                               data_save_rounds=4,
                                               n_classes=10,
                                               data_save_dir=osp.join(get_data_save_base(),
                                                                      'test_graph', 'auto_cascade'))
        return mgs, poolayer, concat_layer, auto_cascade

    def test_fit():
        print("test fit")
        mgs, poolayer, concat_layer, auto_cascade = _init()
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit(X_train, y_train)
        LOGGER.info('test fit passed!')

    def test_fit_transform():
        print("test fit_transform")
        mgs, poolayer, concat_layer, auto_cascade = _init()
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit_transform(X_train, y_train, X_test)
        LOGGER.info('test fit_transform(x, y, x_test) passed!')

    def test_fit_transform2():
        print("test fit_transform2")
        mgs, poolayer, concat_layer, auto_cascade = _init()
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit_transform(X_train, y_train, X_test, y_test)
        LOGGER.info('test fit_transform(x, y, x_test, y_test) passed!')

    def test_transform():
        print("test transform")
        mgs, poolayer, concat_layer, auto_cascade = _init()
        mgs.keep_in_mem = True
        auto_cascade.keep_in_mem = True
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit(X_train, y_train)
        model.transform(X_test)
        LOGGER.info('test transform passed!')

    def test_fit_predict():
        print("test fit and predict")
        mgs, poolayer, concat_layer, auto_cascade = _init()
        mgs.keep_in_mem = True
        auto_cascade.keep_in_mem = True
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit(X_train, y_train)
        model.predict(X_test)
        LOGGER.info('test fit_predict passed!')

    def test_fit_evaluate():
        print("test fit and evaluate")
        mgs, poolayer, concat_layer, auto_cascade = _init()
        mgs.keep_in_mem = True
        auto_cascade.keep_in_mem = True
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit(X_train, y_train)
        model.evaluate(X_test, y_test)
        LOGGER.info('test fit_evaluate passed!')

    test_fit()
    # test_fit_transform()
    test_fit_transform2()
    test_transform()
    test_fit_predict()
    test_fit_evaluate()


mnist_test_graph()


