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
import copy
from forestlayer.layers.window import Window, Pooling
from forestlayer.layers.layer import MultiGrainScanLayer, PoolingLayer, ConcatLayer, CascadeLayer, AutoGrowingCascadeLayer
from forestlayer.estimators.arguments import RandomForest, CompletelyRandomForest, GBDT
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

        est_configs = [
            CompletelyRandomForest(n_estimators=40),
            CompletelyRandomForest(n_estimators=40),
            RandomForest(n_estimators=40),
            RandomForest(n_estimators=40)
        ]

        cascade = CascadeLayer(est_configs=est_configs,
                               n_classes=10,
                               keep_in_mem=True,
                               data_save_dir=osp.join(get_data_save_base(), 'test_layer', 'cascade'))

        auto_cascade = AutoGrowingCascadeLayer(est_configs=est_configs,
                                               early_stopping_rounds=3,
                                               data_save_rounds=4,
                                               stop_by_test=True,
                                               n_classes=10,
                                               data_save_dir=osp.join(get_data_save_base(),
                                                                      'test_layer', 'auto_cascade'))

        return mgs, poolayer, concat_layer, cascade, auto_cascade

    def test_fit_transform():
        print('test fit_transform')

        mgs, poolayer, concat_layer, cascade, auto_cascade = _init()

        res_train, res_test = mgs.fit_transform(X, y_train, X_test, y_test)

        res_train, res_test = poolayer.fit_transform(res_train, None, res_test, None)

        res_train, res_test = concat_layer.fit_transform(res_train, None, res_test)

        res_train, res_test = auto_cascade.fit_transform(res_train, y_train, res_test)

        print(res_train.shape, res_test.shape)

    def test_fit():
        print('test fit')

        mgs, poolayer, concat_layer, cascade, auto_cascade = _init()
        res_train = mgs.fit(X, y_train)

        res_train = poolayer.fit(res_train, y_train)

        res_train = concat_layer.fit(res_train, None)

        res_train = auto_cascade.fit(res_train, y_train)

    def test_predict():
        print('test predict')

        mgs, poolayer, concat_layer, cascade, auto_cascade = _init()
        mgs.keep_in_mem = True
        res_train = mgs.fit(X, y_train)
        predicted = mgs.predict(X_test)

        res_train = poolayer.fit(res_train, y_train)
        predicted = poolayer.predict(predicted)

        res_train = concat_layer.fit(res_train, None)
        predicted = concat_layer.predict(predicted)
        auto_cascade.keep_in_mem = True
        res_train = auto_cascade.fit(res_train, y_train)
        auto_cascade.evaluate(predicted, y_test)

    test_fit_transform()

    test_fit()

    test_predict()


def UCI_ADULT_based_test():
    start_time = time.time()
    (x_train, y_train, x_test, y_test) = uci_adult.load_data()

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(x_train.shape[1], 'features')

    end_time = time.time()
    print('time cost: {}'.format(end_time - start_time))

    def _init():
        est_configs = [
            CompletelyRandomForest(n_estimators=40),
            CompletelyRandomForest(n_estimators=40),
            RandomForest(n_estimators=40),
            RandomForest(n_estimators=40)
        ]

        gc = CascadeLayer(est_configs=est_configs,
                          n_classes=2,
                          data_save_dir=osp.join(get_data_save_base(), 'test_layer', 'cascade'))

        agc = AutoGrowingCascadeLayer(est_configs=est_configs,
                                      early_stopping_rounds=2,
                                      stop_by_test=False,
                                      data_save_rounds=4,
                                      n_classes=2,
                                      data_save_dir=osp.join(get_data_save_base(),
                                                             'test_layer', 'auto_cascade'))
        return gc, agc

    def test_uci_graph():
        print('test uci_graph')
        gc, agc = _init()
        model = Graph()
        model.add(agc)
        model.fit_transform(x_train, y_train, x_test, y_test)

    def test_fit_predict():
        print('test fit and predict')
        gc, agc = _init()
        agc.keep_in_mem = True
        agc.fit(x_train, y_train)
        agc.evaluate(x_test, y_test)

    def test_graph_fit_evaluate():
        print('test fit and evaluate')
        gc, agc = _init()
        agc.keep_in_mem = True
        model = Graph()
        model.add(agc)
        model.fit(x_train, y_train)
        model.evaluate(x_test, y_test)

    def test_graph_transform():
        print('test graph transform')
        gc, agc = _init()
        agc.keep_in_mem = True
        model = Graph()
        model.add(agc)
        model.fit(x_train, y_train)
        model.transform(x_test)

    test_uci_graph()
    test_fit_predict()
    test_graph_fit_evaluate()
    test_graph_transform()


MNIST_based_test()
UCI_ADULT_based_test()


