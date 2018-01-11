# -*- coding:utf-8 -*-
"""
MNIST Example using factory methods.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.layers import Graph, AutoGrowingCascadeLayer
from forestlayer.layers.factory import *
from forestlayer.utils.storage_utils import get_data_save_base, get_model_save_base
from keras.datasets import mnist
import os.path as osp

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, -1, 28, 28)[:200, :, :, :]
x_test = x_test.reshape(10000, -1, 28, 28)[:100, :, :, :]
x_train = x_train / 255.0
x_test = x_test / 255.0
# y_train = y_train[:200]
# y_test = y_test[:100]

print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')

windows = [MGSWindow((7, 7), (2, 2)),
           MGSWindow((11, 11), (2, 2))]

est_for_windows = EstForWin2x2(min_samples_leaf=10)

mgs = MultiGrainScanLayer(windows=windows,
                          est_for_windows=est_for_windows,
                          n_class=10,
                          keep_in_mem=True)

pool = MaxPooling2x2Layer(2, 2)

concatlayer = ConcatLayer()

est_configs = [
    CompletelyRandomForest(),
    CompletelyRandomForest(),
    RandomForest(),
    RandomForest()
]

data_save_dir = osp.join(get_data_save_base(), 'mnist')
model_save_dir = osp.join(get_model_save_base(), 'mnist')

auto_cascade = AutoGrowingCascadeLayer(est_configs=est_configs,
                                       early_stopping_rounds=4,
                                       stop_by_test=True,
                                       n_classes=10,
                                       data_save_dir=data_save_dir,
                                       model_save_dir=model_save_dir,
                                       keep_in_mem=True)

model = Graph()
model.add(mgs)
model.add(pool)
model.add(concatlayer)
model.add(auto_cascade)
model.fit(x_train, y_train)
model.evaluate(x_test, y_test)
