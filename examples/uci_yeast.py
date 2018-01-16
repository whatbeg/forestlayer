# -*- coding:utf-8 -*-
"""
UCI_YEAST Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.datasets import uci_yeast
from forestlayer.estimators.estimator_configs import ExtraRandomForestConfig, RandomForestConfig
from forestlayer.layers.layer import AutoGrowingCascadeLayer
from forestlayer.layers.graph import Graph

(x_train, y_train, x_test, y_test) = uci_yeast.load_data()

print('x_train shape: {}'.format(x_train.shape))
print('x_test.shape: {}'.format(x_test.shape))

est_configs = [
    ExtraRandomForestConfig(),
    ExtraRandomForestConfig(),
    ExtraRandomForestConfig(),
    ExtraRandomForestConfig(),
    RandomForestConfig(),
    RandomForestConfig(),
    RandomForestConfig(),
    RandomForestConfig()
]

auto_cascade = AutoGrowingCascadeLayer(est_configs=est_configs,
                                       early_stopping_rounds=4,
                                       n_classes=10)

model = Graph()
model.add(auto_cascade)
model.fit_transform(x_train, y_train, x_test, y_test)



