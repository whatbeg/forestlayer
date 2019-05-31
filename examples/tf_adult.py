# -*- coding:utf-8 -*-
"""
UCI_ADULT Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.datasets import uci_adult
from forestlayer.datasets import uci_yeast
from forestlayer.layers import Graph, AutoGrowingCascadeLayer
from forestlayer.utils.storage_utils import get_data_save_base
from forestlayer.estimators.estimator_configs import ExtraRandomForestConfig, RandomForestConfig, TensorForestConfig
import forestlayer as fl
import time
import numpy as np
import os.path as osp

# fl.init()

(x_train, y_train, x_test, y_test) = uci_yeast.load_data()  # uci_adult.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

start_time = time.time()

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train.shape[1], 'features')

num_classes = 10
num_features = 8  # 113
num_trees = 50
max_nodes = 1000
est_configs = [
	# TensorForestConfig(n_folds=1, num_classes=num_classes, num_features=num_features, num_trees=num_trees, max_nodes=max_nodes),
	# TensorForestConfig(n_folds=1, num_classes=num_classes, num_features=num_features, num_trees=num_trees, max_nodes=max_nodes),
	# TensorForestConfig(n_folds=1, num_classes=num_classes, num_features=num_features, num_trees=num_trees, max_nodes=max_nodes),
	# TensorForestConfig(n_folds=1, num_classes=num_classes, num_features=num_features, num_trees=num_trees, max_nodes=max_nodes)
	ExtraRandomForestConfig(n_folds=1, n_estimators=50),
    ExtraRandomForestConfig(n_folds=1, n_estimators=50),
    ExtraRandomForestConfig(n_folds=1, n_estimators=50),
    ExtraRandomForestConfig(n_folds=1, n_estimators=50),
]


agc = AutoGrowingCascadeLayer(est_configs=est_configs,
                              early_stopping_rounds=4,
                              max_layers=0,
                              stop_by_test=False,
                              n_classes=10,
                              data_save_rounds=0,
                              data_save_dir=osp.join(get_data_save_base(), 'uci_adult', 'auto_cascade'),
                              keep_in_mem=False,
                              distribute=False,
                              dis_level=0,
                              verbose_dis=False,
                              dtype=np.float32,
                              seed=0)

model = Graph()
model.add(agc)
model.fit_transform(x_train, y_train, x_test, y_test)

end_time = time.time()
print('time cost: {}'.format(end_time - start_time))
