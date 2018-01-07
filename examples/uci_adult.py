# -*- coding:utf-8 -*-
"""
UCI_ADULT Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.datasets import uci_adult
from forestlayer.layers import Graph, AutoGrowingCascadeLayer
from forestlayer.utils.storage_utils import get_data_save_base
import time
import numpy as np
import os.path as osp

start_time = time.time()
(x_train, y_train) = uci_adult.load_data("adult.data")
(x_test, y_test) = uci_adult.load_data("adult.test")

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train.shape[1], 'features')


def get_est_args(est_type):
    est_args = {
        'est_type': est_type,
        'n_folds': 3,
        'n_estimators': 500,
        'max_depth': 100,
        'n_jobs': -1
    }
    return est_args


est_configs = [
    get_est_args('CRF'),
    get_est_args('CRF'),
    get_est_args('RF'),
    get_est_args('RF')
]

auto_cascade_kwargs = {
    'early_stop_rounds': 4,
    'max_layers': 0,
    'stop_by_test': True,
    'n_classes': 2,
    'data_save_rounds': 4,
    'data_save_dir': osp.join(get_data_save_base(), 'uci_adult', 'auto_cascade'),
    'keep_in_mem': False,
    'dtype': np.float32,
}

agc = AutoGrowingCascadeLayer(est_configs=est_configs, kwargs=auto_cascade_kwargs)

model = Graph()
model.add(agc)
model.build()
model.fit_transform(x_train, y_train, x_test, y_test)

end_time = time.time()
print('time cost: {}'.format(end_time - start_time))

