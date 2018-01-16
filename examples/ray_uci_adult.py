# -*- coding:utf-8 -*-
"""
UCI_ADULT Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.datasets import uci_adult
from forestlayer.estimators import get_estimator_kfold
from forestlayer.estimators.estimator_configs import RandomForestConfig, ExtraRandomForestConfig
import ray
import time

ray.init()

(x_train, y_train, x_test, y_test) = uci_adult.load_data()

start_time = time.time()


@ray.remote
def fit(est, X, y):
    return est.fit(X, y)


rf_args = RandomForestConfig().get_est_args()
crf_args = ExtraRandomForestConfig().get_est_args()

est_configs = [
    get_estimator_kfold(name='1', est_type='RF', keep_in_mem=False, est_args=rf_args.copy()),
    get_estimator_kfold(name='2', est_type='CRF', keep_in_mem=False, est_args=crf_args.copy())
]

data = [fit.remote(est, x_train, y_train) for est in est_configs]

x_train_probas = ray.get(data)

for x_p in x_train_probas:
    print(x_p.shape)

end_time = time.time()
print('time cost: {}'.format(end_time - start_time))

# @ray.remote(num_cpus=2)
# def load_uci_data():
#     return uci_adult.load_data()
#
#
#
# data = load_uci_data.remote()
# (x_train, y_train, x_test, y_test) = ray.get(data)
#
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# print(x_train.shape[1], 'features')
