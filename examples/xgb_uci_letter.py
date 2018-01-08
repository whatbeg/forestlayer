# -*- coding:utf-8 -*-
"""
UCI_LETTER Example using XGBoost.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.datasets import uci_letter
from forestlayer.layers import Graph, AutoGrowingCascadeLayer
from forestlayer.utils.storage_utils import get_data_save_base
from forestlayer.estimators.arguments import MultiClassXGBoost
import time
import os.path as osp

start_time = time.time()

(X_train, y_train, X_test, y_test) = uci_letter.load_data()


est_configs = [
    MultiClassXGBoost(num_class=26, verbose_eval=False),
    MultiClassXGBoost(num_class=26, verbose_eval=False)
]

agc = AutoGrowingCascadeLayer(est_configs=est_configs,
                              early_stopping_rounds=4,
                              stop_by_test=True,
                              n_classes=26,
                              data_save_dir=osp.join(get_data_save_base(), 'uci_adult', 'auto_cascade'),
                              keep_in_mem=False)

model = Graph()
model.add(agc)
model.fit_transform(X_train, y_train, X_test, y_test)

# import xgboost as xgb
#
# est_args = {
#     'nthread': -1,
#     'num_class': 26,
#     "silent": True,
#     "objective": "multi:softmax",
#     "eval_metric": "merror",
#     "eta": 0.03,
#     "subsample": 0.9,
#     "colsample_bytree": 0.85,
#     "colsample_bylevel": 0.9,
#     "max_depth": 10
# }
#
# print(X_train.shape[0], y_train.shape[0])
#
# xg_train = xgb.DMatrix(X_train, label=y_train.reshape(-1))
# watch_list = [(xg_train, 'train')]
#
# est = xgb.train(est_args, dtrain=xg_train, num_boost_round=160, early_stopping_rounds=30,
#                 evals=watch_list, verbose_eval=10)

end_time = time.time()

print("Time cost: {}".format(end_time-start_time))
