# -*- coding:utf-8 -*-
"""
Argument class definition.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0


class EstimatorArgument(object):
    def __init__(self):
        self.est_args = {}

    def get_est_args(self):
        return self.est_args


class MultiClassXGBoost(EstimatorArgument):
    def __init__(self, n_folds=3, nthread=-1, num_class=None, silent=True, objective="multi:softprob",
                 eval_metric="merror", eta=0.03, subsample=0.9, colsample_bytree=0.85,
                 colsample_bylevel=0.9, max_depth=6, verbose_eval=10):
        super(MultiClassXGBoost, self).__init__()
        assert num_class is not None, 'You must set number of classes!'
        self.est_args = {
            'est_type': 'XGB',
            'n_folds': n_folds,
            'nthread': nthread,
            'num_class': num_class,
            'silent': silent,
            'objective': objective,
            'eval_metric': eval_metric,
            'eta': eta,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'max_depth': max_depth,
            'verbose_eval': verbose_eval
        }


class BinClassXGBoost(EstimatorArgument):
    def __init__(self, n_folds=3, nthread=-1, num_class=2, silent=True, objective="binary:logistic",
                 eval_metric="merror", eta=0.03, subsample=0.9, colsample_bytree=0.85,
                 colsample_bylevel=0.9, max_depth=6, verbose_eval=10):
        super(BinClassXGBoost, self).__init__()
        assert num_class is not None, 'You must set number of classes!'
        self.est_args = {
            'est_type': 'XGB',
            'n_folds': n_folds,
            'nthread': nthread,
            'num_class': num_class,
            'silent': silent,
            'objective': objective,
            'eval_metric': eval_metric,
            'eta': eta,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'max_depth': max_depth,
            'verbose_eval': verbose_eval
        }



