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
        return self.est_args.copy()


class MultiClassXGBoost(EstimatorArgument):
    def __init__(self, n_folds=3, nthread=-1, booster='gbtree', scale_pos_weight=1, num_class=None, silent=True,
                 objective="multi:softprob", eval_metric="merror", eta=0.03, subsample=0.9, num_boost_round=60,
                 early_stopping_rounds=20, colsample_bytree=0.85, colsample_bylevel=0.9, max_depth=6,
                 verbose_eval=10, learning_rates=None):
        super(MultiClassXGBoost, self).__init__()
        assert num_class is not None, 'You must set number of classes!'
        self.est_args = {
            'est_type': 'XGB',
            'n_folds': n_folds,
            'nthread': nthread,
            'booster': booster,
            'scale_pos_weight': scale_pos_weight,
            'num_class': num_class,
            'silent': silent,
            'objective': objective,
            'eval_metric': eval_metric,
            'eta': eta,
            'subsample': subsample,
            'num_boost_round': num_boost_round,
            'early_stopping_rounds': early_stopping_rounds,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'max_depth': max_depth,
            'verbose_eval': verbose_eval,
            'learning_rates': learning_rates
        }


class BinClassXGBoost(EstimatorArgument):
    def __init__(self, n_folds=3, nthread=-1, booster='gbtree', scale_pos_weight=1, num_class=2, silent=True,
                 objective="binary:logistic", eval_metric="auc", eta=0.03, subsample=0.9, num_boost_round=160,
                 early_stopping_rounds=30, colsample_bytree=0.85, colsample_bylevel=0.9, max_depth=6,
                 verbose_eval=20, learning_rates=None):
        super(BinClassXGBoost, self).__init__()
        assert num_class is not None, 'You must set number of classes!'
        self.est_args = {
            'est_type': 'XGB',
            'n_folds': n_folds,
            'nthread': nthread,
            'booster': booster,
            'scale_pos_weight': scale_pos_weight,
            'silent': silent,
            'objective': objective,
            'eval_metric': eval_metric,
            'eta': eta,
            'subsample': subsample,
            'num_boost_round': num_boost_round,
            'early_stopping_rounds': early_stopping_rounds,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'max_depth': max_depth,
            'verbose_eval': verbose_eval,
            'learning_rates': learning_rates
        }


class RandomForest(EstimatorArgument):
    def __init__(self, n_folds=3, n_estimators=500, max_depth=100, n_jobs=-1, max_features='sqrt',
                 min_samples_leaf=1):
        super(RandomForest, self).__init__()
        self.est_args = {
            'est_type': 'RF',
            'n_folds': n_folds,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'n_jobs': n_jobs,
            'max_features': max_features,
            'min_samples_leaf': min_samples_leaf
        }


class CompletelyRandomForest(EstimatorArgument):
    def __init__(self, n_folds=3, n_estimators=500, max_depth=100, n_jobs=-1, max_features=1,
                 min_samples_leaf=1):
        super(CompletelyRandomForest, self).__init__()
        self.est_args = {
            'est_type': 'CRF',
            'n_folds': n_folds,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'n_jobs': n_jobs,
            'max_features': max_features,
            'min_samples_leaf': min_samples_leaf
        }


class GBDT(EstimatorArgument):
    def __init__(self, n_folds=3, n_estimators=500, max_depth=6, n_jobs=-1, max_features='sqrt',
                 min_samples_leaf=1, subsample=1.0, ):
        super(GBDT, self).__init__()
        self.est_args = {
            'est_type': 'CRF',
            'n_folds': n_folds,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'n_jobs': n_jobs,
            'max_features': max_features,
            'min_samples_leaf': min_samples_leaf,
            'subsample': subsample
        }



