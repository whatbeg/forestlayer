# -*- coding:utf-8 -*-
"""
Argument class definition.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from ..utils.log_utils import get_logger


class EstimatorArgument(object):
    """
    Estimator Argument is a class describes an estimator and its specific arguments, which is used to create concrete
    estimators in training.
    Every time we create a multi-grain scan layer or a cascade layer, we don't put concrete estimator instances into
    initialization parameters, but put the EstimatorArgument that describes the estimator.
    """
    def __init__(self):
        self.LOGGER = get_logger('estimator.argument')
        self.est_args = {}

    def get_est_args(self):
        """
        Get estimator argument.

        :return: estimator argument
        """
        return self.est_args.copy()


class MultiClassXGBoost(EstimatorArgument):
    """
    Multi-class XGBoost Classifier Argument.
    """
    def __init__(self, n_folds=3, nthread=-1, booster='gbtree', scale_pos_weight=1, num_class=None, silent=True,
                 objective="multi:softprob", eval_metric="merror", eta=0.03, subsample=0.9, num_boost_round=60,
                 early_stopping_rounds=20, colsample_bytree=0.85, colsample_bylevel=0.9, max_depth=6,
                 verbose_eval=10, learning_rates=None):
        """
        Multi-class XGBoost Classifier Argument describes arguments of multi-class xgboost classifier.
        Parameters can refer to xgboost document: http://xgboost.readthedocs.io/en/latest/python/python_api.html

        :param n_folds: how many folds to execute in cross validation
        :param nthread: number of threads to execute
        :param booster: booster, default is 'gbtree'
        :param scale_pos_weight: used to handle class imbalance
        :param num_class: number of classes to classify
        :param silent:
        :param objective: objective, default is 'multi:softprob'
        :param eval_metric: evaluation metrics, default is 'merror'
        :param eta:
        :param subsample:
        :param num_boost_round:
        :param early_stopping_rounds:
        :param colsample_bytree:
        :param colsample_bylevel:
        :param max_depth:
        :param verbose_eval:
        :param learning_rates:
        """
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
    """
    Binary Class XGBoost Classifier.
    """
    def __init__(self, n_folds=3, nthread=-1, booster='gbtree', scale_pos_weight=1, num_class=2, silent=True,
                 objective="binary:logistic", eval_metric="auc", eta=0.03, subsample=0.9, num_boost_round=160,
                 early_stopping_rounds=30, colsample_bytree=0.85, colsample_bylevel=0.9, max_depth=6,
                 verbose_eval=20, learning_rates=None):
        """
        Binary-class XGBoost Classifier Argument describes arguments of multi-class xgboost classifier.
        Parameter can refer to xgboost document: http://xgboost.readthedocs.io/en/latest/python/python_api.html

        :param n_folds: how many folds to execute in cross validation
        :param nthread: number of threads to execute
        :param booster: booster, default is 'gbtree'
        :param scale_pos_weight: used to handle class imbalance
        :param num_class: number of classes to classify
        :param silent:
        :param objective: objective, default is 'multi:softprob'
        :param eval_metric: evaluation metrics, default is 'merror'
        :param eta:
        :param subsample:
        :param num_boost_round:
        :param early_stopping_rounds:
        :param colsample_bytree:
        :param colsample_bylevel:
        :param max_depth:
        :param verbose_eval:
        :param learning_rates:
        """
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
    """
    Random Forest Argument.
    """
    def __init__(self, n_folds=3, n_estimators=500, max_depth=100, n_jobs=-1, max_features='sqrt',
                 min_samples_leaf=1):
        """
        Random Forest Argument.

        :param n_folds:
        :param n_estimators:
        :param max_depth:
        :param n_jobs:
        :param max_features:
        :param min_samples_leaf:
        """
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
    """
    Completely Random Forest Argument.
    """
    def __init__(self, n_folds=3, n_estimators=500, max_depth=100, n_jobs=-1, max_features=1,
                 min_samples_leaf=1):
        """
        Completely Random Forest Argument.

        :param n_folds:
        :param n_estimators:
        :param max_depth:
        :param n_jobs:
        :param max_features:
        :param min_samples_leaf:
        """
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
    """
    Gradient Boosting Decision Tree Argument.
    """
    def __init__(self, n_folds=3, n_estimators=500, max_depth=6, n_jobs=-1, max_features='sqrt',
                 min_samples_leaf=1, subsample=1.0):
        """
        Gradient Boosting Decision Tree Argument.

        :param n_folds:
        :param n_estimators:
        :param max_depth:
        :param n_jobs:
        :param max_features:
        :param min_samples_leaf:
        :param subsample:
        """
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



