# -*- coding:utf-8 -*-
"""
Initialize estimators.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from base_estimator import *
from sklearn_estimator import *
from xgboost_estimator import *
from kfold_wrapper import *
from arguments import *
from ..utils.metrics import Accuracy, MSE


def est_class_from_type(task, est_type):
    if task == 'classification':
        if est_type == 'CRF':
            return CompletelyRFClassifier
        if est_type == 'RF':
            return RFClassifier
        if est_type == 'GBDT':
            return GBDTClassifier
        if est_type == 'XGB':
            return XGBoostClassifier
        raise ValueError('Unknown Estimator: {}'.format(est_type))
    elif task == 'regression':
        if est_type == 'CRF':
            return CompletelyRFRegressor
        if est_type == 'RF':
            return RFRegressor
        if est_type == 'GBDT':
            return GBDTRegressor
        if est_type == 'XGB':
            return XGBoostRegressor
        raise ValueError('Unknown Estimator: {}'.format(est_type))
    else:
        raise ValueError('Unknown task: {}'.format(task))


def get_estimator(name, task, est_type, est_args):
    est_class = est_class_from_type(task, est_type)
    return est_class(name, est_args)


def get_estimator_kfold(name, n_folds=3, task='classification', est_type='RF', eval_metrics=None, seed=None,
                        cache_dir=None, keep_in_mem=True, est_args=None):
    est_class = est_class_from_type(task, est_type)
    if eval_metrics is None:
        if task == 'classification':
            eval_metrics = [Accuracy('accuracy')]
        else:
            eval_metrics = [MSE('MSE')]
    return KFoldWrapper(name,
                        n_folds,
                        est_class,
                        seed=seed,
                        eval_metrics=eval_metrics,
                        cache_dir=cache_dir,
                        keep_in_mem=keep_in_mem,
                        est_args=est_args)
