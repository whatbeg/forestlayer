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
from ..utils.metrics import Accuracy


def est_class_from_type(est_type):
    if est_type == 'CRF':
        return CompletelyRFClassifier
    if est_type == 'RF':
        return RFClassifier
    if est_type == 'GBDT':
        return GBDTClassifier
    if est_type == 'XGB':
        return XGBoostClassifier
    raise ValueError('Unknown Estimator')


def get_estimator(name, est_type, est_args):
    est_class = est_class_from_type(est_type)
    return est_class(name, est_args)


def get_estimator_kfold(name, n_folds, est_type, eval_metrics=None, seed=None, keep_in_mem=True, est_args=None):
    est_class = est_class_from_type(est_type)
    if eval_metrics is None:
        eval_metrics = [Accuracy('accuracy')]
    return KFoldWrapper(name,
                        n_folds,
                        est_class,
                        seed=seed,
                        eval_metrics=eval_metrics,
                        keep_in_mem=keep_in_mem,
                        est_args=est_args)
