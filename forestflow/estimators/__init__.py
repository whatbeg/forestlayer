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
from ..utils.metrics import accuracy_pb


def est_class_from_type(est_type):
    if est_type == 'CompletelyRandomForestClassifier':
        return CompletelyRFClassifier
    if est_type == 'RandomForestClassifier':
        return RFClassifier
    if est_type == 'GBDTClassifier':
        return GBDTClassifier
    raise ValueError('Unknown Estimator')


def get_estimator(name, est_type, est_args):
    est_class = est_class_from_type(est_type)
    return est_class(name, est_args)


def get_estimator_kfold(name, n_folds, est_type, est_args, seed=None):
    if est_type == "XGBoostClassifier":
        return XGBoostClassifier(name, n_folds, seed=seed, **est_args)
    est_class = est_class_from_type(est_type)
    return KFoldWrapper(name, n_folds, est_class, seed=seed, eval_metrics=[('accuracy', accuracy_pb)], **est_args)
