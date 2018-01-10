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
    """
    Get estimator class from task ('classification' or 'regression') and estimator type (a string)
    :param task: what task does this estimator to execute ('classification' or 'regression')
    :param est_type: estimator type (a string)
    :return: a concrete estimator instance
    """
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
    """
    Get an estimator
    :param name: estimator name
    :param task: what task does this estimator to execute ('classification' or 'regression')
    :param est_type: estimator type (a string)
    :param est_args: estimator arguments
    :return: a concrete estimator instance
    """
    est_class = est_class_from_type(task, est_type)
    return est_class(name, est_args)


def get_estimator_kfold(name, n_folds=3, task='classification', est_type='RF', eval_metrics=None, seed=None,
                        cache_dir=None, keep_in_mem=True, est_args=None):
    """
    A factory method to get a k-fold estimator
    :param name: estimator name
    :param n_folds: how many folds to execute in cross validation
    :param task: what task does this estimator to execute ('classification' or 'regression')
    :param est_type: estimator type (a string)
    :param eval_metrics: evaluation metrics. [Default: Accuracy (classification), MSE (regression)]
    :param seed: random seed
    :param cache_dir: data cache dir to cache intermediate data
    :param keep_in_mem: whether keep the model in memory
    :param est_args: estimator arguments
    :return: a KFoldWrapper instance of concrete estimator
    """
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
