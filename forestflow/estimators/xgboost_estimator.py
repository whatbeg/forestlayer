# -*- coding:utf-8 -*-
"""
XGBoost Estimator, K-fold wrapper version.
"""

import os.path as osp
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from ..utils.log_utils import get_logger
from ..utils.storage_utils import name2path

LOGGER = get_logger('estimators.xgboost_estimator')


class XGBoostClassifier(object):
    def __init__(self, name, n_folds, seed, **est_args):
        self.name = name
        self.n_folds = n_folds
        self.seed = seed
        self.est_args = dict(est_args)
        self.fit_estimators = [None for _ in range(n_folds)]

    # TODO: Complete XGBoostClassifier
    def fit_transform(self, X, y, y_stratify, cache_dir=None,
                      test_sets=None, keep_model_in_mem=True):
        pass

    def _predict_proba(self, est, X):
        return est.predict_proba(X)
