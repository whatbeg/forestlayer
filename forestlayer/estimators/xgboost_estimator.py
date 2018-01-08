# -*- coding:utf-8 -*-
"""
XGBoost Estimator, K-fold wrapper version.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from base_estimator import *
from ..utils.log_utils import get_logger
import xgboost as xgb

LOGGER = get_logger('estimators.xgboost')


class XGBoostClassifier(BaseEstimator):
    def __init__(self, name, est_args):
        super(XGBoostClassifier, self).__init__(est_class=XGBoostClassifier, name=name, est_args=est_args)
        self.cache_suffix = '.pkl'
        self.est = None
        self.n_class = est_args.get('num_class')

    def fit(self, X, y, cache_dir=None):
        cache_path = self._cache_path(cache_dir=cache_dir)
        # cache it
        if is_path_exists(cache_path):
            LOGGER.info('Found estimator from {}, skip fit'.format(cache_path))
            return
        if not isinstance(X, xgb.DMatrix):
            X = xgb.DMatrix(X, label=y)
        watch_list = [(X, 'train'), ]
        est = xgb.train(self.est_args, dtrain=X, evals=watch_list)
        if cache_path is not None:
            LOGGER.info("Save estimator to {} ...".format(cache_path))
            check_dir(cache_path)
            self._save_model_to_disk(self.est, cache_path)
            # keep out memory
            self.est = None
        else:
            # keep in memory
            self.est = est

    def _fit(self, est, X, y):
        pass

    def _predict_proba(self, est, X):
        assert self.n_class is not None, 'num_class is None!'
        if type(X) == list or not isinstance(X, xgb.DMatrix):
            xg_test = xgb.DMatrix(X)
        else:
            xg_test = X
        y_proba = np.array(est.predict(xg_test)).reshape((-1, self.n_class))
        return y_proba

    def _load_model_from_disk(self, cache_path):
        pass

    def _save_model_to_disk(self, est, cache_path):
        pass
