# -*- coding:utf-8 -*-
"""
Base Estimators Wrapper Definition.
"""

import numpy as np
from ..utils.storage_utils import *
from ..utils.log_utils import get_logger

LOGGER = get_logger('estimators.base_estimator')


class BaseEstimator(object):
    def __init__(self, est_class=None, name=None, est_args=None):
        self.name = name
        self.est_class = est_class
        self.est_args = est_args if est_args is not None else {}
        self.cache_suffix = '.pkl'
        self.est = None

    def _init_estimators(self):
        return self.est_class(**self.est_args)

    def fit(self, X, y, cache_dir=None):
        LOGGER.info('X_train.shape={}, y_train.shape={}'.format(X.shape, y.shape))
        cache_path = self._cache_path(cache_dir=cache_dir)
        # cache it
        if is_path_exists(cache_path):
            LOGGER.info('Found estimator from {}, skip fit'.format(cache_path))
            return
        est = self._init_estimators()
        self._fit(est, X, y)
        if cache_path is not None:
            LOGGER.info("Save estimator to {} ...".format(cache_path))
            check_dir(cache_path)
            self._save_model_to_disk(self.est, cache_path)
            # keep out memory
            self.est = None
        else:
            # keep in memory
            self.est = est

    def predict_proba(self, X, cache_dir=None, batch_size=None):
        LOGGER.debug("X.shape={}".format(X.shape))
        cache_path = self._cache_path(cache_dir)
        # cache
        if cache_path is not None:
            LOGGER.info("Load estimator from {} ...".format(cache_path))
            est = self._load_model_from_disk(cache_path)
            LOGGER.info("done ...")
        else:
            est = self.est
        batch_size = batch_size or self._default_predict_batch_size(est, X)
        if batch_size > 0:
            y_proba = self._batch_predict_proba(est, X, batch_size)
        else:
            y_proba = self._predict_proba(est, X)
        LOGGER.debug("y_proba.shape={}".format(y_proba.shape))
        return y_proba

    def _batch_predict_proba(self, est, X, batch_size):
        LOGGER.debug("X.shape={}, batch_size={}".format(X.shape, batch_size))
        verbose_backup = 0
        # clear verbose
        if hasattr(est, "verbose"):
            verbose_backup = est.verbose
            est.verbose = 0
        n_datas = X.shape[0]
        y_pred_proba = None
        for j in range(0, n_datas, batch_size):
            LOGGER.info("[batch_predict_proba][batch_size={}] ({}/{})".format(batch_size, j, n_datas))
            y_cur = self._predict_proba(est, X[j:j+batch_size])
            if j == 0:
                n_classes = y_cur.shape[1]
                y_pred_proba = np.empty((n_datas, n_classes), dtype=np.float32)
            y_pred_proba[j:j+batch_size, :] = y_cur
        # restore verbose
        if hasattr(est, "verbose"):
            est.verbose = verbose_backup
        return y_pred_proba

    def _cache_path(self, cache_dir):
        if cache_dir is None:
            return None
        return osp.join(cache_dir, name2path(self.name) + self.cache_suffix)

    def _load_model_from_disk(self, cache_path):
        raise NotImplementedError()

    def _save_model_to_disk(self, est, cache_path):
        raise NotImplementedError()

    def _default_predict_batch_size(self, est, X):
        """
        You can re-implement this function when inherient this class

        Return
        ------
        predict_batch_size (int): default=0
            if = 0,  predict_proba without batches
            if > 0, then predict_proba without baches
            sklearn predict_proba is not so inefficient, has to do this
        """
        return 0

    def _fit(self, est, X, y):
        est.fit(X, y)

    def _predict_proba(self, est, X):
        return est.predict_proba(X)

    def copy(self):
        return BaseEstimator(est_class=self.est_class, **self.est_args)


