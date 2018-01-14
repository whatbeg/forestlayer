# -*- coding:utf-8 -*-
"""
K-fold wrapper definition.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import os.path as osp
import numpy as np
import ray
from sklearn.model_selection import KFold, StratifiedKFold
from ..utils.metrics import Accuracy
from ..utils.log_utils import get_logger
from ..utils.storage_utils import name2path


class KFoldWrapper(object):
    def __init__(self, name, n_folds, est_class, seed=None,
                 eval_metrics=None, cache_dir=None, keep_in_mem=None, est_args=None):
        """
        Initialize a KFoldWrapper.

        :param name:
        :param n_folds:
        :param est_class:
        :param seed:
        :param eval_metrics:
        :param cache_dir:
        :param keep_in_mem:
        :param est_args:
        """
        self.LOGGER = get_logger('estimators.kfold_wrapper')
        self.name = name
        self.n_folds = n_folds
        self.est_class = est_class
        self.est_args = est_args if est_args is not None else {}
        self.seed = seed
        self.eval_metrics = eval_metrics if eval_metrics is not None else []
        if cache_dir is not None:
            self.cache_dir = osp.join(cache_dir, name2path(self.name))
        else:
            self.cache_dir = None
        self.keep_in_mem = keep_in_mem
        self.fit_estimators = [None for _ in range(n_folds)]
        self.n_dims = None

    def _init_estimator(self, k):
        """
        Initialize k-th estimator in K-fold CV.

        :param k: the order number of k-th estimator.
        :return: initialed estimator
        """
        est_args = self.est_args.copy()
        est_name = '{}/{}'.format(self.name, k)
        est_args['random_state'] = self.seed
        return self.est_class(est_name, est_args)

    def fit_transform(self, X, y, y_stratify=None, test_sets=None):
        """
        Fit and transform.

        :param X: (ndarray) n x k or n1 x n2 x k
                            to support windows_layer, X could have dim >2
        :param y: (ndarray) y (ndarray):
                            n or n1 x n2
        :param y_stratify: (list) used for StratifiedKFold or None means no stratify
        :param test_sets: (list) optional.
                   A list of (prefix, X_test, y_test) pairs.
                   predict_proba for X_test will be returned
                   use with keep_model_in_mem=False to save mem usage
                   y_test could be None, otherwise use eval_metrics for debugging
        :return:
        """
        if self.keep_in_mem is None:
            self.keep_in_mem = False
        assert 2 <= len(X.shape) <= 3, "X.shape should be n x k or n x n2 x k"
        assert len(X.shape) == len(y.shape) + 1
        if y_stratify is not None:
            assert X.shape[0] == len(y_stratify)
        test_sets = test_sets if test_sets is not None else []
        # K-Fold split
        n_stratify = X.shape[0]
        if self.n_folds == 1:
            cv = [(range(len(X)), range(len(X)))]
        else:
            if y_stratify is None:
                skf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
                cv = [(t, v) for (t, v) in skf.split(range(n_stratify))]
            else:
                skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
                cv = [(t, v) for (t, v) in skf.split(range(n_stratify), y_stratify)]
        # K-fold fit
        y_proba_train = None
        y_probas_test = []
        self.n_dims = X.shape[-1]
        inverse = False
        for k in range(self.n_folds):
            est = self._init_estimator(k)
            if not inverse:
                train_idx, val_idx = cv[k]
            else:
                val_idx, train_idx = cv[k]
            # fit on k-fold train
            est.fit(X[train_idx].reshape((-1, self.n_dims)), y[train_idx].reshape(-1), cache_dir=self.cache_dir)

            # predict on k-fold validation
            y_proba = est.predict_proba(X[val_idx].reshape((-1, self.n_dims)), cache_dir=self.cache_dir)
            if not est.is_classification:
                y_proba = y_proba[:, np.newaxis]  # add one dimension
            if len(X.shape) == 3:
                y_proba = y_proba.reshape((len(val_idx), -1, y_proba.shape[-1]))
            self.log_eval_metrics(self.name, y[val_idx], y_proba, "train_{}".format(k))

            # merging result
            if k == 0:
                if len(X.shape) == 2:
                    y_proba_cv = np.zeros((n_stratify, y_proba.shape[1]), dtype=np.float32)
                else:
                    y_proba_cv = np.zeros((n_stratify, y_proba.shape[1], y_proba.shape[2]), dtype=np.float32)
                y_proba_train = y_proba_cv
            y_proba_train[val_idx, :] += y_proba

            if self.keep_in_mem:
                self.fit_estimators[k] = est

            # test
            for vi, (prefix, X_test, y_test) in enumerate(test_sets):
                y_proba = est.predict_proba(X_test.reshape((-1, self.n_dims)), cache_dir=self.cache_dir)
                if not est.is_classification:
                    y_proba = y_proba[:, np.newaxis]
                if len(X.shape) == 3:
                    y_proba = y_proba.reshape((X_test.shape[0], X_test.shape[1], y_proba.shape[-1]))
                if k == 0:
                    y_probas_test.append(y_proba)
                else:
                    y_probas_test[vi] += y_proba
        if inverse and self.n_folds > 1:
            y_proba_train /= (self.n_folds - 1)
        for y_proba in y_probas_test:
            y_proba /= self.n_folds

        # log
        self.log_eval_metrics(self.name, y, y_proba_train, "train")
        # y_test can be None
        for vi, (test_name, X_test, y_test) in enumerate(test_sets):
            if y_test is not None:
                self.log_eval_metrics(self.name, y_test, y_probas_test[vi], test_name)
        return y_proba_train, y_probas_test

    def transform(self, x_tests):
        """
        Transform data.

        :param x_tests:
        :return:
        """
        # TODO: using model loaded from disk
        if x_tests is None or x_tests == []:
            return []
        if isinstance(x_tests, (list, tuple)):
            self.LOGGER.warn('transform(x_tests) only support single ndarray instead of list of ndarrays')
            x_tests = x_tests[0]
        proba_result = None
        for k, est in enumerate(self.fit_estimators):
            y_proba = est.predict_proba(x_tests.reshape((-1, self.n_dims)), cache_dir=self.cache_dir)
            if not est.is_classification:
                y_proba = y_proba[:, np.newaxis]  # add one dimension
            if len(x_tests.shape) == 3:
                y_proba = y_proba.reshape((x_tests.shape[0], x_tests.shape[1], y_proba.shape[-1]))
            if k == 0:
                proba_result = y_proba
            else:
                proba_result += y_proba
            proba_result /= self.n_folds
        return proba_result

    def log_eval_metrics(self, est_name, y_true, y_proba, y_name):
        """
        y_true (ndarray): n or n1 x n2
        y_proba (ndarray): n x n_classes or n1 x n2 x n_classes
        """
        if self.eval_metrics is None:
            return
        for metric in self.eval_metrics:
            acc = metric.calc_proba(y_true, y_proba)
            self.LOGGER.info("{}({} - {}) = {:.4f}{}".format(
                metric.__class__.__name__, est_name, y_name, acc, '%' if isinstance(metric, Accuracy) else ''))

    def _predict_proba(self, est, X):
        """
        Predict probability inner method.

        :param est:
        :param X:
        :return:
        """
        return est.predict_proba(X)

    def copy(self):
        """
        copy.

        :return:
        """
        return KFoldWrapper(name=self.name,
                            n_folds=self.n_folds,
                            est_class=self.est_class,
                            seed=self.seed,
                            eval_metrics=self.eval_metrics,
                            cache_dir=self.cache_dir,
                            keep_in_mem=self.keep_in_mem,
                            est_args=self.est_args)


@ray.remote
class DistributedKFoldWrapper(object):
    def __init__(self, name, n_folds, est_class, seed=None,
                 eval_metrics=None, cache_dir=None, keep_in_mem=None, est_args=None):
        """
        Initialize a KFoldWrapper.

        :param name:
        :param n_folds:
        :param est_class:
        :param seed:
        :param eval_metrics:
        :param cache_dir:
        :param keep_in_mem:
        :param est_args:
        """
        # log_info is used to store logging string, will be return to master node after fit/fit_transform
        self.log_info = []
        self.log_warn = []
        self.name = name
        self.n_folds = n_folds
        self.est_class = est_class
        self.est_args = est_args if est_args is not None else {}
        self.seed = seed
        self.eval_metrics = eval_metrics if eval_metrics is not None else []
        if cache_dir is not None:
            self.cache_dir = osp.join(cache_dir, name2path(self.name))
        else:
            self.cache_dir = None
        self.keep_in_mem = keep_in_mem
        self.fit_estimators = [None for _ in range(n_folds)]
        self.n_dims = None

    def get_fit_estimators(self):
        return self.fit_estimators

    def _init_estimator(self, k):
        """
        Initialize k-th estimator in K-fold CV.

        :param k: the order number of k-th estimator.
        :return: initialed estimator
        """
        est_args = self.est_args.copy()
        est_name = '{}/{}'.format(self.name, k)
        est_args['random_state'] = self.seed
        return self.est_class(est_name, est_args)

    def fit_transform(self, X, y, y_stratify=None, test_sets=None):
        """
        Fit and transform.

        :param X: (ndarray) n x k or n1 x n2 x k
                            to support windows_layer, X could have dim >2
        :param y: (ndarray) y (ndarray):
                            n or n1 x n2
        :param y_stratify: (list) used for StratifiedKFold or None means no stratify
        :param test_sets: (list) optional.
                   A list of (prefix, X_test, y_test) pairs.
                   predict_proba for X_test will be returned
                   use with keep_model_in_mem=False to save mem usage
                   y_test could be None, otherwise use eval_metrics for debugging
        :return:
        """
        if self.keep_in_mem is None:
            self.keep_in_mem = False
        assert 2 <= len(X.shape) <= 3, "X.shape should be n x k or n x n2 x k"
        assert len(X.shape) == len(y.shape) + 1
        if y_stratify is not None:
            assert X.shape[0] == len(y_stratify)
        test_sets = test_sets if test_sets is not None else []
        # K-Fold split
        n_stratify = X.shape[0]
        if self.n_folds == 1:
            cv = [(range(len(X)), range(len(X)))]
        else:
            if y_stratify is None:
                skf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
                cv = [(t, v) for (t, v) in skf.split(range(n_stratify))]
            else:
                skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
                cv = [(t, v) for (t, v) in skf.split(range(n_stratify), y_stratify)]
        # K-fold fit
        y_proba_train = None
        y_probas_test = []
        self.n_dims = X.shape[-1]
        inverse = False
        for k in range(self.n_folds):
            est = self._init_estimator(k)
            if not inverse:
                train_idx, val_idx = cv[k]
            else:
                val_idx, train_idx = cv[k]
            # fit on k-fold train
            est.fit(X[train_idx].reshape((-1, self.n_dims)), y[train_idx].reshape(-1), cache_dir=self.cache_dir)

            # predict on k-fold validation
            y_proba = est.predict_proba(X[val_idx].reshape((-1, self.n_dims)), cache_dir=self.cache_dir)
            if not est.is_classification:
                y_proba = y_proba[:, np.newaxis]  # add one dimension
            if len(X.shape) == 3:
                y_proba = y_proba.reshape((len(val_idx), -1, y_proba.shape[-1]))
            self.log_eval_metrics(self.name, y[val_idx], y_proba, "train_{}".format(k))

            # merging result
            if k == 0:
                if len(X.shape) == 2:
                    y_proba_cv = np.zeros((n_stratify, y_proba.shape[1]), dtype=np.float32)
                else:
                    y_proba_cv = np.zeros((n_stratify, y_proba.shape[1], y_proba.shape[2]), dtype=np.float32)
                y_proba_train = y_proba_cv
            y_proba_train[val_idx, :] += y_proba

            if self.keep_in_mem:
                self.fit_estimators[k] = est

            # test
            for vi, (prefix, X_test, y_test) in enumerate(test_sets):
                y_proba = est.predict_proba(X_test.reshape((-1, self.n_dims)), cache_dir=self.cache_dir)
                if not est.is_classification:
                    y_proba = y_proba[:, np.newaxis]
                if len(X.shape) == 3:
                    y_proba = y_proba.reshape((X_test.shape[0], X_test.shape[1], y_proba.shape[-1]))
                if k == 0:
                    y_probas_test.append(y_proba)
                else:
                    y_probas_test[vi] += y_proba
        if inverse and self.n_folds > 1:
            y_proba_train /= (self.n_folds - 1)
        for y_proba in y_probas_test:
            y_proba /= self.n_folds

        # log
        self.log_eval_metrics(self.name, y, y_proba_train, "train")
        # y_test can be None
        for vi, (test_name, X_test, y_test) in enumerate(test_sets):
            if y_test is not None:
                self.log_eval_metrics(self.name, y_test, y_probas_test[vi], test_name)
        return y_proba_train, y_probas_test, self.log_info

    def transform(self, x_tests):
        """
        Transform data.

        :param x_tests:
        :return:
        """
        # TODO: using model loaded from disk
        if x_tests is None or x_tests == []:
            return []
        if isinstance(x_tests, (list, tuple)):
            self.log_warn.append('transform(x_tests) only support single ndarray instead of list of ndarrays')
            x_tests = x_tests[0]
        proba_result = None
        for k, est in enumerate(self.fit_estimators):
            y_proba = est.predict_proba(x_tests.reshape((-1, self.n_dims)), cache_dir=self.cache_dir)
            if not est.is_classification:
                y_proba = y_proba[:, np.newaxis]  # add one dimension
            if len(x_tests.shape) == 3:
                y_proba = y_proba.reshape((x_tests.shape[0], x_tests.shape[1], y_proba.shape[-1]))
            if k == 0:
                proba_result = y_proba
            else:
                proba_result += y_proba
            proba_result /= self.n_folds
        return proba_result

    def log_eval_metrics(self, est_name, y_true, y_proba, y_name):
        """
        Logging evaluation metrics.

        :param est_name: estimator name.
        :param y_true: (ndarray) n or n1 x n2
        :param y_proba: (ndarray) n x n_classes or n1 x n2 x n_classes
        :param y_name: 'train_{no.}' or 'train' or 'test', identify a name for this info.
        :return:
        """
        if self.eval_metrics is None:
            return
        for metric in self.eval_metrics:
            acc = metric.calc_proba(y_true, y_proba)
            self.log_info.append("{}({} - {}) = {:.4f}{}".format(
                metric.__class__.__name__, est_name, y_name, acc, '%' if isinstance(metric, Accuracy) else ''))

    def _predict_proba(self, est, X):
        """
        Predict probability inner method.

        :param est:
        :param X:
        :return:
        """
        return est.predict_proba(X)

    def copy(self):
        """
        copy.

        :return:
        """
        return DistributedKFoldWrapper.remote(name=self.name,
                                              n_folds=self.n_folds,
                                              est_class=self.est_class,
                                              seed=self.seed,
                                              eval_metrics=self.eval_metrics,
                                              cache_dir=self.cache_dir,
                                              keep_in_mem=self.keep_in_mem,
                                              est_args=self.est_args)










