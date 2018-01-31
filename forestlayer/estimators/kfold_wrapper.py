# -*- coding:utf-8 -*-
"""
K-fold wrapper definition.
This page of code was partly borrowed from Ji. Feng.
"""

from __future__ import print_function
import os.path as osp
import numpy as np
import ray
import copy
try:
    import cPickle as pickle
except ImportError:
    import pickle
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost.sklearn import XGBClassifier, XGBRegressor
from .sklearn_estimator import *
from .estimator_configs import EstimatorConfig
from ..utils.log_utils import get_logger
from ..utils.storage_utils import name2path, getmbof
from ..utils.metrics import Accuracy, MSE
from collections import defaultdict
MAX_RAND_SEED = np.iinfo(np.int32).max


class KFoldWrapper(object):
    def __init__(self, name, n_folds, est_class, seed=None, dtype=np.float32,
                 eval_metrics=None, cache_dir=None, keep_in_mem=None, est_args=None, cv_seed=None):
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
        self.dtype = dtype
        self.cv_seed = cv_seed
        # assign a cv_seed
        if self.cv_seed is None:
            if isinstance(self.seed, np.random.RandomState):
                self.cv_seed = copy.deepcopy(self.seed)
            else:
                self.cv_seed = self.seed
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
        # TODO: consider if add a random_state, actually random_state of each estimator can be set in est_configs in
        # main program by users, so we need not to set random_state there.
        # More importantly, if some estimators have no random_state parameter, this assignment can throw problems.
        if isinstance(self.est_class, (XGBClassifier, XGBRegressor)):
            if est_args.get('seed', "NONE") == "NONE":
                est_args['seed'] = copy.deepcopy(self.seed)
        elif est_args.get('random_state', "NONE") == "NONE":
            est_args['random_state'] = copy.deepcopy(self.seed)
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
                skf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.cv_seed)
                cv = [(t, v) for (t, v) in skf.split(range(n_stratify))]
            else:
                skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.cv_seed)
                cv = [(t, v) for (t, v) in skf.split(range(n_stratify), y_stratify)]
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

            # predict on k-fold validation, this y_proba.dtype is float64
            y_proba = est.predict_proba(X[val_idx].reshape((-1, self.n_dims)),
                                        cache_dir=self.cache_dir)
            if not est.is_classification:
                y_proba = y_proba[:, np.newaxis]  # add one dimension
            if len(X.shape) == 3:
                y_proba = y_proba.reshape((len(val_idx), -1, y_proba.shape[-1]))
            self.log_eval_metrics(self.name, y[val_idx], y_proba, "train_{}".format(k))

            # merging result
            if k == 0:
                if len(X.shape) == 2:
                    y_proba_cv = np.zeros((n_stratify, y_proba.shape[1]), dtype=self.dtype)
                else:
                    y_proba_cv = np.zeros((n_stratify, y_proba.shape[1], y_proba.shape[2]), dtype=self.dtype)
                y_proba_train = y_proba_cv
            y_proba_train[val_idx, :] += y_proba

            if self.keep_in_mem:
                self.fit_estimators[k] = est

            # test
            for vi, (prefix, X_test, _) in enumerate(test_sets):
                # keep float32 data type, save half of memory and communication.
                y_proba = est.predict_proba(X_test.reshape((-1, self.n_dims)),
                                            cache_dir=self.cache_dir)
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

        # log train average
        self.log_eval_metrics(self.name, y, y_proba_train, "train_avg")
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
        NOTE: This copy does not confirm object consistency now, be careful to use it.
        TODO: check object consistency.

        :return:
        """
        return KFoldWrapper(name=self.name,
                            n_folds=self.n_folds,
                            est_class=self.est_class,
                            seed=self.seed,
                            eval_metrics=self.eval_metrics,
                            cache_dir=self.cache_dir,
                            keep_in_mem=self.keep_in_mem,
                            est_args=self.est_args,
                            cv_seed=self.cv_seed)


@ray.remote
class DistributedKFoldWrapper(object):
    def __init__(self, name, n_folds, est_class, seed=None, dtype=np.float32,
                 eval_metrics=None, cache_dir=None, keep_in_mem=None, est_args=None, cv_seed=None):
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
        # log_info is used to store logging strings, which will be return to master node after fit/fit_transform
        # every log info in logs consists of several parts: (level, base_string, data)
        # for example, ('INFO', 'Accuracy(win - 0 - estimator - 0 - 3folds - train_0) = {:.4f}%', 26.2546)
        self.logs = []
        self.name = name
        self.n_folds = n_folds
        self.est_class = est_class
        self.est_args = est_args if est_args is not None else {}
        self.seed = seed
        if isinstance(seed, basestring):
            self.seed = pickle.loads(seed)
        self.dtype = dtype
        self.cv_seed = cv_seed
        if self.cv_seed is None:
            self.cv_seed = self.seed
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
        if isinstance(self.est_class, (XGBClassifier, XGBRegressor)):
            if est_args.get('seed', "NONE") == "NONE":
                est_args['seed'] = copy.deepcopy(self.seed)
        # there is no 'random_state' in est_args
        elif est_args.get('random_state', "NONE") == "NONE":
            est_args['random_state'] = copy.deepcopy(self.seed)
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
                skf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.cv_seed)
                cv = [(t, v) for (t, v) in skf.split(range(n_stratify))]
            else:
                skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.cv_seed)
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
                    y_proba_cv = np.zeros((n_stratify, y_proba.shape[1]), dtype=self.dtype)
                else:
                    y_proba_cv = np.zeros((n_stratify, y_proba.shape[1], y_proba.shape[2]), dtype=self.dtype)
                y_proba_train = y_proba_cv
            y_proba_train[val_idx, :] += y_proba

            if self.keep_in_mem:
                self.fit_estimators[k] = est

            # test
            for vi, (prefix, X_test, y_test) in enumerate(test_sets):
                y_proba = est.predict_proba(X_test.reshape((-1, self.n_dims)),
                                            cache_dir=self.cache_dir)
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

        # log train average
        self.log_eval_metrics(self.name, y, y_proba_train, "train_avg")
        # y_test can be None
        for vi, (test_name, X_test, y_test) in enumerate(test_sets):
            if y_test is not None:
                self.log_eval_metrics(self.name, y_test, y_probas_test[vi], test_name)
        return y_proba_train, y_probas_test, self.logs

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
            self.logs.append(('WARN', 'transform(x_tests) only support single ndarray instead of list of ndarrays', 0))
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
            self.logs.append(('INFO', "{a}({b} - {c}) = {d}{e}".format(
                a=metric.__class__.__name__, b=est_name, c=y_name, d="{:.4f}",
                e='%' if isinstance(metric, Accuracy) else ''), acc))

    @staticmethod
    def _predict_proba(est, X):
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
        NOTE: This copy does not confirm object consistency now, be careful to use it.
        TODO: check object consistency.

        :return:
        """
        return DistributedKFoldWrapper.remote(name=self.name,
                                              n_folds=self.n_folds,
                                              est_class=self.est_class,
                                              seed=self.seed,
                                              eval_metrics=self.eval_metrics,
                                              cache_dir=self.cache_dir,
                                              keep_in_mem=self.keep_in_mem,
                                              est_args=self.est_args,
                                              cv_seed=self.cv_seed)


class SplittingKFoldWrapper(object):
    """
    Wrapper for splitting forests to smaller forests.
    TODO: support intelligent load-aware splitting method.
    """
    def __init__(self, dis_level=0, estimators=None, ei2wi=None, num_workers=None, seed=None, task='classification',
                 eval_metrics=None, keep_in_mem=False, cv_seed=None, dtype=np.float32):
        """
        Initialize SplittingKFoldWrapper.

        :param dis_level: distributed level, or parallelization level, 0 / 1 / 2
                           0 means lowest parallelization level, parallelization is len(self.est_configs).
                           1 means we will split the forests in some condition to making more full use of
                            cluster resources, so the parallelization may be larger than len(self.est_configs).
                           2 means that anyway we must split forests.
                           Now 2 is the HIGHEST_DISLEVEL
        :param estimators: base estimators.
        :param ei2wi: estimator to window it belongs to.
        :param num_workers: number of workers in the cluster.
        :param seed: random state.
        :param task: regression or classification.
        :param eval_metrics: evaluation metrics.
        :param keep_in_mem: boolean, if keep the model in mem, now we do not support model
                             saving of splitting_kfold_wrapper.
        :param cv_seed: cross validation random state.
        :param dtype: data type.
        """
        self.LOGGER = get_logger('estimators.splitting_kfold_wrapper')
        self.dis_level = dis_level
        self.estimators = estimators
        self.ei2wi = ei2wi
        self.num_workers = num_workers
        self.seed = seed
        self.task = task
        self.dtype = dtype
        self.eval_metrics = eval_metrics
        self.keep_in_mem = keep_in_mem
        self.cv_seed = cv_seed

    def determine_split(self, num_estimators):
        if self.dis_level == 0:
            return False
        if self.dis_level == 1:
            if self.num_workers >= num_estimators / 2:
                return True
        if self.dis_level == 2:
            return True
        return False

    def splitting(self, ests):
        """
        Splitting method.
        Judge if we should to split and how we split.

        :param ests:
        :return:
        """
        assert isinstance(ests, list), 'estimators should be a list, but {}'.format(type(ests))
        num_ests = len(ests)
        # should_split = False
        # if self.num_workers >= num_ests / 2:
        #     should_split = True
        # # if user do not want to split, and pass an argument split which is False, we don't split!
        # if self.split is False:
        #     should_split = False
        should_split = self.determine_split(num_ests)
        split_ests = []
        split_group = []
        self.LOGGER.info('dis_level = {}, num_workers = {}, num_estimators = {}, should_split? {}'.format(
            self.dis_level, self.num_workers, num_ests, should_split))
        if self.cv_seed is None:
            self.cv_seed = self.seed
        if should_split:
            i = 0
            new_ei2wi = dict()
            for ei, est in enumerate(ests):
                wi, wei = self.ei2wi[ei]
                num_trees = est.get_est_args().get('n_estimators', 500)
                est_name = 'win - {} - estimator - {} - {}folds'.format(wi, wei, est.get_est_args().get('n_folds', 3))
                if self.seed is not None:
                    common_seed = (self.seed + hash("[estimator] {}".format(est_name))) % 1000000007
                    seed = np.random.RandomState(common_seed)
                    seed2 = np.random.RandomState(common_seed)
                    seed2.randint(MAX_RAND_SEED, size=num_trees/2)
                else:
                    seed = np.random.mtrand._rand
                    seed2 = np.random.mtrand._rand
                    seed2.randint(MAX_RAND_SEED, size=num_trees/2)
                self.LOGGER.debug('{} trees split to {} + {}'.format(num_trees, num_trees / 2, num_trees - num_trees/2))
                args = est.get_est_args().copy()
                args['n_estimators'] = num_trees / 2
                sub_est1 = self._init_estimators(args, wi, wei, seed, self.cv_seed)
                args['n_estimators'] = num_trees - num_trees / 2
                sub_est2 = self._init_estimators(args, wi, wei, seed2, self.cv_seed)
                split_ests.append(sub_est1)
                split_ests.append(sub_est2)
                split_group.append([i, i + 1])
                new_ei2wi[i] = (wi, wei)
                new_ei2wi[i + 1] = (wi, wei)
                i += 2
            self.ei2wi = new_ei2wi
        else:
            for ei, est in enumerate(ests):
                wi, wei = self.ei2wi[ei]
                gen_est = self._init_estimators(est.get_est_args().copy(), wi, wei, self.seed, self.cv_seed)
                split_ests.append(gen_est)
            split_group = [[i, ] for i in range(len(ests))]
        return split_ests, split_group

    def _init_estimators(self, args, wi, ei, seed, cv_seed):
        """
        Initialize distributed kfold wrapper. dumps the seed if seed is a np.random.RandomState.

        :param args:
        :param wi:
        :param ei:
        :param seed:
        :param cv_seed:
        :return:
        """
        est_args = args.copy()
        est_name = 'win - {} - estimator - {} - {}folds'.format(wi, ei, est_args['n_folds'])
        n_folds = int(est_args['n_folds'])
        est_args.pop('n_folds')
        est_type = est_args['est_type']
        est_args.pop('est_type')
        # seed, if seed is not None and is integer, we add it with estimator name.
        # if seed is already a RandomState, just pickle it in order to pass to every worker.
        if seed is not None and not isinstance(seed, np.random.RandomState):
            seed = (seed + hash("[estimator] {}".format(est_name))) % 1000000007
        if isinstance(seed, np.random.RandomState):
            seed = pickle.dumps(seed, pickle.HIGHEST_PROTOCOL)
        # we must keep the cross validation seed same, but keep the seed not the same
        # so that no duplicate forest are generated, but exactly same cross validation datasets are generated.
        if cv_seed is not None and not isinstance(cv_seed, np.random.RandomState):
            cv_seed = (cv_seed + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            cv_seed = (0 + hash("[estimator] {}".format(est_name))) % 1000000007
        return get_dist_estimator_kfold(name=est_name,
                                        n_folds=n_folds,
                                        task=self.task,
                                        est_type=est_type,
                                        eval_metrics=self.eval_metrics,
                                        seed=seed,
                                        dtype=self.dtype,
                                        keep_in_mem=self.keep_in_mem,
                                        est_args=est_args,
                                        cv_seed=cv_seed)

    def fit(self, x_wins_train, y_win):
        """
        Fit. This method do splitting fit and collect/merge results of distributed forests.

        :param x_wins_train:
        :param y_win:
        :return:
        """
        split_ests, split_group = self.splitting(self.estimators)
        self.LOGGER.debug('split_group = {}'.format(split_group))
        self.LOGGER.debug('ei2wi = {}'.format(self.ei2wi))
        x_wins_train_obj_ids = [ray.put(x_wins_train[wi]) for wi in range(len(x_wins_train))]
        y_win_obj_ids = [ray.put(y_win[wi]) for wi in range(len(y_win))]
        y_stratify = [ray.put(y_win[wi][:, 0]) for wi in range(len(y_win))]
        # the base kfold_wrapper of SplittingKFoldWrapper must be DistributedKFoldWrapper,
        # so with the y_proba_train, y_proba_tests, there is a log info list will be return.
        # so, ests_output is like (y_proba_train, y_proba_tests, logs)
        ests_output = [est.fit_transform.remote(x_wins_train_obj_ids[self.ei2wi[ei][0]],
                                                y_win_obj_ids[self.ei2wi[ei][0]],
                       y_stratify[self.ei2wi[ei][0]]) for ei, est in enumerate(split_ests)]
        est_group = []
        for grp in split_group:
            if len(grp) == 2:
                # Tree reduce
                est_group.append(merge.remote(ests_output[grp[0]], ests_output[grp[1]]))
            else:
                est_group.append(ests_output[grp[0]])
        est_group_result = ray.get(est_group)
        return est_group_result

    def fit_transform(self, x_wins_train, y_win, test_sets):
        """
        Fit and transform. This method do splitting fit and collect/merge results of distributed forests.

        :param x_wins_train:
        :param y_win:
        :param test_sets:
        :return:
        """
        split_ests, split_group = self.splitting(self.estimators)
        self.LOGGER.debug('split_group = {}'.format(split_group))
        self.LOGGER.debug('new ei2wi = {}'.format(self.ei2wi))
        x_wins_train_obj_ids = [ray.put(x_wins_train[wi]) for wi in range(len(x_wins_train))]
        y_win_obj_ids = [ray.put(y_win[wi]) for wi in range(len(y_win))]
        y_stratify = [ray.put(y_win[wi][:, 0]) for wi in range(len(y_win))]
        test_sets_obj_ids = [ray.put(test_sets[wi]) for wi in range(len(test_sets))]
        # the base kfold_wrapper of SplittingKFoldWrapper must be DistributedKFoldWrapper,
        # so with the y_proba_train, y_proba_tests, there is a log info list will be return.
        # so, ests_output is like (y_proba_train, y_proba_tests, logs)
        ests_output = [est.fit_transform.remote(x_wins_train_obj_ids[self.ei2wi[ei][0]],
                                                y_win_obj_ids[self.ei2wi[ei][0]], y_stratify[self.ei2wi[ei][0]],
                                                test_sets=test_sets_obj_ids[self.ei2wi[ei][0]])
                       for ei, est in enumerate(split_ests)]
        est_group = []
        for grp in split_group:
            if len(grp) == 2:
                # Tree reduce
                est_group.append(merge.remote(ests_output[grp[0]], ests_output[grp[1]]))
            else:
                est_group.append(ests_output[grp[0]])
        est_group_result = ray.get(est_group)
        return est_group_result


class CascadeSplittingKFoldWrapper(object):
    """
    Wrapper for splitting forests to smaller forests.
    TODO: support intelligent load-aware splitting method.
    """
    def __init__(self, dis_level=0, estimators=None, num_workers=None, seed=None, task='classification',
                 eval_metrics=None, keep_in_mem=False, cv_seed=None, dtype=np.float32,
                 layer_id=None):
        """
        Initialize CascadeSplittingKFoldWrapper.

        :param dis_level: distributed level, or parallelization level, 0 / 1 / 2
                           0 means lowest parallelization level, parallelization is len(self.est_configs).
                           1 means we will split the forests in some condition to making more full use of
                            cluster resources, so the parallelization may be larger than len(self.est_configs).
                           2 means that anyway we must split forests.
                           Now 2 is the HIGHEST_DISLEVEL
        :param estimators: base estimators.
        :param num_workers: number of workers in the cluster.
        :param seed: random state.
        :param task: regression or classification.
        :param eval_metrics: evaluation metrics.
        :param keep_in_mem: boolean, if keep the model in mem, now we do not support model
                             saving of cascade_splitting_kfold_wrapper.
        :param cv_seed: cross validation random state.
        :param dtype: data type.
        """
        self.LOGGER = get_logger('estimators.cascade_splitting_kfold_wrapper')
        self.dis_level = dis_level
        self.estimators = estimators
        self.num_workers = num_workers
        self.seed = seed
        self.task = task
        self.dtype = dtype
        self.eval_metrics = eval_metrics
        self.keep_in_mem = keep_in_mem
        self.cv_seed = cv_seed
        # cascade
        self.layer_id = layer_id
        for ei, est in enumerate(self.estimators):
            # convert estimators from EstimatorConfig to dictionary.
            if isinstance(est, EstimatorConfig):
                self.estimators[ei] = est.get_est_args().copy()

    def determine_split(self, num_estimators):
        if self.dis_level == 0:
            return False
        if self.dis_level == 1:
            if self.num_workers >= num_estimators / 2:
                return True
        if self.dis_level == 2:
            return True
        return False

    def splitting(self, ests):
        """
        Splitting method.
        Judge if we should to split and how we split.

        :param ests:
        :return:
        """
        assert isinstance(ests, list), 'estimators should be a list, but {}'.format(type(ests))
        num_ests = len(ests)
        should_split = self.determine_split(num_ests)
        split_ests = []
        split_group = []
        self.LOGGER.info('dis_level = {}, num_workers = {}, num_estimators = {}, should_split? {}'.format(
            self.dis_level, self.num_workers, num_ests, should_split))
        if self.cv_seed is None:
            self.cv_seed = self.seed
        if should_split:
            i = 0
            for ei, est in enumerate(ests):
                num_trees = est.get('n_estimators', 500)
                est_name = 'layer - {} - estimator - {} - {}folds'.format(self.layer_id, ei,
                                                                          est.get('n_folds', 3))
                if self.seed is not None:
                    common_seed = (self.seed + hash("[estimator] {}".format(est_name))) % 1000000007
                    seed = np.random.RandomState(common_seed)
                    seed2 = np.random.RandomState(common_seed)
                    seed2.randint(MAX_RAND_SEED, size=num_trees/2)
                else:
                    seed = np.random.mtrand._rand
                    seed2 = np.random.mtrand._rand
                    seed2.randint(MAX_RAND_SEED, size=num_trees/2)
                self.LOGGER.debug('{} trees split to {} + {}'.format(num_trees, num_trees / 2, num_trees - num_trees/2))
                args = est.copy()
                args['n_estimators'] = num_trees / 2
                sub_est1 = self._init_estimators(args, self.layer_id, ei, seed, self.cv_seed)
                args['n_estimators'] = num_trees - num_trees / 2
                sub_est2 = self._init_estimators(args, self.layer_id, ei, seed2, self.cv_seed)
                split_ests.append(sub_est1)
                split_ests.append(sub_est2)
                split_group.append([i, i + 1])
                i += 2
        else:
            for ei, est in enumerate(ests):
                gen_est = self._init_estimators(est.copy(), self.layer_id, ei, self.seed, self.cv_seed)
                split_ests.append(gen_est)
            split_group = [[i, ] for i in range(len(ests))]
        return split_ests, split_group

    def _init_estimators(self, args, layer_id, ei, seed, cv_seed):
        """
        Initialize distributed kfold wrapper. dumps the seed if seed is a np.random.RandomState.

        :param args:
        :param layer_id:
        :param ei:
        :param seed:
        :param cv_seed:
        :return:
        """
        est_args = args.copy()
        est_name = 'layer - {} - estimator - {} - {}folds'.format(layer_id, ei, est_args['n_folds'])
        n_folds = int(est_args['n_folds'])
        est_args.pop('n_folds')
        est_type = est_args['est_type']
        est_args.pop('est_type')
        # seed, if seed is not None and is integer, we add it with estimator name.
        # if seed is already a RandomState, just pickle it in order to pass to every worker.
        if seed is not None and not isinstance(seed, np.random.RandomState):
            seed = (seed + hash("[estimator] {}".format(est_name))) % 1000000007
        if isinstance(seed, np.random.RandomState):
            seed = pickle.dumps(seed, pickle.HIGHEST_PROTOCOL)
        # we must keep the cross validation seed same, but keep the seed not the same
        # so that no duplicate forest are generated, but exactly same cross validation datasets are generated.
        if cv_seed is not None and not isinstance(cv_seed, np.random.RandomState):
            cv_seed = (cv_seed + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            cv_seed = (0 + hash("[estimator] {}".format(est_name))) % 1000000007
        return get_dist_estimator_kfold(name=est_name,
                                        n_folds=n_folds,
                                        task=self.task,
                                        est_type=est_type,
                                        eval_metrics=self.eval_metrics,
                                        seed=seed,
                                        dtype=self.dtype,
                                        keep_in_mem=self.keep_in_mem,
                                        est_args=est_args,
                                        cv_seed=cv_seed)

    def fit(self, x_train, y_train, y_stratify):
        split_ests, split_group = self.splitting(self.estimators)
        self.LOGGER.debug('split_group = {}'.format(split_group))
        x_train_obj_id = ray.put(x_train)
        y_train_obj_id = ray.put(y_train)
        y_stratify_obj_id = ray.put(y_stratify)
        # the base kfold_wrapper of SplittingKFoldWrapper must be DistributedKFoldWrapper,
        # so with the y_proba_train, y_proba_tests, there is a log info list will be return.
        # so, ests_output is like (y_proba_train, y_proba_tests, logs)
        ests_output = [est.fit_transform.remote(x_train_obj_id, y_train_obj_id, y_stratify_obj_id, test_sets=None)
                       for est in split_ests]
        est_group = []
        for grp in split_group:
            if len(grp) == 2:
                # Tree reduce
                est_group.append(merge.remote(ests_output[grp[0]], ests_output[grp[1]]))
            else:
                est_group.append(ests_output[grp[0]])
        est_group_result = ray.get(est_group)
        return est_group_result, split_ests, split_group

    def fit_transform(self, x_train, y_train, y_stratify, test_sets=None):
        split_ests, split_group = self.splitting(self.estimators)
        self.LOGGER.debug('split_group = {}'.format(split_group))
        x_train_obj_id = ray.put(x_train)
        y_train_obj_id = ray.put(y_train)
        y_stratify_obj_id = ray.put(y_stratify)
        test_sets_obj_id = ray.put(test_sets)
        # the base kfold_wrapper of SplittingKFoldWrapper must be DistributedKFoldWrapper,
        # so with the y_proba_train, y_proba_tests, there is a log info list will be return.
        # so, ests_output is like (y_proba_train, y_proba_tests, logs)
        ests_output = [est.fit_transform.remote(x_train_obj_id, y_train_obj_id, y_stratify_obj_id,
                                                test_sets=test_sets_obj_id)
                       for est in split_ests]
        est_group = []
        for grp in split_group:
            if len(grp) == 2:
                # Tree reduce
                est_group.append(merge.remote(ests_output[grp[0]], ests_output[grp[1]]))
            else:
                est_group.append(ests_output[grp[0]])
        est_group_result = ray.get(est_group)
        return est_group_result, split_ests, split_group


@ray.remote
def merge(tup_1, tup_2):
    """
    Merge 2 tuple of (y_proba_train, y_proba_tests, logs).
    NOTE: Now in splitting mode, the logs will be approximate log, because we should calculate metrics after collect
     y_proba, but now we calculate metrics on small forests, respectively, so the average of metrics of two forests
     must be inaccurate, you should mind this and do not worry about it. After merge, the average y_proba is the true
     proba, so the final results is absolutely right!

    :param tup_1: tuple like (y_proba_train, y_proba_tests, logs)
    :param tup_2: tuple like (y_proba_train, y_proba_tests, logs)
    :return:
    """
    tests = []
    for i in range(len(tup_1[1])):
        tests.append((tup_1[1][i] + tup_2[1][i])/2.0)
    mean_dict = defaultdict(float)
    logs = []
    for t1 in tup_1[2]:
        if t1[0] == 'INFO':
            mean_dict[','.join(t1[:2])] += t1[2]
        else:
            logs.append(t1)
    for t2 in tup_2[2]:
        if t2[0] == 'INFO':
            mean_dict[','.join(t2[:2])] += t2[2]
        else:
            logs.append(t2)
    for key in mean_dict.keys():
        mean_dict[key] = mean_dict[key] / 2.0
        key_split = key.split(',')
        logs.append((key_split[0], "Approximate " + key_split[1], mean_dict[key]))
    logs.sort()
    return (tup_1[0] + tup_2[0])/2.0, tests, logs


def est_class_from_type(task, est_type):
    """
    Get estimator class from task ('classification' or 'regression') and estimator type (a string).

    :param task: what task does this estimator to execute ('classification' or 'regression')
    :param est_type: estimator type (a string)
    :return: a concrete estimator instance
    """
    if task == 'classification':
        if est_type == 'FLCRF':
            return FLCRFClassifier
        if est_type == 'FLRF':
            return FLRFClassifier
        if est_type == 'FLGBDT':
            return FLGBDTClassifier
        if est_type == 'FLXGB':
            return FLXGBoostClassifier
        if est_type == 'FLLGBM':
            raise FLLGBMClassifier
        raise ValueError('Unknown Estimator: {}'.format(est_type))
    elif task == 'regression':
        if est_type == 'FLCRF':
            return FLCRFRegressor
        if est_type == 'FLRF':
            return FLRFRegressor
        if est_type == 'FLGBDT':
            return FLGBDTRegressor
        if est_type == 'FLXGB':
            return FLXGBoostRegressor
        if est_type == 'FLLGBM':
            return FLLGBMRegressor
        raise ValueError('Unknown Estimator: {}'.format(est_type))
    else:
        raise ValueError('Unknown task: {}'.format(task))


def get_estimator(name, task, est_type, est_args):
    """
    Get an estimator.

    :param name: estimator name
    :param task: what task does this estimator to execute ('classification' or 'regression')
    :param est_type: estimator type (a string)
    :param est_args: estimator arguments
    :return: a concrete estimator instance
    """
    est_class = est_class_from_type(task, est_type)
    return est_class(name, est_args)


def get_estimator_kfold(name, n_folds=3, task='classification', est_type='FLRF', eval_metrics=None, seed=None,
                        dtype=np.float32, cache_dir=None, keep_in_mem=True, est_args=None, cv_seed=None):
    """
    A factory method to get a k-fold estimator.

    :param name: estimator name
    :param n_folds: how many folds to execute in cross validation
    :param task: what task does this estimator to execute ('classification' or 'regression')
    :param est_type: estimator type (a string)
    :param eval_metrics: evaluation metrics. [Default: Accuracy (classification), MSE (regression)]
    :param seed: random seed
    :param dtype: data type
    :param cache_dir: data cache dir to cache intermediate data
    :param keep_in_mem: whether keep the model in memory
    :param est_args: estimator arguments
    :param cv_seed: random seed for cross validation
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
                        dtype=dtype,
                        eval_metrics=eval_metrics,
                        cache_dir=cache_dir,
                        keep_in_mem=keep_in_mem,
                        est_args=est_args,
                        cv_seed=cv_seed)


def get_dist_estimator_kfold(name, n_folds=3, task='classification', est_type='RF', eval_metrics=None, seed=None,
                             dtype=np.float32, cache_dir=None, keep_in_mem=True, est_args=None, cv_seed=None):
    """
    A factory method to get a distributed k-fold estimator.

    :param name: estimator name
    :param n_folds: how many folds to execute in cross validation
    :param task: what task does this estimator to execute ('classification' or 'regression')
    :param est_type: estimator type (a string)
    :param eval_metrics: evaluation metrics. [Default: Accuracy (classification), MSE (regression)]
    :param seed: random seed
    :param dtype: data type
    :param cache_dir: data cache dir to cache intermediate data
    :param keep_in_mem: whether keep the model in memory
    :param est_args: estimator arguments
    :param cv_seed: random seed for cross validation
    :return: a KFoldWrapper instance of concrete estimator
    """
    est_class = est_class_from_type(task, est_type)
    if eval_metrics is None:
        if task == 'classification':
            eval_metrics = [Accuracy('accuracy')]
        else:
            eval_metrics = [MSE('MSE')]
    return DistributedKFoldWrapper.remote(name,
                                          n_folds,
                                          est_class,
                                          seed=seed,
                                          dtype=dtype,
                                          eval_metrics=eval_metrics,
                                          cache_dir=cache_dir,
                                          keep_in_mem=keep_in_mem,
                                          est_args=est_args,
                                          cv_seed=cv_seed)

