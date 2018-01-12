# -*- coding:utf-8 -*-
"""
Base layers definition.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import numpy as np
import copy
import datetime
import os.path as osp
import pickle
from ..utils.log_utils import get_logger, list2str
from ..utils.utils import check_list_depth
from ..utils.storage_utils import check_dir
from ..utils.metrics import Accuracy, AUC, MSE
from ..estimators import get_estimator_kfold, EstimatorArgument


class Layer(object):
    """Abstract base layer class.

    # Properties
        name: String
        input_shape: Shape tuple.
        output_shape: Shape tuple.
        input, output: Input / output tensors.
        num_estimators: Number of estimators in this layer.
        estimators: Estimators in this layer.

    # Methods
        call(x): Where the layer logic lives.
        __call__(x): Wrapper around the layer logic (`call`).

    # Class Methods
        from_config(config)
    """
    def __init__(self, batch_size=None, dtype=None, name=None):
        """
        Initialize a layer.

        :param batch_size:
        :param dtype:
        :param name:
        """
        self.LOGGER = get_logger('layer')
        self.batch_size = batch_size
        if not name:
            prefix = self.__class__.__name__
            name = _to_snake_case(prefix) + "_" + str(id(self))
        self.name = name
        # Set dtype.
        if dtype is None:
            dtype = np.float32
        self.dtype = dtype

    def call(self, x_trains):
        raise NotImplementedError

    def __call__(self, x_trains):
        raise NotImplementedError

    def fit(self, x_trains, y_trains):
        """
        Fit datasets, return a list or single ndarray: train_outputs.
        NOTE: may change x_trains, y_trains

        :param x_trains: train data
        :param y_trains: train labels
        :return: train_outputs
        """
        raise NotImplementedError

    def fit_transform(self, x_trains, y_trains, x_tests=None, y_tests=None):
        """
        Fit and Transform datasets, return two lists or two single ndarrays: train_outputs, test_outputs.

        :param x_trains: train datasets
        :param y_trains: train labels
        :param x_tests: test datasets
        :param y_tests: test labels
        :return: train_outputs, test_outputs
        """
        raise NotImplementedError

    def transform(self, inputs):
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError

    def predict_proba(self, inputs):
        raise NotImplementedError

    def evaluate(self, inputs, labels):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class MultiGrainScanLayer(Layer):
    """
    Multi-grain Scan Layer
    """
    def __init__(self, batch_size=None, dtype=None, name=None, task='classification',
                 windows=None, est_for_windows=None, n_class=None, keep_in_mem=False, eval_metrics=None, seed=None):
        """
        Initialize a multi-grain scan layer.

        :param batch_size:
        :param dtype:
        :param name:
        :param task:
        :param windows:
        :param est_for_windows:
        :param n_class:
        :param keep_in_mem:
        :param eval_metrics:
        :param seed:
        """
        if not name:
            prefix = 'multi_grain_scan'
            name = prefix + '_' + str(id(self))
        super(MultiGrainScanLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        self.windows = windows  # [Win, Win, Win, ...]
        self.est_for_windows = est_for_windows  # [[est1, est2], [est1, est2], [est1, est2], ...]
        assert task in ['regression', 'classification'], 'task unknown! task = {}'.format(task)
        self.task = task
        if self.task == 'regression':
            self.n_class = 1
        else:
            assert n_class is not None
            self.n_class = n_class
        self.seed = seed
        self.keep_in_mem = keep_in_mem
        self.eval_metrics = eval_metrics

    def call(self, x_train, **kwargs):
        pass

    def __call__(self, x_train, **kwargs):
        pass

    def scan(self, window, x):
        """
        Multi-grain scan.

        :param window:
        :param x:
        :return:
        """
        return window.fit_transform(x)

    def _init_estimator(self, est_arguments, wi, ei):
        """
        Initialize an estimator.

        :param est_arguments:
        :param wi:
        :param ei:
        :return:
        """
        est_args = est_arguments.get_est_args()
        est_name = 'win - {} - estimator - {} - {}folds'.format(wi, ei, est_args['n_folds'])
        n_folds = int(est_args['n_folds'])
        est_args.pop('n_folds')
        est_type = est_args['est_type']
        est_args.pop('est_type')
        # seed
        if self.seed is not None:
            seed = (self.seed + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            seed = None
        return get_estimator_kfold(name=est_name,
                                   n_folds=n_folds,
                                   task=self.task,
                                   est_type=est_type,
                                   eval_metrics=self.eval_metrics,
                                   seed=seed,
                                   keep_in_mem=self.keep_in_mem,
                                   est_args=est_args)

    def _check_input(self, x, y):
        if isinstance(x, (list, tuple)):
            assert len(x) == 1, "Multi grain scan Layer only supports exactly one input now!"
            x = x[0]
        if isinstance(y, (list, tuple)):
            assert len(y) == 1, "Multi grain scan Layer only supports exactly one input now!"
            y = y[0]
        return x, y

    def fit(self, x_train, y_train):
        """
        Fit.

        :param x_train:
        :param y_train:
        :return:
        """
        x_train, y_train = self._check_input(x_train, y_train)
        x_wins_train = []
        for win in self.windows:
            x_wins_train.append(self.scan(win, x_train))
        self.LOGGER.info('X_wins of train: {}'.format([win.shape for win in x_wins_train]))
        x_win_est_train = []
        for wi, ests_for_win in enumerate(self.est_for_windows):
            if not isinstance(ests_for_win, (list, tuple)):
                ests_for_win = [ests_for_win]
            win_est_train = []
            # X_wins[wi] = (60000, 11, 11, 49)
            _, nh, nw, _ = x_wins_train[wi].shape
            # (60000, 121, 49)
            x_wins_train[wi] = x_wins_train[wi].reshape((x_wins_train[wi].shape[0], -1, x_wins_train[wi].shape[-1]))
            y_win = y_train[:, np.newaxis].repeat(x_wins_train[wi].shape[1], axis=1)
            for ei, est in enumerate(ests_for_win):
                if isinstance(est, EstimatorArgument):
                    est = self._init_estimator(est, wi, ei)
                # (60000, 121, 10)
                y_proba_train, _ = est.fit_transform(x_wins_train[wi], y_win, y_win[:, 0])
                y_proba_train = y_proba_train.reshape((-1, nh, nw, self.n_class)).transpose((0, 3, 1, 2))
                win_est_train.append(y_proba_train)
                self.est_for_windows[wi][ei] = est
            x_win_est_train.append(win_est_train)
        if len(x_win_est_train) == 0:
            return x_wins_train
        self.LOGGER.info('x_win_est_train.shape: {}'.format(list2str(x_win_est_train, 2)))
        return x_win_est_train

    def fit_transform(self, x_train, y_train, x_test=None, y_test=None):
        """
        Fit and transform.

        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        if x_test is None:
            return self.fit(x_train, y_train), None
        x_train, y_train = self._check_input(x_train, y_train)
        x_test, y_test = self._check_input(x_test, y_test)
        # Construct test sets
        x_wins_train = []
        x_wins_test = []
        for win in self.windows:
            x_wins_train.append(self.scan(win, x_train))
        for win in self.windows:
            x_wins_test.append(self.scan(win, x_test))
        # Deprecated: [[win, win], [win, win], ...], len = len(test_sets)
        # Deprecated: test_sets = [('testOfWin{}'.format(i), x, y) for i, x, y in enumerate(zip(x_wins_test, y_tests))]
        self.LOGGER.info('X_wins of train: {}'.format([win.shape for win in x_wins_train]))
        self.LOGGER.info('X_wins of  test: {}'.format([win.shape for win in x_wins_test]))
        x_win_est_train = []
        x_win_est_test = []
        for wi, ests_for_win in enumerate(self.est_for_windows):
            if not isinstance(ests_for_win, (list, tuple)):
                ests_for_win = [ests_for_win]
            win_est_train = []
            win_est_test = []
            # X_wins[wi] = (60000, 11, 11, 49)
            _, nh, nw, _ = x_wins_train[wi].shape
            # (60000, 121, 49)
            x_wins_train[wi] = x_wins_train[wi].reshape((x_wins_train[wi].shape[0], -1, x_wins_train[wi].shape[-1]))
            y_win = y_train[:, np.newaxis].repeat(x_wins_train[wi].shape[1], axis=1)
            x_wins_test[wi] = x_wins_test[wi].reshape((x_wins_test[wi].shape[0], -1, x_wins_test[wi].shape[-1]))
            y_win_test = y_test[:, np.newaxis].repeat(x_wins_test[wi].shape[1], axis=1)
            test_sets = [('testOfWin{}'.format(wi), x_wins_test[wi], y_win_test)]
            # fit estimators for this window
            for ei, est in enumerate(ests_for_win):
                if isinstance(est, EstimatorArgument):
                    est = self._init_estimator(est, wi, ei)
                # (60000, 121, 10)
                y_proba_train, y_probas_test = est.fit_transform(x_wins_train[wi], y_win, y_win[:, 0], test_sets)
                y_proba_train = y_proba_train.reshape((-1, nh, nw, self.n_class)).transpose((0, 3, 1, 2))
                assert len(y_probas_test) == 1, 'assume there is only one test set!'
                y_probas_test = y_probas_test[0]
                y_probas_test = y_probas_test.reshape((-1, nh, nw, self.n_class)).transpose((0, 3, 1, 2))
                win_est_train.append(y_proba_train)
                win_est_test.append(y_probas_test)
                if self.keep_in_mem:
                    self.est_for_windows[wi][ei] = est
            x_win_est_train.append(win_est_train)
            x_win_est_test.append(win_est_test)
        if len(x_win_est_train) == 0:
            return x_wins_train, x_wins_test
        self.LOGGER.info('x_win_est_train.shape: {}'.format(list2str(x_win_est_train, 2)))
        self.LOGGER.info(' x_win_est_test.shape: {}'.format(list2str(x_win_est_test, 2)))
        return x_win_est_train, x_win_est_test

    def transform(self, x_train):
        """
        Transform.

        :param x_train:
        :return:
        """
        assert x_train is not None, 'x_trains should not be None!'
        if isinstance(x_train, (list, tuple)):
            assert len(x_train) == 1, "Multi grain scan Layer only supports exactly one input now!"
            x_train = x_train[0]
        x_wins_train = []
        for win in self.windows:
            x_wins_train.append(self.scan(win, x_train))
        # [[win, win], [win, win], ...], len = len(test_sets)
        self.LOGGER.info('X_wins of train: {}'.format([win.shape for win in x_wins_train]))
        x_win_est_train = []
        for wi, ests_for_win in enumerate(self.est_for_windows):
            if not isinstance(ests_for_win, (list, tuple)):
                ests_for_win = [ests_for_win]
            win_est_train = []
            # X_wins[wi] = (60000, 11, 11, 49)
            _, nh, nw, _ = x_wins_train[wi].shape
            # (60000, 121, 49)
            x_wins_train[wi] = x_wins_train[wi].reshape((x_wins_train[wi].shape[0], -1, x_wins_train[wi].shape[-1]))
            for ei, est in enumerate(ests_for_win):
                # (60000, 121, 10)
                y_proba_train = est.transform(x_wins_train[wi])
                y_proba_train = y_proba_train.reshape((-1, nh, nw, self.n_class)).transpose((0, 3, 1, 2))
                win_est_train.append(y_proba_train)
            x_win_est_train.append(win_est_train)
        if len(x_win_est_train) == 0:
            return x_wins_train
        self.LOGGER.info('[transform] win_est_train.shape: {}'.format(list2str(x_win_est_train, 2)))
        return x_win_est_train

    def predict(self, X):
        return self.transform(X)

    def predict_proba(self, X):
        return self.transform(X)

    def evaluate(self, inputs, labels):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class PoolingLayer(Layer):
    """
    Pooling layer.
    """
    def __init__(self, batch_size=None, dtype=None, name=None, pools=None):
        """
        Initialize a pooling layer.

        :param batch_size:
        :param dtype:
        :param name:
        :param pools:
        """
        super(PoolingLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        # [[pool/7x7/est1, pool/7x7/est2], [pool/11x11/est1, pool/11x11/est1], [pool/13x13/est1, pool/13x13/est1], ...]
        self.pools = pools

    def call(self, x_trains, **kwargs):
        pass

    def __call__(self, x_trains):
        pass

    def fit(self, x_trains, y_trains=None):
        """
        Fit.

        :param x_trains:
        :param y_trains:
        :return:
        """
        # inputs shape: [[(60000, 10, 11, 11), (60000, 10, 11, 11)], [.., ..], ...]
        if len(self.pools) != len(x_trains):
            raise ValueError('len(pools) does not equal to len(inputs), you must set right pools!')
        for pi, pool in enumerate(self.pools):
            if not isinstance(pool, (list, tuple)):
                pool = [pool]
            if len(pool) != len(x_trains[pi]):
                raise ValueError('len(pools[{}]) does not equal to'
                                 ' len(inputs[{}]), you must set right pools!'.format(pi, pi))
            for pj, pl in enumerate(pool):
                x_trains[pi][pj] = pl.fit_transform(x_trains[pi][pj])
        self.LOGGER.info('x_trains pooled: {}'.format(list2str(x_trains, 2)))
        return x_trains

    def fit_transform(self, x_trains, y_trains=None, x_tests=None, y_tests=None):
        """
        Fit transform.

        :param x_trains:
        :param y_trains:
        :param x_tests:
        :param y_tests:
        :return:
        """
        if x_tests is None:
            return self.fit(x_trains, y_trains), None
        # inputs shape: [[(60000, 10, 11, 11), (60000, 10, 11, 11)], [.., ..], ...]
        if len(self.pools) != len(x_trains):
            raise ValueError('len(pools) does not equal to len(x_trains), you must set right pools!')
        if len(self.pools) != len(x_tests):
            raise ValueError('len(pools) does not equal to len(x_tests), you must set right pools!')
        for pi, pool in enumerate(self.pools):
            if not isinstance(pool, (list, tuple)):
                pool = [pool]
            if len(pool) != len(x_trains[pi]):
                raise ValueError('len(pools[{}]) does not equal to'
                                 ' len(train inputs[{}]), you must set right pools!'.format(pi, pi))
            if len(pool) != len(x_tests[pi]):
                raise ValueError('len(pools[{}]) does not equal to'
                                 ' len(test inputs[{}]), you must set right pools!'.format(pi, pi))
            for pj, pl in enumerate(pool):
                x_trains[pi][pj] = pl.fit_transform(x_trains[pi][pj])
                x_tests[pi][pj] = pl.fit_transform(x_tests[pi][pj])
        self.LOGGER.info('x_trains pooled: {}'.format(list2str(x_trains, 2)))
        self.LOGGER.info('x_tests  pooled: {}'.format(list2str(x_tests, 2)))
        return x_trains, x_tests

    def transform(self, xs):
        """
        Transform.

        :param xs:
        :return:
        """
        assert xs is not None, 'x_trains should not be None!'
        if not isinstance(xs, (list, tuple)):
            xs = [xs]
        # inputs shape: [[(60000, 10, 11, 11), (60000, 10, 11, 11)], [.., ..], ...]
        if len(self.pools) != len(xs):
            raise ValueError('len(pools) does not equal to len(inputs), you must set right pools!')
        for pi, pool in enumerate(self.pools):
            if not isinstance(pool, (list, tuple)):
                pool = [pool]
            if len(pool) != len(xs[pi]):
                raise ValueError('len(pools[{}]) does not equal to'
                                 ' len(inputs[{}]), you must set right pools!'.format(pi, pi))
            for pj, pl in enumerate(pool):
                xs[pi][pj] = pl.transform(xs[pi][pj])
        self.LOGGER.info('[transform] x_trains pooled: {}'.format(list2str(xs, 2)))
        return xs

    def evaluate(self, inputs, labels):
        raise NotImplementedError

    def predict(self, X):
        return self.transform(X)

    def predict_proba(self, X):
        return self.transform(X)


class ConcatLayer(Layer):
    """
    Concatenate layer.
    """
    def __init__(self, batch_size=None, dtype=None, name=None, axis=-1):
        """
        Initialize a concat layer.

        :param batch_size:
        :param dtype:
        :param name:
        :param axis:
        """
        super(ConcatLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        # [[pool/7x7/est1, pool/7x7/est2], [pool/11x11/est1, pool/11x11/est1], [pool/13x13/est1, pool/13x13/est1], ...]
        # to
        # [Concat(axis=axis), Concat(axis=axis), Concat(axis=axis), ...]
        self.axis = axis

    def call(self, X, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def _fit(self, xs):
        """
        fit inner method.

        :param xs:
        :return:
        """
        if not isinstance(xs, (list, tuple)):
            xs = [xs]
        # inputs shape: [[(60000, 10, 6, 6), (60000, 10, 6, 6)], [.., ..], ...]
        concat_results = []
        for bottoms in xs:
            if self.axis == -1:
                for i, bottom in enumerate(bottoms):
                    bottoms[i] = bottom.reshape((bottom.shape[0], -1))
                concat_res = np.concatenate(bottoms, 1)
            else:
                concat_res = np.concatenate(bottoms, self.axis)
            concat_results.append(concat_res)
        return concat_results

    def transform(self, xs):
        """
        Transform.

        :param xs:
        :return:
        """
        concat = self._fit(xs)
        self.LOGGER.info("[transform] concatenated shape: {}".format(list2str(concat, 1)))
        return concat

    def evaluate(self, inputs, labels=None):
        raise NotImplementedError

    def fit(self, x_trains, y_trains=None):
        """
        Fit.

        :param x_trains:
        :param y_trains:
        :return:
        """
        concat_train = self._fit(x_trains)
        self.LOGGER.info("concat train shape: {}".format(list2str(concat_train, 1)))
        return concat_train

    def fit_transform(self, x_trains, y_trains, x_tests=None, y_tests=None):
        """
        Fit transform.

        :param x_trains:
        :param y_trains:
        :param x_tests:
        :param y_tests:
        :return:
        """
        if x_tests is None:
            return self.fit(x_trains, y_trains), None
        # inputs shape: [[(60000, 10, 6, 6), (60000, 10, 6, 6)], [.., ..], ...]
        concat_train = self._fit(x_trains)
        self.LOGGER.info("concat train shape: {}".format(list2str(concat_train, 1)))
        concat_test = self._fit(x_tests)
        self.LOGGER.info(" concat test shape: {}".format(list2str(concat_test, 1)))
        return concat_train, concat_test

    def predict(self, X):
        return self.transform(X)

    def predict_proba(self, X):
        return self.transform(X)


class CascadeLayer(Layer):
    def __init__(self, batch_size=None, dtype=None, name=None, task='classification', est_configs=None,
                 layer_id='anonymous', n_classes=None, keep_in_mem=False, data_save_dir=None, model_save_dir=None,
                 metrics=None, seed=None):
        """Cascade Layer.
        A cascade layer contains several estimators, it accepts single input, go through these estimators, produces
        predicted probability by every estimators, and stacks them together for next cascade layer.

        :param batch_size: cascade layer do not need batch_size actually.
        :param dtype: data type
        :param name: name of this layer
        :param task: classification or regression, [default = classification]
        :param est_configs: list of estimator arguments, every argument can be `dict` or `EstimatorArgument` instance
                            identify the estimator configuration to construct at this layer
        :param layer_id: layer id, if this layer is an independent layer, layer id is anonymous [default]
        :param n_classes: number of classes to classify
        :param keep_in_mem: identifies whether keep the model in memory, if fit_transform,
                            we recommend set it False to save memory and speed up the application
                            TODO: support dump model to disk to save memory
        :param data_save_dir: directory to save intermediate data into
        :param model_save_dir: directory to save fit estimators into
        :param metrics: str, evaluation metrics used in training model and evaluating testing data.
                        Support: 'accuracy', 'auc', 'mse', default is accuracy (classification) and mse (regression).
        :param seed: random seed, also called random state in scikit-learn random forest

        # Properties
            eval_metrics: evaluation metrics
            fit_estimators: estimator instances after fit
            train_avg_metric: training average metric
            test_avg_metric: testing average metric

        # Raises
            RuntimeError: if estimator.fit_transform returns None data
            ValueError: if estimator.fit_transform returns wrong shape data
        """
        self.est_configs = [] if est_configs is None else est_configs
        # transform EstimatorArgument to dict that represents estimator arguments
        for eci, est_config in enumerate(self.est_configs):
            if isinstance(est_config, EstimatorArgument):
                self.est_configs[eci] = est_config.get_est_args().copy()
        self.layer_id = layer_id
        if not name:
            name = 'layer-{}'.format(self.layer_id)
        super(CascadeLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        self.task = task
        self.n_classes = n_classes
        if self.task == 'regression':
            self.n_classes = 1
        self.keep_in_mem = keep_in_mem
        self.data_save_dir = data_save_dir
        check_dir(self.data_save_dir)  # check dir, if not exists, create the dir
        self.model_save_dir = model_save_dir
        check_dir(self.model_save_dir)
        self.seed = seed
        self.larger_better = True
        self.metrics = metrics
        if self.metrics == 'accuracy':
            self.eval_metrics = [Accuracy('accuracy')]
        elif self.metrics == 'auc':
            self.eval_metrics = [AUC('AUC')]
        elif self.metrics == 'mse':
            self.eval_metrics = [MSE('Mean Square Error')]
        else:
            if self.task == 'regression':
                self.eval_metrics = [MSE('Mean Square Error')]
            else:
                self.eval_metrics = [Accuracy('Accuracy')]
        # whether this layer the last layer of Auto-growing cascade layer
        self.complete = False
        self.fit_estimators = [None for _ in range(self.n_estimators)]
        self.train_avg_metric = None
        self.test_avg_metric = None
        self.eval_proba_test = None

    def call(self, inputs, **kwargs):
        return inputs

    def __call__(self, inputs, **kwargs):
        self.call(inputs, **kwargs)

    def _concat(self, x, depth):
        """
        Concatenation inner method, to make multiple inputs to be single input, so that to feed it into classifiers.

        :param x: input data, single ndarray(depth=0) or list(depth=1) or 2D list (depth=2), at most 2D.
        :param depth: as stated above, single ndarray(depth=0) or list(depth=1) or 2D list (depth=2)
        :return: concatenated data
        """
        if depth == 0:
            return x
        elif depth == 1:
            for i, bottom in enumerate(x):
                x[i] = bottom.reshape((bottom.shape[0], -1))
            x = np.concatenate(x, 1)
        elif depth == 2:
            for i, bottoms in enumerate(x):
                for j, bot in enumerate(bottoms):
                    bottoms[j] = bot.reshape((bot.shape[0], -1))
                x[i] = np.concatenate(bottoms, 1)
            for i, bottom in enumerate(x):
                x[i] = bottom.reshape((bottom.shape[0], -1))
            x = np.concatenate(x, 1)
        else:
            raise ValueError('_concat failed. depth should be less than 2!')
        return x

    def _validate_input(self, x_train, y_train, x_test=None, y_test=None):
        """
        Validate input, check if x_train / x_test s' depth, and do some necessary transform like concatenation.

        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        assert x_train is not None and y_train is not None, 'x_train is None or y_train should not be None'
        train_depth = 0
        if isinstance(x_train, (list, tuple)):
            train_depth = check_list_depth(x_train)
        x_train = self._concat(x_train, train_depth)
        if x_test is not None:
            test_depth = 0
            if isinstance(x_test, (list, tuple)):
                test_depth = check_list_depth(x_test)
            x_test = self._concat(x_test, test_depth)
        if isinstance(y_train, (list, tuple)) and y_train is not None:
            y_train = None if len(y_train) == 0 else y_train[0]
        if isinstance(y_test, (list, tuple)) and y_test is not None:
            y_test = None if len(y_test) == 0 else y_test[0]
        return x_train, y_train, x_test, y_test

    def _init_estimators(self, layer_id, est_id):
        """
        Initialize a k_fold estimator.

        :param layer_id:
        :param est_id:
        :return:
        """
        est_args = self.est_configs[est_id].copy()
        est_name = 'layer - {} - estimator - {} - {}folds'.format(layer_id, est_id, est_args['n_folds'])
        n_folds = int(est_args['n_folds'])
        est_args.pop('n_folds')
        est_type = est_args['est_type']
        est_args.pop('est_type')
        # seed
        if self.seed is not None:
            seed = (self.seed + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            seed = None
        return get_estimator_kfold(name=est_name,
                                   n_folds=n_folds,
                                   task=self.task,
                                   est_type=est_type,
                                   eval_metrics=self.eval_metrics,
                                   seed=seed,
                                   keep_in_mem=self.keep_in_mem,
                                   est_args=est_args)

    def fit(self, x_train, y_train):
        """
        Fit and Transform datasets, return one numpy ndarray: train_output
        NOTE: Only one train set and one test set.

        :param x_train: train datasets
        :param y_train: train labels
        :return: train_output
        """
        x_train, y_train, _, _ = self._validate_input(x_train, y_train)
        assert x_train.shape[0] == y_train.shape[0], 'x_train.shape[0] = {} not equal to y_train.shape[0]' \
                                                     ' = {}'.format(x_train.shape[0], y_train.shape[0])
        self.LOGGER.info('X_train.shape={}, y_train.shape={}'.format(x_train.shape, y_train.shape))
        n_trains = x_train.shape[0]
        n_classes = self.n_classes  # if regression, n_classes = 1
        if self.task == 'classification' and n_classes is None:
            n_classes = np.unique(y_train)
        if self.task == 'regression' and n_classes is None:
            n_classes = 1
        x_proba_train = np.zeros((n_trains, n_classes * self.n_estimators), dtype=np.float32)
        eval_proba_train = np.zeros((n_trains, n_classes))
        # fit estimators, get probas (classification) or targets (regression)
        for ei in range(self.n_estimators):
            est = self._init_estimators(self.layer_id, ei)
            # fit and transform
            y_stratify = y_train if self.task == 'classification' else None
            y_proba_train, _ = est.fit_transform(x_train, y_train, y_stratify, test_sets=None)
            # print(y_proba_train.shape, y_proba_test.shape)
            if y_proba_train is None:
                raise RuntimeError("layer - {} - estimator - {} fit FAILED!,"
                                   " y_proba_train is None!".format(self.layer_id, ei))
            self.check_shape(y_proba_train, n_trains, n_classes)
            if self.keep_in_mem:
                self.fit_estimators[ei] = est
            x_proba_train[:, ei * n_classes:ei * n_classes + n_classes] = y_proba_train
            eval_proba_train += y_proba_train
        eval_proba_train /= self.n_estimators
        # now supports one eval_metrics
        metric = self.eval_metrics[0]
        train_avg_acc = metric.calc_proba(y_train, eval_proba_train,
                                          'layer - {} - [train] average'.format(self.layer_id), logger=self.LOGGER)
        self.train_avg_metric = train_avg_acc
        return x_proba_train

    def fit_transform(self, x_train, y_train, x_test=None, y_test=None):
        """
        Fit and Transform datasets, return two numpy ndarray: train_output, test_output
        NOTE: Only one train set and one test set.
        if x_test is None, we invoke _fit_transform to get one numpy ndarray: train_output

        :param x_train: training data
        :param y_train: training label
        :param x_test: testing data
        :param y_test: testing label, can be None,
                       if None, we see that the fit_transform must give the predictions of x_test.
        :return: train_output, test_output
        """
        if x_test is None:
            return self.fit(x_train, y_train), None
        x_train, y_train, x_test, y_test = self._validate_input(x_train, y_train, x_test, y_test)
        if y_test is None:
            y_test_shape = (0,)
        else:
            y_test_shape = y_test.shape
        self.LOGGER.info('X_train.shape={}, y_train.shape={}'.format(x_train.shape, y_train.shape))
        self.LOGGER.info(' X_test.shape={},  y_test.shape={}'.format(x_test.shape, y_test_shape))
        n_trains = x_train.shape[0]
        n_tests = x_test.shape[0]
        n_classes = self.n_classes  # if regression, n_classes = 1
        if self.task == 'classification' and n_classes is None:
            n_classes = np.unique(y_train)
        if self.task == 'regression' and n_classes is None:
            n_classes = 1
        x_proba_train = np.zeros((n_trains, n_classes * self.n_estimators), dtype=np.float32)
        x_proba_test = np.zeros((n_tests, n_classes * self.n_estimators), dtype=np.float32)
        eval_proba_train = np.zeros((n_trains, n_classes))
        eval_proba_test = np.zeros((n_tests, n_classes))
        # fit estimators, get probas
        for ei in range(self.n_estimators):
            est = self._init_estimators(self.layer_id, ei)
            # fit and transform
            y_stratify = y_train if self.task == 'classification' else None
            y_proba_train, y_proba_test = est.fit_transform(x_train, y_train, y_stratify,
                                                            test_sets=[('test', x_test, y_test)])
            # if only one element on test_sets, return one test result like y_proba_train
            if isinstance(y_proba_test, (list, tuple)) and len(y_proba_test) == 1:
                y_proba_test = y_proba_test[0]
            # print(y_proba_train.shape, y_proba_test.shape)
            if y_proba_train is None:
                raise RuntimeError("layer - {} - estimator - {} fit FAILED!,"
                                   " y_proba_train is None".format(self.layer_id, ei))
            self.check_shape(y_proba_train, n_trains, n_classes)
            if y_proba_test is not None:
                self.check_shape(y_proba_test, n_tests, n_classes)
            if self.keep_in_mem:
                self.fit_estimators[ei] = est
            x_proba_train[:, ei*n_classes:ei*n_classes + n_classes] = y_proba_train
            x_proba_test[:, ei*n_classes:ei*n_classes + n_classes] = y_proba_test
            eval_proba_train += y_proba_train
            eval_proba_test += y_proba_test
        eval_proba_train /= self.n_estimators
        eval_proba_test /= self.n_estimators
        metric = self.eval_metrics[0]
        train_avg_metric = metric.calc_proba(y_train, eval_proba_train,
                                             'layer - {} - [train] average {}'.format(self.layer_id, metric.name),
                                             logger=self.LOGGER)
        self.train_avg_metric = train_avg_metric
        # judge whether y_test is None, which means users are to predict test probas
        if y_test is not None:
            test_avg_metric = metric.calc_proba(y_test, eval_proba_test,
                                                'layer - {} - [test] average'.format(self.layer_id), logger=self.LOGGER)
            self.test_avg_metric = test_avg_metric
        # if y_test is None, we need to generate test prediction, so keep eval_proba_test
        if y_test is None:
            self.eval_proba_test = eval_proba_test
        return x_proba_train, x_proba_test

    def check_shape(self, y_proba, n, n_classes):
        if y_proba.shape != (n, n_classes):
            raise ValueError('output shape incorrect!,'
                             ' should be {}, but {}'.format((n, n_classes), y_proba.shape))

    @property
    def n_estimators(self):
        """
        Number of estimators of this layer.

        :return:
        """
        return len(self.est_configs)

    def transform(self, X):
        """
        Transform datasets, return one numpy ndarray.
        NOTE: Only one train set and one test set.

        :param X: train datasets
        :return:
        """
        if isinstance(X, (list, tuple)):
            X = None if len(X) == 0 else X[0]
        n_trains = X.shape[0]
        n_classes = self.n_classes
        x_proba = np.zeros((n_trains, n_classes * self.n_estimators), dtype=np.float32)
        # fit estimators, get probas
        for ei, est in enumerate(self.fit_estimators):
            # transform by n-folds CV
            y_proba = est.transform(X)
            if y_proba is None:
                raise RuntimeError("layer - {} - estimator - {} transform FAILED!".format(self.layer_id, ei))
            self.check_shape(y_proba, n_trains, n_classes)
            x_proba[:, ei * n_classes:ei * n_classes + n_classes] = y_proba
        return x_proba

    @property
    def is_classification(self):
        return self.task == 'classification'

    def predict(self, X):
        """
        Predict data X.

        :param X:
        :return:
        """
        proba_sum = self.predict_proba(X)
        n_classes = self.n_classes
        return np.argmax(proba_sum.reshape((-1, n_classes)), axis=1)

    def predict_proba(self, X):
        """
        Transform datasets, return one numpy ndarray.
        NOTE: Only one train set and one test set.

        :param X: train datasets
        :return:
        """
        if isinstance(X, (list, tuple)):
            X = None if len(X) == 0 else X[0]
        n_trains = X.shape[0]
        n_classes = self.n_classes
        proba_sum = np.zeros((n_trains, n_classes), dtype=np.float32)
        # fit estimators, get probas
        for ei, est in enumerate(self.fit_estimators):
            # transform by n-folds CV
            y_proba_train = est.transform(X)
            if y_proba_train is None:
                raise RuntimeError("layer - {} - estimator - {} transform FAILED!".format(self.layer_id, ei))
            self.check_shape(y_proba_train, n_trains, n_classes)
            proba_sum += y_proba_train
        return proba_sum

    def evaluate(self, X, y, eval_metrics=None):
        """
        Evaluate dataset (X, y) with evaluation metrics.

        :param X: data
        :param y: label
        :param eval_metrics: evaluation metrics
        :return: None
        """
        if eval_metrics is None:
            eval_metrics = [Accuracy('evaluate')]
        if isinstance(y, (list, tuple)):
            assert len(y) == 1, 'only support single labels array'
            y = y[0]
        pred = self.predict(X)
        for metric in eval_metrics:
            metric.calc(y, pred, logger=self.LOGGER)


class AutoGrowingCascadeLayer(Layer):
    def __init__(self, batch_size=None, dtype=np.float32, name=None, task='classification', est_configs=None,
                 early_stopping_rounds=None, max_layers=0, look_index_cycle=None, data_save_rounds=0,
                 stop_by_test=True, n_classes=None, keep_in_mem=False, data_save_dir=None, model_save_dir=None,
                 metrics=None, keep_test_result=False, seed=None):
        """AutoGrowingCascadeLayer
        An AutoGrowingCascadeLayer is a virtual layer that consists of many single cascade layers.
        `auto-growing` means this kind of layer can decide the depth of cascade forest,
         by training error or testing error.

        :param batch_size: cascade layer do not need batch_size actually.
        :param dtype: data type
        :param name: name of this layer
        :param task: classification or regression, [default = classification]
        :param est_configs: list of estimator arguments, every argument can be `dict` or `EstimatorArgument` instance
                            identify the estimator configuration to construct at this layer
        :param early_stopping_rounds: early stopping rounds, if there is no increase in performance (training accuracy
                                      or testing accuracy) over `early_stopping_rounds` layer, we stop the training
                                      process to save time and storage. And we keep first optimal_layer_id cascade
                                      layer models, and predict/evaluate according to these cascade layer.
        :param max_layers: max layers to growing
                           0 means using Early Stopping to automatically find the layer number
        :param look_index_cycle: (2d list): default = None = [[i,] for i in range(n_groups)]
                                 specification for layer i, look for the array in
                                 look_index_cycle[i % len(look_index_cycle)]
                                 .e.g. look_index_cycle = [[0,1],[2,3],[0,1,2,3]]
                                 means layer 1 look for the grained 0,1; layer 2 look for grained 2,3;
                                 layer 3 look for every grained, and layer 4 cycles back as layer 1
        :param data_save_rounds: int [default = 0, means no savings for intermediate results]
        :param stop_by_test: boolean, identifies whether conduct early stopping by testing metric
                             [default = False]
        :param n_classes: number of classes
        :param keep_in_mem: boolean, identifies whether keep model in memory. [default = False] to save memory
        :param data_save_dir: str [default = None]
                              each data_save_rounds save the intermediate results in data_save_dir
                              if data_save_rounds = 0, then no savings for intermediate results
        :param model_save_dir: directory to save fit estimators into
        :param metrics: evaluation metrics used in training model and evaluating testing data
        :param seed: random seed, also called random state in scikit-learn random forest
        """
        self.est_configs = [] if est_configs is None else est_configs
        super(AutoGrowingCascadeLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        self.task = task
        self.early_stop_rounds = early_stopping_rounds
        self.max_layers = max_layers
        self.n_classes = n_classes
        if self.task == 'regression':
            self.n_classes = 1
        # if look_index_cycle is None, you need set look_index_cycle in fit / fit_transform
        self.look_index_cycle = look_index_cycle
        self.data_save_rounds = data_save_rounds
        self.data_save_dir = data_save_dir
        check_dir(self.data_save_dir)  # check data save dir, if not exists, create the dir
        self.model_save_dir = model_save_dir
        check_dir(self.model_save_dir)
        self.keep_in_mem = keep_in_mem
        self.stop_by_test = stop_by_test
        self.metrics = metrics
        if self.metrics == 'accuracy':
            self.eval_metrics = [Accuracy('accuracy')]
        elif self.metrics == 'auc':
            self.eval_metrics = [AUC('auc')]
        elif self.metrics == 'mse':
            self.eval_metrics = [MSE('Mean Square Error')]
        else:
            if self.task == 'regression':
                self.eval_metrics = [MSE('Mean Square Error')]
            else:
                self.eval_metrics = [Accuracy('accuracy')]
        self.seed = seed
        # properties
        self.layer_fit_cascades = []
        self.n_layers = 0
        self.opt_layer_id = 0
        self.n_group_train = 0
        self.group_starts = []
        self.group_ends = []
        self.group_dims = []
        self.test_results = None
        self.keep_test_result = keep_test_result

    def _create_cascade_layer(self, task='classification', est_configs=None, n_classes=None,
                              data_save_dir=None, model_save_dir=None, layer_id=None, keep_in_mem=False,
                              dtype=None, metrics=None, seed=None):
        """
        Create a cascade layer.

        :param task:
        :param est_configs:
        :param n_classes:
        :param data_save_dir:
        :param model_save_dir:
        :param layer_id:
        :param keep_in_mem:
        :param dtype:
        :param metrics:
        :param seed:
        :return:
        """
        return CascadeLayer(dtype=dtype, task=task, est_configs=est_configs, layer_id=layer_id, n_classes=n_classes,
                            keep_in_mem=keep_in_mem, data_save_dir=data_save_dir, model_save_dir=model_save_dir,
                            metrics=metrics, seed=seed)

    def call(self, x_trains):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def add(self, est):
        """
        Add an estimator to the auto growing cascade layer.

        :param est:
        :return:
        """
        if isinstance(est, EstimatorArgument):
            self.est_configs.append(est.get_est_args())
        elif isinstance(est, dict):
            self.est_configs.append(est)
        else:
            raise ValueError("Unknown estimator information {}".format(est))

    @property
    def _percent(self):
        return '%' if isinstance(self.eval_metrics[0], Accuracy) else ''

    @property
    def is_classification(self):
        return self.task == 'classification'

    @property
    def larger_better(self):
        """
        True if the evaluation metric larger is better.
        :return:
        """
        if isinstance(self.eval_metrics[0], (MSE, )):
            return False
        return True

    def fit(self, x_trains, y_train):
        """
        Fit with x_trains, y_trains.

        :param x_trains:
        :param y_train:
        :return:
        """
        x_trains, y_train, _, _ = self._validate_input(x_trains, y_train)
        if self.stop_by_test is True:
            self.LOGGER.warn('stop_by_test is True, but we do not obey it when fit(x_train, y_train)!')
        self.layer_fit_cascades = []
        n_groups_train = len(x_trains)
        self.n_group_train = n_groups_train
        n_trains = len(y_train)
        # Initialize the groups
        x_train_group = np.zeros((n_trains, 0), dtype=x_trains[0].dtype)
        group_starts, group_ends, group_dims = [], [], []
        # train set
        for i, x_train in enumerate(x_trains):
            assert x_train.shape[0] == n_trains, 'x_train.shape[0]={} not equal to' \
                                                 ' n_trains={}'.format(x_train.shape[0], n_trains)
            x_train = x_train.reshape(n_trains, -1)
            group_dims.append(x_train.shape[1])
            group_starts.append(i if i == 0 else group_ends[i - 1])
            group_ends.append(group_starts[i] + group_dims[i])
            x_train_group = np.hstack((x_train_group, x_train))

        self.LOGGER.info('group_starts={}'.format(group_starts))
        self.LOGGER.info('group_dims={}'.format(group_dims))
        self.LOGGER.info('X_train_group={}'.format(x_train_group.shape))
        self.group_starts = group_starts
        self.group_ends = group_ends
        self.group_dims = group_dims
        if self.look_index_cycle is None:
            self.look_index_cycle = [[i, ] for i in range(n_groups_train)]
        x_cur_train = None
        x_proba_train = np.zeros((n_trains, 0), dtype=np.float32)
        layer_id = 0
        layer_metric_list = []
        opt_data = [None, None]
        try:
            while True:
                if layer_id >= self.max_layers > 0:
                    break
                # clear x_cur_train
                x_cur_train = np.zeros((n_trains, 0), dtype=np.float32)
                train_ids = self.look_index_cycle[layer_id % n_groups_train]
                for gid in train_ids:
                    x_cur_train = np.hstack((x_cur_train, x_train_group[:, group_starts[gid]:group_ends[gid]]))
                x_cur_train = np.hstack((x_cur_train, x_proba_train))
                data_save_dir = self.data_save_dir
                if data_save_dir is not None:
                    data_save_dir = osp.join(data_save_dir, 'cascade_layer_{}'.format(layer_id))
                model_save_dir = self.model_save_dir
                if model_save_dir is not None:
                    model_save_dir = osp.join(model_save_dir, 'cascade_layer_{}'.format(layer_id))
                cascade = self._create_cascade_layer(task=self.task,
                                                     est_configs=self.est_configs,
                                                     n_classes=self.n_classes,
                                                     data_save_dir=data_save_dir,
                                                     model_save_dir=model_save_dir,
                                                     layer_id=layer_id,
                                                     keep_in_mem=self.keep_in_mem,
                                                     dtype=self.dtype,
                                                     metrics=self.eval_metrics,
                                                     seed=self.seed)
                x_proba_train, _ = cascade.fit_transform(x_cur_train, y_train)
                if self.keep_in_mem:
                    self.layer_fit_cascades.append(cascade)
                layer_metric_list.append(cascade.train_avg_metric)
                # detect best layer id
                opt_layer_id = get_opt_layer_id(layer_metric_list, self.larger_better)
                self.opt_layer_id = opt_layer_id
                # if this layer is the best layer, set the opt_data
                if opt_layer_id == layer_id:
                    opt_data = [x_cur_train, y_train]
                # early stopping
                if layer_id - opt_layer_id >= self.early_stop_rounds > 0:
                    # log and save the final results of the optimal layer
                    self.LOGGER.info('[Result][Early Stop][Optimal Layer Detected] opt_layer={},'.format(opt_layer_id) +
                                     ' {}_train={:.4f}{},'.format(self.eval_metrics[0].name,
                                                                  layer_metric_list[opt_layer_id], self._percent))
                    self.n_layers = layer_id + 1
                    self._save_data(opt_layer_id, *opt_data)
                    # wash the fit cascades after optimal layer id to save memory
                    if self.keep_in_mem:
                        for li in range(opt_layer_id + 1, layer_id + 1):
                            self.layer_fit_cascades[li] = None
                    return x_cur_train
                if self.data_save_rounds > 0 and (layer_id + 1) % self.data_save_rounds == 0:
                    self._save_data(layer_id, *opt_data)
                layer_id += 1
            # Max Layer Reached
            opt_data = [x_cur_train, y_train]
            opt_layer_id = get_opt_layer_id(layer_metric_list, larger_better=self.larger_better)
            self.opt_layer_id = opt_layer_id
            self.LOGGER.info('[Result][Max Layer Reach] max_layer={}, {}_train={:.4f}{},'
                             ' optimal_layer={}, {}_optimal_train={:.4f}{}'.format(
                                self.max_layers,
                                self.eval_metrics[0].name, layer_metric_list[-1], self._percent, opt_layer_id,
                                self.eval_metrics[0].name, layer_metric_list[opt_layer_id], self._percent))
            self._save_data(layer_id, *opt_data)
            self.n_layers = layer_id + 1
            # wash the fit cascades after optimal layer id to save memory
            if self.keep_in_mem:
                for li in range(opt_layer_id + 1, layer_id + 1):
                    self.layer_fit_cascades[li] = None
            return x_cur_train
        except KeyboardInterrupt:
            pass

    def fit_transform(self, x_trains, y_train, x_tests=None, y_test=None):
        """
        NOTE: Only support ONE x_train and one x_test, so y_train is a single numpy array instead of list of it.

        :param x_trains:
        :param y_train:
        :param x_tests:
        :param y_test:
        :return:
        """
        if x_tests is None:
            return self.fit(x_trains, y_train), None
        x_trains, y_train, x_tests, y_test = self._validate_input(x_trains, y_train, x_tests, y_test)
        self.layer_fit_cascades = []
        n_groups_train = len(x_trains)
        self.n_group_train = n_groups_train
        n_groups_test = len(x_tests)
        n_trains = len(y_train)
        n_tests = x_tests[0].shape[0]  # y_test might be None
        if y_test is None and self.stop_by_test is True:
            self.stop_by_test = False
            self.LOGGER.warn('stop_by_test is True, but we do not obey it when fit(x_train, y_train, x_test, None)!')
        assert n_groups_train == n_groups_test, 'n_group_train must equal to n_group_test!,' \
                                                ' but {} and {}'.format(n_groups_train, n_groups_test)
        # Initialize the groups
        x_train_group = np.zeros((n_trains, 0), dtype=x_trains[0].dtype)
        x_test_group = np.zeros((n_tests, 0), dtype=x_tests[0].dtype)
        group_starts, group_ends, group_dims = [], [], []
        # train set
        for i, x_train in enumerate(x_trains):
            assert x_train.shape[0] == n_trains, 'x_train.shape[0] = {} not equal to {}'.format(
                x_train.shape[0], n_trains)
            x_train = x_train.reshape(n_trains, -1)
            group_dims.append(x_train.shape[1])
            group_starts.append(i if i == 0 else group_ends[i - 1])
            group_ends.append(group_starts[i] + group_dims[i])
            x_train_group = np.hstack((x_train_group, x_train))
        # test set
        for i, x_test in enumerate(x_tests):
            assert x_test.shape[0] == n_tests
            x_test = x_test.reshape(n_tests, -1)
            assert x_test.shape[1] == group_dims[i]
            x_test_group = np.hstack((x_test_group, x_test))

        self.LOGGER.info('group_starts={}'.format(group_starts))
        self.LOGGER.info('group_dims={}'.format(group_dims))
        self.LOGGER.info('X_train_group={}, X_test_group={}'.format(x_train_group.shape, x_test_group.shape))
        self.group_starts = group_starts
        self.group_ends = group_ends
        self.group_dims = group_dims
        if self.look_index_cycle is None:
            self.look_index_cycle = [[i, ] for i in range(n_groups_train)]
        x_cur_train, x_cur_test = None, None
        x_proba_train = np.zeros((n_trains, 0), dtype=np.float32)
        x_proba_test = np.zeros((n_tests, 0), dtype=np.float32)
        cascade = None  # for save test results
        layer_id = 0
        layer_train_metrics, layer_test_metrics = [], []
        opt_data = [None, None]
        try:
            while True:
                if layer_id >= self.max_layers > 0:
                    break
                x_cur_train = np.zeros((n_trains, 0), dtype=np.float32)
                x_cur_test = np.zeros((n_tests, 0), dtype=np.float32)
                train_ids = self.look_index_cycle[layer_id % n_groups_train]
                for gid in train_ids:
                    x_cur_train = np.hstack((x_cur_train, x_train_group[:, group_starts[gid]:group_ends[gid]]))
                    x_cur_test = np.hstack((x_cur_test, x_test_group[:, group_starts[gid]:group_ends[gid]]))
                x_cur_train = np.hstack((x_cur_train, x_proba_train))
                x_cur_test = np.hstack((x_cur_test, x_proba_test))
                data_save_dir = self.data_save_dir
                if data_save_dir is not None:
                    data_save_dir = osp.join(data_save_dir, 'cascade_layer_{}'.format(layer_id))
                model_save_dir = self.model_save_dir
                if model_save_dir is not None:
                    model_save_dir = osp.join(model_save_dir, 'cascade_layer_{}'.format(layer_id))
                cascade = self._create_cascade_layer(task=self.task,
                                                     est_configs=self.est_configs,
                                                     n_classes=self.n_classes,
                                                     data_save_dir=data_save_dir,
                                                     model_save_dir=model_save_dir,
                                                     layer_id=layer_id,
                                                     keep_in_mem=self.keep_in_mem,
                                                     dtype=self.dtype,
                                                     seed=self.seed)
                x_proba_train, x_proba_test = cascade.fit_transform(x_cur_train, y_train, x_cur_test, y_test)
                if self.keep_in_mem:
                    self.layer_fit_cascades.append(cascade)
                layer_train_metrics.append(cascade.train_avg_metric)
                layer_test_metrics.append(cascade.test_avg_metric)
                # detect best layer id
                if self.stop_by_test:
                    opt_layer_id = get_opt_layer_id(layer_test_metrics, self.larger_better)
                else:
                    opt_layer_id = get_opt_layer_id(layer_train_metrics, self.larger_better)
                self.opt_layer_id = opt_layer_id
                # if this layer is the best layer, set the opt_data
                if opt_layer_id == layer_id:
                    opt_data = [x_cur_train, y_train, x_cur_test, y_test]
                    # detected best layer, save test result
                    if y_test is None and cascade is not None:
                        self.save_test_result(x_proba_test=cascade.eval_proba_test)
                        if self.keep_test_result:
                            self.test_results = cascade.eval_proba_test
                # early stopping
                if layer_id - opt_layer_id >= self.early_stop_rounds > 0:
                    # log and save the final results of the optimal layer
                    if y_test is not None:
                        self.LOGGER.info('[Result][Early Stop][Optimal Layer Detected]'
                                         ' opt_layer={},'.format(opt_layer_id) +
                                         ' {}_train={:.4f}{}, {}_test={:.4f}{}'.format(
                                          self.eval_metrics[0].name, layer_train_metrics[opt_layer_id],
                                          self._percent, self.eval_metrics[0].name, layer_test_metrics[opt_layer_id],
                                          self._percent))
                    else:
                        self.LOGGER.info('[Result][Early Stop][Optimal Layer Detected]'
                                         ' opt_layer={},'.format(opt_layer_id) +
                                         ' {}_train={:.4f}{}'.format(self.eval_metrics[0].name,
                                                                     layer_train_metrics[opt_layer_id],
                                                                     self._percent))
                    self.n_layers = layer_id + 1
                    self.save_data(opt_layer_id, *opt_data)
                    # wash the fit cascades after optimal layer id to save memory
                    if self.keep_in_mem:  # if not keep_in_mem, self.layer_fit_cascades is None originally
                        for li in range(opt_layer_id + 1, layer_id + 1):
                            self.layer_fit_cascades[li] = None
                    return x_cur_train, x_cur_test
                if self.data_save_rounds > 0 and (layer_id + 1) % self.data_save_rounds == 0:
                    self.save_data(layer_id, *opt_data)
                layer_id += 1
            # Max Layer Reached
            opt_data = [x_cur_train, y_train, x_cur_test, y_test]
            # detect best layer id
            if self.stop_by_test:
                opt_layer_id = get_opt_layer_id(layer_test_metrics, larger_better=self.larger_better)
            else:
                opt_layer_id = get_opt_layer_id(layer_train_metrics, self.larger_better)
            self.opt_layer_id = opt_layer_id
            if y_test is not None:
                self.LOGGER.info('[Result][Max Layer Reach] max_layer={}, {}_train={:.4f}{}, {}_test={:.4f}{}'
                                 ' optimal_layer={}, {}_optimal_train={:.4f}{},'
                                 ' {}_optimal_test={:.4f}{}'.format(
                                    self.max_layers, self.eval_metrics[0].name, layer_train_metrics[-1], self._percent,
                                    self.eval_metrics[0].name, layer_test_metrics[-1], self._percent, opt_layer_id,
                                    self.eval_metrics[0].name, layer_train_metrics[opt_layer_id], self._percent,
                                    self.eval_metrics[0].name, layer_test_metrics[opt_layer_id], self._percent))
            else:
                self.LOGGER.info('[Result][Max Layer Reach] max_layer={}, {}_train={:.4f}{},'
                                 ' optimal_layer={}, {}_optimal_train={:.4f}{}'.format(
                                  self.max_layers,
                                  self.eval_metrics[0].name, layer_train_metrics[-1], self._percent, opt_layer_id,
                                  self.eval_metrics[0].name, layer_train_metrics[opt_layer_id], self._percent))
            self.save_data(layer_id, *opt_data)
            self.n_layers = layer_id + 1
            # if y_test is None, we predict x_test and save its predictions
            if y_test is None and cascade is not None:
                self.save_test_result(x_proba_test=cascade.eval_proba_test)
                if self.keep_test_result:
                    self.test_results = cascade.eval_proba_test
            # wash the fit cascades after optimal layer id to save memory
            if self.keep_in_mem:
                for li in range(opt_layer_id + 1, layer_id + 1):
                    self.layer_fit_cascades[li] = None
            return x_cur_train, x_cur_test
        except KeyboardInterrupt:
            pass

    def transform(self, X, y=None):
        """
        Transform inputs X.

        :param X:
        :param y:
        :return:
        """
        if not isinstance(X, (list, tuple)):
            X = [X]
        n_groups = len(X)
        n_examples = len(X[0])
        # Initialize the groups
        x_test_group = np.zeros((n_examples, 0), dtype=X[0].dtype)
        # test set
        for i, x_test in enumerate(X):
            assert x_test.shape[0] == n_examples
            x_test = x_test.reshape(n_examples, -1)
            assert x_test.shape[1] == self.group_dims[i]
            x_test_group = np.hstack((x_test_group, x_test))

        self.LOGGER.info('[transform] group_starts={}'.format(self.group_starts))
        self.LOGGER.info('[transform] group_dims={}'.format(self.group_dims))
        self.LOGGER.info('[transform] X_test_group={}'.format(x_test_group.shape))

        if self.look_index_cycle is None:
            self.look_index_cycle = [[i, ] for i in range(n_groups)]
        x_proba_test = np.zeros((n_examples, 0), dtype=np.float32)
        layer_id = 0
        try:
            while layer_id <= self.opt_layer_id:
                self.LOGGER.info('Transforming layer - {} / {}'.format(layer_id, self.n_layers))
                x_cur_test = np.zeros((n_examples, 0), dtype=np.float32)
                train_ids = self.look_index_cycle[layer_id % n_groups]
                for gid in train_ids:
                    x_cur_test = np.hstack((x_cur_test, x_test_group[:, self.group_starts[gid]:self.group_ends[gid]]))
                x_cur_test = np.hstack((x_cur_test, x_proba_test))
                cascade = self.layer_fit_cascades[layer_id]
                x_proba_test = cascade.transform(x_cur_test)
                layer_id += 1
            return x_proba_test
        except KeyboardInterrupt:
            pass

    def evaluate(self, inputs, labels, eval_metrics=None):
        """
        Evaluate inputs.

        :param inputs:
        :param labels:
        :param eval_metrics:
        :return:
        """
        if eval_metrics is None:
            eval_metrics = [Accuracy('evaluate')]
        if isinstance(labels, (list, tuple)):
            assert len(labels) == 1, 'only support single labels array'
            labels = labels[0]
        predictions = self.predict(inputs)
        for metric in eval_metrics:
            metric.calc(labels, predictions, logger=self.LOGGER)

    def predict_proba(self, X):
        """
        Predict probability of X.

        :param X:
        :return:
        """
        if not isinstance(X, (list, tuple)):
            X = [X]
        x_proba = self.transform(X)
        total_proba = np.zeros((X[0].shape[0], self.n_classes), dtype=np.float32)
        for i in range(len(self.est_configs)):
            total_proba += x_proba[:, i * self.n_classes:i * self.n_classes + self.n_classes]
        return total_proba

    def predict(self, X):
        """
        Predict with inputs X.

        :param X:
        :return:
        """
        total_proba = self.predict_proba(X)
        if self.is_classification:
            return np.argmax(total_proba.reshape((-1, self.n_classes)), axis=1)
        else:
            return total_proba.reshape((-1, self.n_classes))

    def _depack(self, x, depth):
        if depth == 0:
            return [x]
        elif depth == 1:
            return x
        elif depth == 2:
            x_cp = []
            for bottom in x:
                for xj in bottom:
                    x_cp.append(xj)
            return x_cp
        else:
            raise ValueError('_concat failed. depth should be less than 2!')

    def _validate_input(self, x_train, y_train, x_test=None, y_test=None):
        assert x_train is not None and y_train is not None, 'x_train is None or y_train is None'
        train_depth = 0
        if isinstance(x_train, (list, tuple)):
            train_depth = check_list_depth(x_train)
        x_train = self._depack(x_train, train_depth)
        if x_test is not None:
            test_depth = 0
            if isinstance(x_test, (list, tuple)):
                test_depth = check_list_depth(x_test)
            x_test = self._depack(x_test, test_depth)
        # only supports one y_train
        if isinstance(y_train, (list, tuple)) and len(y_train) > 0:
            y_train = y_train[0]
        if y_test is not None and isinstance(y_test, (list, tuple)):
            y_test = y_test[0]
        return x_train, y_train, x_test, y_test

    @property
    def num_layers(self):
        """
        Number of layers.

        :return:
        """
        return self.n_layers

    def _save_data(self, layer_id, x_train, y_train):
        """
        Save the intermediate training data of the layer.

        :param layer_id:
        :param x_train:
        :param y_train:
        :return:
        """
        if self.data_save_dir is None:
            return
        data_path = osp.join(self.data_save_dir, "layer_{}-{}.pkl".format(layer_id, 'train'))
        check_dir(data_path)
        data = {"X": x_train, "y": y_train}
        self.LOGGER.info("Saving Data in {} ... X.shape={}, y.shape={}".format(
            data_path, data["X"].shape, data["y"].shape))
        with open(data_path, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def save_data(self, layer_id, x_train, y_train, x_test, y_test):
        """
        Save the intermediate training data and testing data in this layer.

        :param layer_id:
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        if self.data_save_dir is None:
            return
        for phase in ['train', 'test']:
            data_path = osp.join(self.data_save_dir, "layer_{}-{}.pkl".format(layer_id, phase))
            check_dir(data_path)
            if phase == 'train':
                data = {"X": x_train, "y": y_train}
            else:
                data = {"X": x_test, "y": y_test if y_test is not None else np.zeros((0,))}
            self.LOGGER.info("Saving {} Data in {} ... X.shape={}, y.shape={}".format(
                phase, data_path, data["X"].shape, data["y"].shape))
            with open(data_path, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def save_test_result(self, x_proba_test):
        """
        Save prediction result for testing data without label.

        :param x_proba_test:
        :return:
        """
        if self.data_save_dir is None:
            return
        if x_proba_test is None:
            self.LOGGER.info('x_proba_test is None, DO NOT SAVE!')
            return
        if x_proba_test.shape[1] != self.n_classes:
            self.LOGGER.info('x_proba_test.shape[1] = {} is not equal to n_classes'.format(x_proba_test.shape[1]))
        prefix = datetime.datetime.now().strftime('%m_%d_%H_%M')
        file_name = osp.join(self.data_save_dir, 'submission_' + prefix + '.csv')
        self.LOGGER.info('[Save][Test Output] x_proba_test={}, Saving to {}'.format(x_proba_test.shape, file_name))
        if self.is_classification:
            np.savetxt(file_name, np.argmax(x_proba_test, axis=1), fmt="%d", delimiter=',')
        else:
            np.savetxt(file_name, x_proba_test, fmt="%f", delimiter=',')


def _to_snake_case(name):
    import re
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != '_':
        return insecure
    return 'private' + insecure


def get_opt_layer_id(acc_list, larger_better=True):
    """ Return layer id with max accuracy on training data """
    if larger_better:
        opt_layer_id = np.argsort(-np.asarray(acc_list), kind='mergesort')[0]
    else:
        opt_layer_id = np.argsort(np.asarray(acc_list), kind='mergesort')[0]
    return opt_layer_id
