# -*- coding:utf-8 -*-
"""
Base layers definition
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import numpy as np
import copy
import datetime
from ..utils.log_utils import get_logger, list2str
from ..utils.storage_utils import *
from ..utils.metrics import Accuracy, AUC
from ..estimators import get_estimator_kfold, EstimatorArgument
from .. import backend as F

LOGGER = get_logger('layer')
# LOGGER.setLevel(logging.ERROR)


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
        """
        self.batch_size = batch_size
        if not name:
            prefix = self.__class__.__name__
            name = _to_snake_case(prefix) + "_" + str(id(self))
        self.name = name
        # Set dtype.
        if dtype is None:
            dtype = F.floatx()
        self.dtype = dtype

    def call(self, x_trains):
        raise NotImplementedError

    def __call__(self, x_trains):
        raise NotImplementedError

    def fit(self, x_trains, y_trains):
        """
        Fit datasets, return a list or single ndarray: train_outputs
        NOTE: may change x_trains, y_trains
        :param x_trains: train data
        :param y_trains: train labels
        :return: train_outputs
        """
        raise NotImplementedError

    def fit_transform(self, x_trains, y_trains, x_tests=None, y_tests=None):
        """
        Fit and Transform datasets, return two lists or two single ndarrays: train_outputs, test_outputs
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
    def __init__(self, batch_size=None, dtype=None, name=None,
                 windows=None, est_for_windows=None, n_class=None):
        if not name:
            prefix = 'multi_grain_scan'
            name = prefix + '_' + str(id(self))
        super(MultiGrainScanLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        self.windows = windows  # [Win, Win, Win, ...]
        self.est_for_windows = est_for_windows  # [[est1, est2], [est1, est2], [est1, est2], ...]
        assert n_class is not None
        self.n_class = n_class

    def call(self, x_trains, **kwargs):
        pass

    def __call__(self, x_trains, **kwargs):
        pass

    def scan(self, window, x):
        return window.fit_transform(x)

    def fit(self, x_trains, y_trains):
        if isinstance(x_trains, (list, tuple)):
            assert len(x_trains) == 1, "Multi grain scan Layer only supports exactly one input now!"
            x_trains = x_trains[0]
        if isinstance(y_trains, (list, tuple)):
            assert len(y_trains) == 1, "Multi grain scan Layer only supports exactly one input now!"
            y_trains = y_trains[0]
        # Construct test sets
        x_wins_train = []
        for win in self.windows:
            x_wins_train.append(self.scan(win, x_trains))
        # [[win, win], [win, win], ...], len = len(test_sets)
        LOGGER.info('X_wins of train: {}'.format([win.shape for win in x_wins_train]))
        x_win_est_train = []
        for wi, ests_for_win in enumerate(self.est_for_windows):
            if not isinstance(ests_for_win, (list, tuple)):
                ests_for_win = [ests_for_win]
            win_est_train = []
            # X_wins[wi] = (60000, 11, 11, 49)
            _, nh, nw, _ = x_wins_train[wi].shape
            # (60000, 121, 49)
            x_wins_train[wi] = x_wins_train[wi].reshape((x_wins_train[wi].shape[0], -1, x_wins_train[wi].shape[-1]))
            y_win = y_trains[:, np.newaxis].repeat(x_wins_train[wi].shape[1], axis=1)
            for est in ests_for_win:
                # (60000, 121, 10)
                y_proba_train, y_probas_test = est.fit_transform(x_wins_train[wi], y_win, y_win[:, 0])
                y_proba_train = y_proba_train.reshape((-1, nh, nw, self.n_class)).transpose((0, 3, 1, 2))
                for i in range(len(y_probas_test)):
                    # (60000, 10, 11, 11)
                    y_probas_test[i] = y_probas_test[i].reshape((-1, nh, nw, self.n_class)).transpose((0, 3, 1, 2))
                win_est_train.append(y_proba_train)
            x_win_est_train.append(win_est_train)
        if len(x_win_est_train) == 0:
            return x_wins_train
        LOGGER.info('x_win_est_train.shape: {}'.format(list2str(x_win_est_train, 2)))
        return x_win_est_train

    def fit_transform(self, x_trains, y_trains, x_tests=None, y_tests=None):
        if x_tests is None:
            return self.fit(x_trains, y_trains), None
        if isinstance(x_trains, (list, tuple)):
            assert len(x_trains) == 1, "Multi grain scan Layer only supports exactly one input now!"
            x_trains = x_trains[0]
        if isinstance(y_trains, (list, tuple)):
            assert len(y_trains) == 1, "Multi grain scan Layer only supports exactly one input now!"
            y_trains = y_trains[0]
        if x_tests is not None and not isinstance(x_tests, (list, tuple)):
            x_tests = [x_tests]
        if y_tests is not None and not isinstance(y_tests, (list, tuple)):
            y_tests = [y_tests]
        # Construct test sets
        x_wins_train = []
        x_wins_test = []
        for win in self.windows:
            x_wins_train.append(self.scan(win, x_trains))
        for win in self.windows:
            tmp_wins_test = []
            for test_data in x_tests:
                tmp_wins_test.append(self.scan(win, test_data))
            x_wins_test.append(tmp_wins_test)
        # [[win, win], [win, win], ...], len = len(test_sets)
        # test_sets = [('testOfWin{}'.format(i), x, y) for i, x, y in enumerate(zip(x_wins_test, y_tests))]
        LOGGER.info('X_wins of train: {}'.format([win.shape for win in x_wins_train]))
        log_info = 'X_wins of tests: '
        for i in range(len(x_wins_test)):
            log_info += '{}, '.format([win.shape for win in x_wins_test[i]])
        LOGGER.info(log_info)
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
            y_win = y_trains[:, np.newaxis].repeat(x_wins_train[wi].shape[1], axis=1)
            for ti, ts in enumerate(x_wins_test[wi]):
                x_wins_test[wi][ti] = ts.reshape((ts.shape[0], -1, ts.shape[-1]))
            if y_tests is None:
                tmp_y_tests = [None for _ in range(len(x_wins_test))]
            else:
                tmp_y_tests = copy.deepcopy(y_tests)
                for i, y_test in enumerate(y_tests):
                    tmp_y_tests[i] = y_test[:, np.newaxis].repeat(x_wins_test[wi][i].shape[1], axis=1)
            test_sets = [('testOfWin{}'.format(wi), tw, tmp_y_tests[i]) for i, tw in enumerate(x_wins_test[wi])]
            # print(y_win.shape)
            # for k in test_sets:
            #     print(k[0], k[1].shape, k[2].shape)
            for est in ests_for_win:
                # (60000, 121, 10)
                y_proba_train, y_probas_test = est.fit_transform(x_wins_train[wi], y_win, y_win[:, 0], test_sets)
                y_proba_train = y_proba_train.reshape((-1, nh, nw, self.n_class)).transpose((0, 3, 1, 2))
                for i in range(len(y_probas_test)):
                    # (60000, 10, 11, 11)
                    y_probas_test[i] = y_probas_test[i].reshape((-1, nh, nw, self.n_class)).transpose((0, 3, 1, 2))
                win_est_train.append(y_proba_train)
                win_est_test.append(y_probas_test)
            x_win_est_train.append(win_est_train)
            x_win_est_test.append(win_est_test)
        if len(x_win_est_train) == 0:
            return x_wins_train, x_wins_test
        LOGGER.info('x_win_est_train.shape: {}'.format(list2str(x_win_est_train, 2)))
        if len(x_win_est_test) > 0:
            LOGGER.info('x_win_est_test[0].shape: {}'.format(list2str(x_win_est_test[0], 2)))
        if len(x_win_est_test) > 1:
            LOGGER.info('x_win_est_test[1].shape: {}'.format(list2str(x_win_est_test[1], 2)))
        return x_win_est_train, x_win_est_test

    def transform(self, x_trains):
        assert x_trains is not None, 'x_trains should not be None!'
        if isinstance(x_trains, (list, tuple)):
            assert len(x_trains) == 1, "Multi grain scan Layer only supports exactly one input now!"
            x_trains = x_trains[0]
        # Construct test sets
        x_wins_train = []
        for win in self.windows:
            x_wins_train.append(self.scan(win, x_trains))
        # [[win, win], [win, win], ...], len = len(test_sets)
        LOGGER.info('X_wins of train: {}'.format([win.shape for win in x_wins_train]))
        x_win_est_train = []
        for wi, ests_for_win in enumerate(self.est_for_windows):
            if not isinstance(ests_for_win, (list, tuple)):
                ests_for_win = [ests_for_win]
            win_est_train = []
            # X_wins[wi] = (60000, 11, 11, 49)
            _, nh, nw, _ = x_wins_train[wi].shape
            # (60000, 121, 49)
            x_wins_train[wi] = x_wins_train[wi].reshape((x_wins_train[wi].shape[0], -1, x_wins_train[wi].shape[-1]))
            for est in ests_for_win:
                # (60000, 121, 10)
                y_proba_train = est.transform(x_wins_train[wi])
                y_proba_train = y_proba_train.reshape((-1, nh, nw, self.n_class)).transpose((0, 3, 1, 2))
                win_est_train.append(y_proba_train)
            x_win_est_train.append(win_est_train)
        if len(x_win_est_train) == 0:
            return x_wins_train
        LOGGER.info('[transform] win_est_train.shape: {}'.format(list2str(x_win_est_train, 2)))
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
    def __init__(self, batch_size=None, dtype=None, name=None, pools=None):
        super(PoolingLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        # [[pool/7x7/est1, pool/7x7/est2], [pool/11x11/est1, pool/11x11/est1], [pool/13x13/est1, pool/13x13/est1], ...]
        self.pools = pools

    def call(self, x_trains, **kwargs):
        pass

    def __call__(self, x_trains):
        pass

    def fit(self, x_trains, y_trains=None):
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
        LOGGER.info('x_trains pooled: {}'.format(list2str(x_trains, 2)))
        return x_trains

    def fit_transform(self, x_trains, y_trains=None, x_tests=None, y_tests=None):
        if x_tests is None:
            return self.fit(x_trains, y_trains), None
        # inputs shape: [[(60000, 10, 11, 11), (60000, 10, 11, 11)], [.., ..], ...]
        if len(self.pools) != len(x_trains):
            raise ValueError('len(pools) does not equal to len(inputs), you must set right pools!')
        x_tests = x_tests if x_tests is not None else []
        for pi, pool in enumerate(self.pools):
            if not isinstance(pool, (list, tuple)):
                pool = [pool]
            if len(pool) != len(x_trains[pi]):
                raise ValueError('len(pools[{}]) does not equal to'
                                 ' len(inputs[{}]), you must set right pools!'.format(pi, pi))
            for pj, pl in enumerate(pool):
                x_trains[pi][pj] = pl.fit_transform(x_trains[pi][pj])
                for ti, ts in enumerate(x_tests[pi][pj]):
                    x_tests[pi][pj][ti] = pl.fit_transform(ts)
        LOGGER.info('x_trains pooled: {}'.format(list2str(x_trains, 2)))
        if len(x_tests) > 0:
            LOGGER.info('x_tests pooled: {}'.format(list2str(x_tests, 3)))
        return x_trains, x_tests

    def transform(self, x_trains):
        assert x_trains is not None, 'x_trains should not be None!'
        if not isinstance(x_trains, (list, tuple)):
            x_trains = [x_trains]
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
                x_trains[pi][pj] = pl.transform(x_trains[pi][pj])
        LOGGER.info('[transform] x_trains pooled: {}'.format(list2str(x_trains, 2)))
        return x_trains

    def evaluate(self, inputs, labels):
        raise NotImplementedError

    def predict(self, X):
        return self.transform(X)

    def predict_proba(self, X):
        return self.transform(X)


class ConcatLayer(Layer):
    def __init__(self, batch_size=None, dtype=None, name=None, axis=-1):
        super(ConcatLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        # [[pool/7x7/est1, pool/7x7/est2], [pool/11x11/est1, pool/11x11/est1], [pool/13x13/est1, pool/13x13/est1], ...]
        # to
        # [Concat(axis=axis), Concat(axis=axis), Concat(axis=axis), ...]
        self.axis = axis

    def call(self, X, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def _fit(self, x_trains):
        if not isinstance(x_trains, (list, tuple)):
            x_trains = [x_trains]
        # inputs shape: [[(60000, 10, 6, 6), (60000, 10, 6, 6)], [.., ..], ...]
        concat_train = []
        for bottoms in x_trains:
            if self.axis == -1:
                for i, bottom in enumerate(bottoms):
                    bottoms[i] = bottom.reshape((bottom.shape[0], -1))
                concat_res = np.concatenate(bottoms, 1)
            else:
                concat_res = np.concatenate(bottoms, self.axis)
            concat_train.append(concat_res)
        return concat_train

    def transform(self, x_trains):
        concat_train = self._fit(x_trains)
        LOGGER.info("[transform] concat train shape: {}".format(list2str(concat_train, 1)))
        return concat_train

    def evaluate(self, inputs, labels=None):
        raise NotImplementedError

    def fit(self, x_trains, y_trains=None):
        concat_train = self._fit(x_trains)
        LOGGER.info("concat train shape: {}".format(list2str(concat_train, 1)))
        return concat_train

    def fit_transform(self, x_trains, y_trains, x_tests=None, y_tests=None):
        if x_tests is None:
            return self.fit(x_trains, y_trains), None
        # inputs shape: [[(60000, 10, 6, 6), (60000, 10, 6, 6)], [.., ..], ...]
        concat_train = []
        for bottoms in x_trains:
            if self.axis == -1:
                for i, bottom in enumerate(bottoms):
                    bottoms[i] = bottom.reshape((bottom.shape[0], -1))
                concat_res = np.concatenate(bottoms, 1)
            else:
                concat_res = np.concatenate(bottoms, self.axis)
            concat_train.append(concat_res)
        LOGGER.info("concat train shape: {}".format(list2str(concat_train, 1)))
        # x_tests = [] if x_tests is None else x_tests
        if len(x_tests) > 0 and len(x_tests[0][0]) != 1:
            raise ValueError("Now Concat Layer only supports one test_data in test_set")
        for bottoms in x_tests:
            for i, bot in enumerate(bottoms):
                bottoms[i] = bot[0]
        concat_test = []
        for bottoms in x_tests:
            if self.axis == -1:
                for i, bottom in enumerate(bottoms):
                    bottoms[i] = bottom.reshape((bottom.shape[0], -1))
                concat_res = np.concatenate(bottoms, 1)
            else:
                concat_res = np.concatenate(bottoms, self.axis)
            concat_test.append(concat_res)
        LOGGER.info("concat test data shape: {}".format(list2str(concat_test, 1)))
        return concat_train, concat_test

    def predict(self, X):
        return self.transform(X)

    def predict_proba(self, X):
        return self.transform(X)


class CascadeLayer(Layer):
    def __init__(self, batch_size=None, dtype=None, name=None, est_configs=None, layer_id='anonymous', n_classes=None,
                 keep_in_mem=False, data_save_dir=None, metrics=None, seed=None):
        """Cascade Layer.
        A cascade layer contains several estimators, it accepts single input, go through these estimators, produces
        predicted probability by every estimators, and stacks them together for next cascade layer.

        :param batch_size: cascade layer do not need batch_size actually.
        :param dtype: data type
        :param name: name of this layer
        :param est_configs: list of estimator arguments, every argument can be `dict` or `EstimatorArgument` instance
                            identify the estimator configuration to construct at this layer
        :param layer_id: layer id, if this layer is an independent layer, layer id is anonymous [default]
        :param n_classes: number of classes to classify
        :param keep_in_mem: identifies whether keep the model in memory, if fit_transform,
                            we recommend set it False to save memory and speed up the application
                            TODO: support dump model to disk to save memory
        :param data_save_dir: directory to save intermediate data into
        :param metrics: evaluation metrics used in training model and evaluating testing data
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
                self.est_configs[eci] = est_config.get_est_args()
        self.layer_id = layer_id
        if not name:
            name = 'layer-{}'.format(self.layer_id)
        super(CascadeLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        self.n_classes = n_classes
        self.keep_in_mem = keep_in_mem
        self.data_save_dir = data_save_dir
        check_dir(self.data_save_dir)  # check dir, if not exists, create the dir
        self.seed = seed
        self.larger_better = True
        self.metrics = metrics
        if self.metrics == 'accuracy':
            self.eval_metrics = [Accuracy('accuracy')]
        elif self.metrics == 'auc':
            self.eval_metrics = [AUC('auc')]
        else:
            self.eval_metrics = [Accuracy('accuracy')]
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

    def fit(self, x_train, y_train):
        """
        Fit and Transform datasets, return one numpy ndarray: train_output
        NOTE: Only one train set and one test set.
        :param x_train: train datasets
        :param y_train: train labels
        :return: train_output
        """
        if isinstance(x_train, (list, tuple)):
            x_train = None if len(x_train) == 0 else x_train[0]
        if isinstance(y_train, (list, tuple)):
            y_train = None if len(y_train) == 0 else y_train[0]
        assert x_train is not None and y_train is not None, 'x_train is None or y_train is None'
        assert x_train.shape[0] == y_train.shape[0], 'x_train.shape[0] = {} not equal to y_train.shape[0] = {}'.format(
            x_train.shape[0], y_train.shape[0]
        )
        LOGGER.info('X_train.shape={},y_train.shape={}'.format(x_train.shape, y_train.shape))
        n_trains = x_train.shape[0]
        n_classes = self.n_classes
        if n_classes is None:
            n_classes = np.unique(y_train)
        x_proba_train = np.zeros((n_trains, n_classes * self.n_estimators), dtype=np.float32)
        eval_proba_train = np.zeros((n_trains, n_classes))
        # fit estimators, get probas
        for ei in range(self.n_estimators):
            est = self._init_estimators(self.layer_id, ei)
            # fit and transform
            y_proba_train, _ = est.fit_transform(x_train, y_train, y_train, test_sets=None)
            # print(y_proba_train.shape, y_proba_test.shape)
            if y_proba_train is None:
                raise RuntimeError("layer - {} - estimator - {} fit FAILED!".format(self.layer_id, ei))
            if y_proba_train.shape != (n_trains, n_classes):
                raise ValueError('output probability shape incorrect!,'
                                 ' should be {}, but {}'.format((n_trains, n_classes), y_proba_train.shape))
            if self.keep_in_mem:
                self.fit_estimators[ei] = est
            x_proba_train[:, ei * n_classes:ei * n_classes + n_classes] = y_proba_train
            eval_proba_train += y_proba_train
        eval_proba_train /= self.n_estimators
        # now supports one eval_metrics
        metric = self.eval_metrics[0]
        train_avg_acc = metric.calc(y_train, np.argmax(eval_proba_train, axis=1),
                                    'layer - {} - [train] average'.format(self.layer_id))
        self.train_avg_metric = train_avg_acc
        return x_proba_train

    def fit_transform(self, x_train, y_train, x_test=None, y_test=None):
        """
        Fit and Transform datasets, return two numpy ndarray: train_output, test_output
        NOTE: Only one train set and one test set.
        if x_test is None, we invoke _fit_transform to get one numpy ndarray: train_output
        :param x_train: train datasets
        :param y_train: train labels
        :param x_test: test datasets
        :param y_test: test labels, can be None
        :return: train_output, test_output
        """
        if x_test is None:
            return self.fit(x_train, y_train), None
        if isinstance(x_train, (list, tuple)):
            x_train = None if len(x_train) == 0 else x_train[0]
        if isinstance(y_train, (list, tuple)):
            y_train = None if len(y_train) == 0 else y_train[0]
        if isinstance(x_test, (list, tuple)):
            x_test = None if len(x_test) == 0 else x_test[0]
        if isinstance(y_test, (list, tuple)):
            y_test = None if len(y_test) == 0 else y_test[0]
        if y_test is None:
            y_test_shape = (0,)
        else:
            y_test_shape = y_test.shape
        LOGGER.info('X_train.shape={},y_train.shape={} / X_test.shape={},y_test.shape={}'.format(
            x_train.shape, y_train.shape, x_test.shape, y_test_shape
        ))
        n_trains = x_train.shape[0]
        n_tests = x_test.shape[0]
        n_classes = self.n_classes
        if n_classes is None:
            n_classes = np.unique(y_train)
        x_proba_train = np.zeros((n_trains, n_classes * self.n_estimators), dtype=np.float32)
        x_proba_test = np.zeros((n_tests, n_classes * self.n_estimators), dtype=np.float32)
        eval_proba_train = np.zeros((n_trains, n_classes))
        eval_proba_test = np.zeros((n_tests, n_classes))
        # fit estimators, get probas
        for ei in range(self.n_estimators):
            est = self._init_estimators(self.layer_id, ei)
            # fit and transform
            y_proba_train, y_proba_test = est.fit_transform(x_train, y_train, y_train,
                                                            test_sets=[('test', x_test, y_test)])
            # if only one element on test_sets, return one test result like y_proba_train
            if isinstance(y_proba_test, (list, tuple)) and len(y_proba_test) == 1:
                y_proba_test = y_proba_test[0]
            # print(y_proba_train.shape, y_proba_test.shape)
            if y_proba_train is None:
                raise RuntimeError("layer - {} - estimator - {} fit FAILED!".format(self.layer_id, ei))
            if y_proba_train.shape != (n_trains, n_classes):
                raise ValueError('output probability shape incorrect!,'
                                 ' should be {}, but {}'.format((n_trains, n_classes), y_proba_train.shape))
            if y_proba_test.shape != (n_tests, n_classes):
                raise ValueError('output probability shape incorrect!'
                                 ' should be {}, but {}'.format((n_trains, n_classes), y_proba_train.shape))
            if self.keep_in_mem:
                self.fit_estimators[ei] = est
            x_proba_train[:, ei*n_classes:ei*n_classes + n_classes] = y_proba_train
            x_proba_test[:, ei*n_classes:ei*n_classes + n_classes] = y_proba_test
            eval_proba_train += y_proba_train
            eval_proba_test += y_proba_test
        eval_proba_train /= self.n_estimators
        eval_proba_test /= self.n_estimators
        metric = self.eval_metrics[0]
        train_avg_metric = metric.calc(y_train, np.argmax(eval_proba_train, axis=1),
                                       'layer - {} - [train] average'.format(self.layer_id), logger=LOGGER)
        self.train_avg_metric = train_avg_metric
        # judge whether y_test is None, which means users are to predict test probas
        if y_test is not None:
            test_avg_metric = metric.calc(y_test, np.argmax(eval_proba_test, axis=1),
                                          'layer - {} - [test] average'.format(self.layer_id), logger=LOGGER)
            self.test_avg_metric = test_avg_metric
        # if y_test is None, we need to generate test prediction, so keep eval_proba_test
        if y_test is None:
            self.eval_proba_test = eval_proba_test
        return x_proba_train, x_proba_test

    @property
    def n_estimators(self):
        """
        Number of estimators of this layer
        :return:
        """
        return len(self.est_configs)

    def _init_estimators(self, layer_id, est_id):
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
                                   est_type=est_type,
                                   eval_metrics=self.eval_metrics,
                                   seed=seed,
                                   keep_in_mem=self.keep_in_mem,
                                   est_args=est_args)

    def transform(self, X):
        """
        Transform datasets, return one numpy ndarray
        NOTE: Only one train set and one test set.
        :param X: train datasets
        :return:
        """
        if isinstance(X, (list, tuple)):
            X = None if len(X) == 0 else X[0]
        n_trains = X.shape[0]
        n_classes = self.n_classes
        x_proba_train = np.zeros((n_trains, n_classes * self.n_estimators), dtype=np.float32)
        # fit estimators, get probas
        for ei, est in enumerate(self.fit_estimators):
            # transform by n-folds CV
            y_proba_train = est.transform(X)
            if y_proba_train is None:
                raise RuntimeError("layer - {} - estimator - {} transform FAILED!".format(self.layer_id, ei))
            if y_proba_train.shape != (n_trains, n_classes):
                raise ValueError('transform output probability shape incorrect!,'
                                 ' should be {}, but {}'.format((n_trains, n_classes), y_proba_train.shape))
            x_proba_train[:, ei * n_classes:ei * n_classes + n_classes] = y_proba_train
        return x_proba_train

    def predict(self, X):
        proba_sum = self.predict_proba(X)
        n_classes = self.n_classes
        return np.argmax(proba_sum.reshape((-1, n_classes)), axis=1)

    def predict_proba(self, X):
        """
        Transform datasets, return one numpy ndarray
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
            if y_proba_train.shape != (n_trains, n_classes):
                raise ValueError('transform output probability shape incorrect!,'
                                 ' should be {}, but {}'.format((n_trains, n_classes), y_proba_train.shape))
            proba_sum += y_proba_train
        return proba_sum

    def evaluate(self, X, y, eval_metrics=None):
        if eval_metrics is None:
            eval_metrics = [Accuracy('evaluate')]
        if isinstance(y, (list, tuple)):
            assert len(y) == 1, 'only support single labels array'
            y = y[0]
        pred = self.predict(X)
        for metric in eval_metrics:
            metric.calc(y, pred, logger=LOGGER)


class AutoGrowingCascadeLayer(Layer):
    def __init__(self, batch_size=None, dtype=np.float32, name=None, est_configs=None,
                 early_stopping_rounds=None, max_layers=0, look_index_cycle=None, data_save_rounds=0,
                 stop_by_test=False, n_classes=None, keep_in_mem=False, data_save_dir=None, metrics=None, seed=None):
        """AutoGrowingCascadeLayer
        An AutoGrowingCascadeLayer is a virtual layer that consists of many single cascade layers.
        `auto-growing` means this kind of layer can decide the depth of cascade forest,
         by training error or testing error.

        :param batch_size: cascade layer do not need batch_size actually.
        :param dtype: data type
        :param name: name of this layer
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
        :param metrics: evaluation metrics used in training model and evaluating testing data
        :param seed: random seed, also called random state in scikit-learn random forest
        """
        self.est_configs = [] if est_configs is None else est_configs
        super(AutoGrowingCascadeLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        self.name = name
        self.early_stop_rounds = early_stopping_rounds
        self.max_layers = max_layers
        self.n_classes = n_classes
        # if look_index_cycle is None, you need set look_index_cycle in fit / fit_transform
        self.look_index_cycle = look_index_cycle
        self.data_save_rounds = data_save_rounds
        self.data_save_dir = data_save_dir
        check_dir(self.data_save_dir)  # check data save dir, if not exists, create the dir
        self.keep_in_mem = keep_in_mem
        self.stop_by_test = stop_by_test
        self.metrics = metrics
        if self.metrics == 'accuracy':
            self.eval_metrics = [Accuracy('accuracy')]
        elif self.metrics == 'auc':
            self.eval_metrics = [AUC('auc')]
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

    def _create_cascade_layer(self, est_configs=None, n_classes=None, data_save_dir=None,
                              layer_id=None, keep_in_mem=False, dtype=None, metrics=None, seed=None):
        return CascadeLayer(dtype=dtype, est_configs=est_configs, layer_id=layer_id, n_classes=n_classes,
                            keep_in_mem=keep_in_mem, data_save_dir=data_save_dir, metrics=metrics, seed=seed)

    def call(self, x_trains):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def add(self, est):
        if isinstance(est, EstimatorArgument):
            self.est_configs.append(est.get_est_args())
        elif isinstance(est, dict):
            self.est_configs.append(est)
        else:
            raise ValueError("Unknown estimator information {}".format(est))

    def fit(self, x_trains, y_train):
        if not isinstance(x_trains, (list, tuple)):
            x_trains = [x_trains]
        # only supports one y_train
        if isinstance(y_train, (list, tuple)):
            y_train = y_train[0]
        if self.stop_by_test is True:
            LOGGER.warn('stop_by_test is True, but we do not obey it when fit(x_train, y_train)!')
        self.layer_fit_cascades = []
        n_groups_train = len(x_trains)
        self.n_group_train = n_groups_train
        n_trains = len(y_train)
        # Initialize the groups
        x_train_group = np.zeros((n_trains, 0), dtype=x_trains[0].dtype)
        group_starts, group_ends, group_dims = [], [], []
        # train set
        for i, x_train in enumerate(x_trains):
            assert x_train.shape[0] == n_trains, 'x_train.shape[0]={} not equal to n_trains={}'.format(
                x_train.shape[0], n_trains
            )
            x_train = x_train.reshape(n_trains, -1)
            group_dims.append(x_train.shape[1])
            group_starts.append(i if i == 0 else group_ends[i - 1])
            group_ends.append(group_starts[i] + group_dims[i])
            x_train_group = np.hstack((x_train_group, x_train))

        LOGGER.info('group_starts={}'.format(group_starts))
        LOGGER.info('group_dims={}'.format(group_dims))
        LOGGER.info('X_train_group={}'.format(x_train_group.shape))
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
                cascade = self._create_cascade_layer(est_configs=self.est_configs,
                                                     n_classes=self.n_classes,
                                                     data_save_dir=osp.join(self.data_save_dir,
                                                                            'cascade_layer_{}'.format(layer_id)),
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
                opt_layer_id = get_opt_layer_id(layer_metric_list)
                self.opt_layer_id = opt_layer_id
                # if this layer is the best layer, set the opt_data
                if opt_layer_id == layer_id:
                    opt_data = [x_cur_train, y_train]
                # early stopping
                if layer_id - opt_layer_id >= self.early_stop_rounds > 0:
                    # log and save the final results of the optimal layer
                    LOGGER.info('[Result][Early Stop][Optimal Layer Detected] opt_layer={},'
                                ' accuracy_train={:.2f}%,'.format(opt_layer_id,
                                                                  layer_metric_list[opt_layer_id]))
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
            opt_layer_id = get_opt_layer_id(layer_metric_list)
            self.opt_layer_id = opt_layer_id
            LOGGER.info('[Result][Max Layer Reach] max_layer={}, accuracy_train={:.2f}%,'
                        ' optimal_layer={}, accuracy_optimal_train={}'.format(self.max_layers,
                                                                              layer_metric_list[-1],
                                                                              opt_layer_id,
                                                                              layer_metric_list[opt_layer_id]))
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
        if not isinstance(x_trains, (list, tuple)):
            x_trains = [x_trains]
        # only supports one y_train
        if isinstance(y_train, (list, tuple)):
            y_train = y_train[0]
        if y_test is not None and isinstance(y_test, (list, tuple)):
            y_test = y_test[0]
        if not isinstance(x_tests, (list, tuple)):
            x_tests = [x_tests]
        self.layer_fit_cascades = []
        n_groups_train = len(x_trains)
        self.n_group_train = n_groups_train
        n_groups_test = len(x_tests)
        n_trains = len(y_train)
        n_tests = x_tests[0].shape[0]
        if y_test is None and self.stop_by_test is True:
            raise ValueError("Since y_test is None so you cannot set stop_by_test True!")
        assert n_groups_train == n_groups_test, 'n_group_train must equal to n_group_test!'
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

        LOGGER.info('group_starts={}'.format(group_starts))
        LOGGER.info('group_dims={}'.format(group_dims))
        LOGGER.info('X_train_group={}, X_test_group={}'.format(x_train_group.shape, x_test_group.shape))
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
                cascade = self._create_cascade_layer(est_configs=self.est_configs,
                                                     n_classes=self.n_classes,
                                                     data_save_dir=osp.join(self.data_save_dir,
                                                                            'cascade_layer_{}'.format(layer_id)),
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
                    opt_layer_id = get_opt_layer_id(layer_test_metrics)
                else:
                    opt_layer_id = get_opt_layer_id(layer_train_metrics)
                self.opt_layer_id = opt_layer_id
                # if this layer is the best layer, set the opt_data
                if opt_layer_id == layer_id:
                    opt_data = [x_cur_train, y_train, x_cur_test, y_test]
                    # detected best layer, save test result
                    if y_test is None:
                        self.save_test_result(x_proba_test=cascade.eval_proba_test)
                # early stopping
                if layer_id - opt_layer_id >= self.early_stop_rounds > 0:
                    # log and save the final results of the optimal layer
                    if y_test is not None:
                        LOGGER.info('[Result][Early Stop][Optimal Layer Detected] opt_layer={},'
                                    ' accuracy_train={:.2f}%, accuracy_test={:.2f}%'.format(
                                      opt_layer_id, layer_train_metrics[opt_layer_id],
                                      layer_test_metrics[opt_layer_id]))
                    else:
                        LOGGER.info('[Result][Early Stop][Optimal Layer Detected] opt_layer={},'
                                    ' accuracy_train={:.2f}%'.format(
                                      opt_layer_id, layer_train_metrics[opt_layer_id]))
                    self.n_layers = layer_id + 1
                    self.save_data(opt_layer_id, *opt_data)
                    # wash the fit cascades after optimal layer id to save memory
                    if self.keep_in_mem:
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
                opt_layer_id = get_opt_layer_id(layer_test_metrics)
            else:
                opt_layer_id = get_opt_layer_id(layer_train_metrics)
            self.opt_layer_id = opt_layer_id
            if y_test is not None:
                LOGGER.info('[Result][Max Layer Reach] max_layer={}, accuracy_train={:.2f}%, accuracy_test={:.2f}%'
                            ' optimal_layer={}, accuracy_optimal_train={:.2f}%,'
                            ' accuracy_optimal_test={:.2f}%'.format(self.max_layers,
                                                                    layer_train_metrics[-1],
                                                                    layer_test_metrics[-1],
                                                                    opt_layer_id,
                                                                    layer_train_metrics[opt_layer_id],
                                                                    layer_test_metrics[opt_layer_id]))
            else:
                LOGGER.info('[Result][Max Layer Reach] max_layer={}, accuracy_train={:.2f}%,'
                            ' accuracy_test={:.2f}% optimal_layer={},'
                            ' accuracy_optimal_train={:.2f}%,'.format(self.max_layers,
                                                                      layer_train_metrics[-1],
                                                                      layer_test_metrics[-1],
                                                                      opt_layer_id,
                                                                      layer_train_metrics[opt_layer_id],
                                                                      layer_test_metrics[opt_layer_id]))
            self._save_data(layer_id, *opt_data)
            self.n_layers = layer_id + 1
            # if y_test is None, we predict x_test and save its predictions
            if y_test is None and cascade is not None:
                self.save_test_result(x_proba_test=cascade.eval_proba_test)
            # wash the fit cascades after optimal layer id to save memory
            if self.keep_in_mem:
                for li in range(opt_layer_id + 1, layer_id + 1):
                    self.layer_fit_cascades[li] = None
            return x_cur_train, x_cur_test
        except KeyboardInterrupt:
            pass

    def transform(self, X, y=None):
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

        LOGGER.info('[transform] group_starts={}'.format(self.group_starts))
        LOGGER.info('[transform] group_dims={}'.format(self.group_dims))
        LOGGER.info('[transform] X_test_group={}'.format(x_test_group.shape))

        if self.look_index_cycle is None:
            self.look_index_cycle = [[i, ] for i in range(n_groups)]
        x_proba_test = np.zeros((n_examples, 0), dtype=np.float32)
        layer_id = 0
        try:
            while layer_id <= self.opt_layer_id:
                LOGGER.info('Transforming layer - {} / {}'.format(layer_id, self.n_layers))
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
        if eval_metrics is None:
            eval_metrics = [Accuracy('evaluate')]
        if isinstance(labels, (list, tuple)):
            assert len(labels) == 1, 'only support single labels array'
            labels = labels[0]
        pred = self.predict(inputs)
        for metric in eval_metrics:
            metric.calc(labels, pred, logger=LOGGER)

    def predict_proba(self, X):
        if not isinstance(X, (list, tuple)):
            X = [X]
        x_proba = self.transform(X)
        total_proba = np.zeros((X[0].shape[0], self.n_classes), dtype=np.float32)
        for i in range(len(self.est_configs)):
            total_proba += x_proba[:, i * self.n_classes:i * self.n_classes + self.n_classes]
        return total_proba

    def predict(self, X):
        total_proba = self.predict_proba(X)
        return np.argmax(total_proba.reshape((-1, self.n_classes)), axis=1)

    @property
    def num_layers(self):
        return self.n_layers

    def _save_data(self, layer_id, x_train, y_train):
        if self.data_save_dir is None:
            return
        data_path = osp.join(self.data_save_dir, "layer_{}-{}.pkl".format(layer_id, 'train'))
        check_dir(data_path)
        data = {"X": x_train, "y": y_train}
        LOGGER.info("Saving Data in {} ... X.shape={}, y.shape={}".format(
            data_path, data["X"].shape, data["y"].shape))
        with open(data_path, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def save_data(self, layer_id, x_train, y_train, x_test, y_test):
        if self.data_save_dir is None:
            return
        for phase in ['train', 'test']:
            data_path = osp.join(self.data_save_dir, "layer_{}-{}.pkl".format(layer_id, phase))
            check_dir(data_path)
            if phase == 'train':
                data = {"X": x_train, "y": y_train}
            else:
                data = {"X": x_test, "y": y_test if y_test is not None else np.zeros((0,))}
            LOGGER.info("Saving {} Data in {} ... X.shape={}, y.shape={}".format(
                phase, data_path, data["X"].shape, data["y"].shape))
            with open(data_path, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def save_test_result(self, x_proba_test):
        """
        Save prediction result for Test data without label.
        """
        if self.data_save_dir is None:
            return
        if x_proba_test is None:
            LOGGER.info('x_proba_test is None, DO NOT SAVE!')
            return
        if x_proba_test.shape[1] != self.n_classes:
            LOGGER.info('x_proba_test.shape[1] = {} is not equal to n_classes'.format(x_proba_test.shape[1]))
        prefix = datetime.datetime.now().strftime('%m_%d_%H_%M')
        file_name = osp.join(self.data_save_dir, 'submission_' + prefix + '.csv')
        LOGGER.info('[Save][Test Output] x_proba_test={}, Saving to {}'.format(x_proba_test.shape, file_name))
        np.savetxt(file_name, np.argmax(x_proba_test, axis=1), fmt="%d", delimiter=',')


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
