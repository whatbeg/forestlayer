# -*- coding:utf-8 -*-
"""
Base layers definition
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import random as rnd
import numpy as np
import copy
from ..utils.log_utils import get_logger, list2str
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
    def __init__(self, **kwargs):

        allowed_kwargs = {'input_shape',
                          'batch_size',
                          'dtype',
                          'name'}
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        name = kwargs.get('name')
        if not name:
            prefix = self.__class__.__name__
            name = prefix + "_" + str(rnd.randint)
        self.name = name

        if 'input_shape' in kwargs:
            self.input_shape = kwargs.get('input_shape')
        else:
            self.input_shape = None

        if 'output_shape' in kwargs:
            self.output_shape = kwargs.get('output_shape')
        else:
            self.output_shape = None

        # Set dtype.
        dtype = kwargs.get('dtype')
        if dtype is None:
            dtype = F.floatx()
        self.dtype = dtype
        self.input_layer = None
        self.output_layer = None

    def call(self, inputs, **kwargs):
        return inputs

    def __call__(self, inputs, **kwargs):
        self.call(inputs, **kwargs)

    def fit(self, inputs, labels):
        raise NotImplementedError

    def fit_transform(self, x_trains, y_trains, x_tests=None, y_tests=None):
        """
        Fit and Transform datasets, return two list: train_outputs, test_outputs
        :param x_trains: train datasets
        :param y_trains: train labels
        :param x_tests: test datasets
        :param y_tests: test labels
        :return: train_outputs, test_outputs
        """
        raise NotImplementedError

    def transform(self, inputs, labels=None):
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError

    def evaluate(self, inputs, labels):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class InputLayer(Layer):
    def __init__(self, input_shape=None, batch_size=None, dtype=None, name=None):
        if not name:
            prefix = 'input'
            name = prefix + '_' + str(id(self))
        super(InputLayer, self).__init__(dtype=dtype, name=name)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_est = 0
        self.estimators = []

    def call(self, inputs, **kwargs):
        return inputs

    def __call__(self, inputs, **kwargs):
        self.call(inputs)

    def fit(self, inputs, labels):
        return inputs

    def fit_transform(self, x_trains, y_trains, x_tests=None, y_tests=None):
        return x_trains, x_tests

    def transform(self, inputs, labels=None):
        return inputs

    def predict(self, inputs):
        raise NotImplementedError

    def evaluate(self, inputs, labels):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class MultiGrainScanLayer(Layer):
    def __init__(self, input_shape=None, batch_size=None, dtype=None, name=None,
                 windows=None, est_for_windows=None, n_class=None):
        if not name:
            prefix = 'multi_grain_scan'
            name = prefix + '_' + str(id(self))
        super(MultiGrainScanLayer, self).__init__(dtype=dtype, name=name)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.windows = windows  # [Win, Win, Win, ...]
        self.est_for_windows = est_for_windows  # [[est1, est2], [est1, est2], [est1, est2], ...]
        assert n_class is not None
        self.n_class = n_class

    def call(self, X, **kwargs):
        pass

    def __call__(self, X, **kwargs):
        pass

    def scan(self, window, X):
        return window.fit_transform(X)

    def fit(self, X, y):
        pass

    def fit_transform(self, x_trains, y_trains, x_tests=None, y_tests=None):
        if not isinstance(x_trains, (list, tuple)):
            x_trains = [x_trains]
        if not isinstance(y_trains, (list, tuple)):
            y_trains = [y_trains]
        if not isinstance(x_tests, (list, tuple)) and x_tests is not None:
            x_tests = [x_tests]
        if not isinstance(y_tests, (list, tuple)) and y_tests is not None:
            y_tests = [y_tests]
        if len(x_trains) != 1 or len(y_trains) != 1:
            raise ValueError("Multi grain scan Layer only supports exactly one input now!")
        if len(x_tests) != 1 or len(y_tests) != 1:
            raise ValueError("Multi grain scan Layer only supports exactly one input now!")
        x_tests = [] if x_tests is None else x_tests
        y_tests = [] if y_tests is None else y_tests
        # Construct test sets
        x_wins_train = []
        x_wins_test = []
        for win in self.windows:
            x_wins_train.append(self.scan(win, x_trains[0]))
        for win in self.windows:
            tmp_wins_test = []
            for test_data in x_tests:
                tmp_wins_test.append(self.scan(win, test_data))
            x_wins_test.append(tmp_wins_test)
        # [[win, win], [win, win], ...], len = len(test_sets)
        # test_sets = [('testOfWin{}'.format(i), x, y) for i, x, y in enumerate(zip(x_wins_test, y_tests))]
        LOGGER.info('X_wins of train: {}'.format([win.shape for win in x_wins_train]))
        LOGGER.info('X_wins of tests[0]: {}'.format([win.shape for win in x_wins_test[0]]))
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
            y_win = y_trains[0][:, np.newaxis].repeat(x_wins_train[wi].shape[1], axis=1)
            for ti, ts in enumerate(x_wins_test[wi]):
                x_wins_test[wi][ti] = ts.reshape((ts.shape[0], -1, ts.shape[-1]))
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

    def transform(self, inputs, labels=None):
        pass

    def predict(self, inputs):
        pass

    def evaluate(self, inputs, labels):
        pass

    def __str__(self):
        return self.__class__.__name__


class PoolingLayer(Layer):
    def __init__(self, input_shape=None, batch_size=None, dtype=None, name=None, pools=None):
        super(PoolingLayer, self).__init__(input_shape=input_shape, batch_size=batch_size, dtype=dtype, name=name)
        # [[pool/7x7/est1, pool/7x7/est2], [pool/11x11/est1, pool/11x11/est1], [pool/13x13/est1, pool/13x13/est1], ...]
        self.pools = pools

    def call(self, X, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def fit(self, inputs, labels):
        pass

    def fit_transform(self, x_trains, y_trains, x_tests=None, y_tests=None):
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

    def transform(self, inputs, labels=None):
        pass

    def evaluate(self, inputs, labels):
        pass

    def predict(self, inputs):
        pass


class ConcatLayer(Layer):
    def __init__(self, input_shape=None, batch_size=None, dtype=None, name=None, axis=-1):
        super(ConcatLayer, self).__init__(input_shape=input_shape, batch_size=batch_size, dtype=dtype, name=name)
        # [[pool/7x7/est1, pool/7x7/est2], [pool/11x11/est1, pool/11x11/est1], [pool/13x13/est1, pool/13x13/est1], ...]
        # to
        # [Concat(axis=axis), Concat(axis=axis), Concat(axis=axis), ...]
        self.axis = axis

    def call(self, X, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def transform(self, inputs, labels=None):
        pass

    def evaluate(self, inputs, labels=None):
        pass

    def fit_transform(self, x_trains, y_trains, x_tests=None, y_tests=None):
        # inputs shape: [[(60000, 10, 6, 6), (60000, 10, 6, 6)], [.., ..], ...]
        concat_data = []
        for bottoms in x_trains:
            if self.axis == -1:
                for i, bottom in enumerate(bottoms):
                    bottoms[i] = bottom.reshape((bottom.shape[0], -1))
                concat_res = np.concatenate(bottoms, 1)
            else:
                concat_res = np.concatenate(bottoms, self.axis)
            concat_data.append(concat_res)
        LOGGER.info("concat data shape: {}".format(list2str(concat_data, 1)))
        x_tests = [] if x_tests is None else x_tests
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
        return concat_data, concat_test

    def predict(self, inputs):
        pass

    def fit(self, inputs, labels=None):
        pass





