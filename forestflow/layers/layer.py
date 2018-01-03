# -*- coding:utf-8 -*-
"""
Base layers definition
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import random as rnd
import numpy as np
from ..utils.log_utils import get_logger
from .. import backend as F

LOGGER = get_logger('layer')


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

    def fit_transform(self, inputs, labels):
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

    def fit_transform(self, inputs, labels):
        return inputs

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
                 windows=None, est_for_windows=None, pools=None, n_class=None):
        if not name:
            prefix = 'multi_grain_scan'
            name = prefix + '_' + str(id(self))
        super(MultiGrainScanLayer, self).__init__(dtype=dtype, name=name)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.windows = windows  # [Win, Win, Win, ...]
        self.est_for_windows = est_for_windows  # [[est1, est2], [est1, est2], [est1, est2], ...]
        # [[pool/7x7/est1, pool/7x7/est2], [pool/11x11/est1, pool/11x11/est1], [pool/13x13/est1, pool/13x13/est1], ...]
        self.poolings = pools
        assert n_class is not None
        self.n_class = n_class

    def call(self, X, **kwargs):
        pass

    def __call__(self, X, **kwargs):
        pass

    def scan(self, window):
        return window.fit_transform()

    def fit(self, X, y):
        pass

    def fit_transform(self, x_train, y_train, x_test=None, y_test=None):
        X_wins = []
        for win in self.windows:
            X_wins.append(self.scan(win))
        LOGGER.info('X_wins: {}'.format(win.shape for win in X_wins))
        X_win_ests = []
        for wi, ests_for_win in enumerate(self.est_for_windows):
            if not isinstance(ests_for_win, (list, tuple)):
                ests_for_win = [ests_for_win]
            ret_ests_for_win = []
            for est in ests_for_win:
                # X_wins[wi] = (60000, 11, 11, 49)
                _, nh, nw, _ = X_wins[wi].shape
                X_wins[wi] = X_wins[wi].reshape((X_wins[wi][0], -1, X_wins[wi][-1]))  # (60000, 121, 49)
                y_win = y_train[:, np.newaxis].repeat(X_wins[wi].shape[1], axis=1)
                y_proba = est.fit_transform(X_wins[wi], y_win, y_win[:, 0])  # (60000, 121, 10)
                y_proba = y_proba.reshape((-1, nh, nw, self.n_class)).transpose((0, 3, 1, 2))  # (60000, 10, 11, 11)
                ret_ests_for_win.append(y_proba)
            X_win_ests.append(ret_ests_for_win)
        if len(X_win_ests) == 0:
            return X_wins
        LOGGER.info('X_win_ests: {}'.format([j.shape for j in sub_res] for sub_res in X_win_ests))
        for pi, pool in enumerate(self.poolings):
            if not isinstance(pool, (list, tuple)):
                pool = [pool]
            for pj, pl in enumerate(pool):
                X_win_ests[pi][pj] = pl.fit_transform(X_win_ests[pi][pj])
        LOGGER.info('X_win_ests pooled: {}'.format([j.shape for j in sub_res] for sub_res in X_win_ests))
        return X_win_ests

    def transform(self, X, y=None):
        pass

    def predict(self, inputs):
        pass

    def evaluate(self, inputs, labels):
        pass

    def __str__(self):
        return self.__class__.__name__





