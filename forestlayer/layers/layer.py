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
from ..utils.metrics import accuracy_pb, auc
from ..estimators import get_estimator_kfold
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
        """
        Initialize a layer.
        :param kwargs:
        """
        allowed_kwargs = {'batch_size',
                          'dtype',
                          'name'}
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        name = kwargs.get('name')
        if not name:
            prefix = self.__class__.__name__
            name = _to_snake_case(prefix) + "_" + str(id(self))
        self.name = name
        # Set dtype.
        dtype = kwargs.get('dtype')
        if dtype is None:
            dtype = F.floatx()
        self.dtype = dtype
        self.input_layer = None
        self.output_layer = None

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

    def transform(self, inputs, labels=None):
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError

    def evaluate(self, inputs, labels):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class InputLayer(Layer):
    # TODO: think whether it's redundant
    def __init__(self, input_shape=None, batch_size=None, dtype=None, name=None):
        if not name:
            prefix = 'input'
            name = prefix + '_' + str(id(self))
        super(InputLayer, self).__init__(dtype=dtype, name=name)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_est = 0
        self.estimators = []

    def call(self, x_trains):
        return x_trains

    def __call__(self, x_trains):
        self.call(x_trains)

    def fit(self, x_trains, y_trains):
        return x_trains

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
    def __init__(self, batch_size=None, dtype=None, name=None,
                 windows=None, est_for_windows=None, n_class=None):
        if not name:
            prefix = 'multi_grain_scan'
            name = prefix + '_' + str(id(self))
        super(MultiGrainScanLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        self.batch_size = batch_size
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

    def transform(self, inputs, labels=None):
        pass

    def predict(self, inputs):
        pass

    def evaluate(self, inputs, labels):
        pass

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

    def transform(self, inputs, labels=None):
        pass

    def evaluate(self, inputs, labels):
        pass

    def predict(self, inputs):
        pass


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

    def transform(self, inputs, labels=None):
        pass

    def evaluate(self, inputs, labels=None):
        pass

    def fit(self, x_trains, y_trains=None):
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

    def predict(self, inputs):
        pass


class CascadeLayer(Layer):
    def __init__(self, est_configs, kwargs):
        """Cascade Layer

        # Arguments:
            name
            est_configs
            n_classes
            layer_id
            data_save_dir
            metrics
            seed
            keep_in_mem
        # Properties
            eval_metrics
            fit_estimators
            train_avg_metric
            test_avg_metric

        # Raises
            RuntimeError: if estimator.fit_transform returns None data
            ValueError: if estimator.fit_transform returns wrong shape data
        """
        self.est_configs = est_configs
        allowed_args = {'n_classes',
                        'data_save_dir',
                        'name',
                        'layer_id',
                        'seed',
                        'metrics',
                        'keep_in_mem',
                        'batch_size',
                        'dtype',
                        }

        for kwarg in kwargs:
            if kwarg not in allowed_args:
                LOGGER.warn("Unidentified argument {}, ignore it!".format(kwarg))
        self.layer_id = kwargs.get('layer_id')
        if self.layer_id is None:
            self.layer_id = "Unknown"
        name = kwargs.get('name')
        if not name:
            name = 'layer-{}'.format(self.layer_id)
        dtype = kwargs.get('dtype')
        super(CascadeLayer, self).__init__(name=name, dtype=dtype)
        self.n_classes = kwargs.get('n_classes')
        self.data_save_dir = kwargs.get('data_save_dir')
        check_dir(self.data_save_dir)  # check dir, if not exists, create the dir
        self.seed = kwargs.get('seed')
        self.metrics = kwargs.get('metrics')
        if self.metrics == 'predict':
            self.eval_metrics = [('accuracy', accuracy_pb)]
        elif self.metrics == 'auc':
            self.eval_metrics = [('auc', auc)]
        else:
            self.eval_metrics = [('accuracy', accuracy_pb)]
        self.keep_in_mem = kwargs.get('keep_in_mem')
        if self.keep_in_mem is None:
            self.keep_in_mem = True
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
        :return: train_output, test_output
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
        for ei, est_config in enumerate(self.est_configs):
            est = self._init_estimators(self.layer_id, ei)
            # fit and transform
            y_proba_train, _ = est.fit_transform(x_train, y_train, y_train, test_sets=None)
            # print(y_proba_train.shape, y_proba_test.shape)
            if y_proba_train is None:
                raise RuntimeError("layer - {} - estimator - {} fit FAILED!".format(self.layer_id, ei))
            if y_proba_train.shape != (n_trains, n_classes):
                raise ValueError('output probability shape incorrect!,'
                                 ' should be {}, but {}'.format((n_trains, n_classes), y_proba_train.shape))
            self.fit_estimators = est
            x_proba_train[:, ei * n_classes:ei * n_classes + n_classes] = y_proba_train
            eval_proba_train += y_proba_train
        eval_proba_train /= self.n_estimators
        train_avg_acc = calc_accuracy(y_train, np.argmax(eval_proba_train, axis=1),
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
        for ei, est_config in enumerate(self.est_configs):
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
            self.fit_estimators = est
            x_proba_train[:, ei*n_classes:ei*n_classes + n_classes] = y_proba_train
            x_proba_test[:, ei*n_classes:ei*n_classes + n_classes] = y_proba_test
            eval_proba_train += y_proba_train
            eval_proba_test += y_proba_test
        eval_proba_train /= self.n_estimators
        eval_proba_test /= self.n_estimators
        train_avg_acc = calc_accuracy(y_train, np.argmax(eval_proba_train, axis=1),
                                      'layer - {} - [train] average'.format(self.layer_id))
        self.train_avg_metric = train_avg_acc
        # judge whether y_test is None, which means users are to predict test probas
        if y_test is not None:
            test_avg_acc = calc_accuracy(y_test, np.argmax(eval_proba_test, axis=1),
                                         'layer - {} - [test] average'.format(self.layer_id))
            self.test_avg_metric = test_avg_acc
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

    def transform(self, inputs, labels=None):
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError

    def evaluate(self, inputs, labels):
        raise NotImplementedError


class AutoGrowingCascadeLayer(Layer):
    def __init__(self, est_configs, kwargs):
        """
        early_stopping_rounds: int
            when not None , means when the accuracy does not increase in
            early_stopping_rounds, the cascade level will stop automatically growing
        est_configs: list of estimator arguments
            identify the estimator configuration to construct at this layer
        max_layers: int
            maximum number of cascade layers allowed for experiments,
            0 means do NOT use Early Stopping to automatically find the layer number
        n_classes: int
            Number of classes
        look_index_cycle (2d list): default=None
            specification for layer i, look for the array in look_index_cycle[i % len(look_index_cycle)]
            default = None <=> [[i,] for i in range(n_groups)]
            .e.g.
                look_index_cycle = [[0,1],[2,3],[0,1,2,3]]
                means layer 1 look for the grained 0,1; layer 2 look for grained 2,3;
                layer 3 look for every grained, and layer 4 cycles back as layer 1
        data_save_rounds: int [default=0]
        data_save_dir: str [default=None]
            each data_save_rounds save the intermediate results in data_save_dir
            if data_save_rounds = 0, then no savings for intermediate results
        """
        self.est_configs = est_configs
        allowed_args = {'early_stop_rounds',
                        'max_layers',
                        'n_classes',
                        'look_index_cycle',
                        'data_save_rounds',
                        'data_save_dir',
                        'keep_in_mem',
                        'stop_by_test',
                        'name',
                        'batch_size',
                        'dtype',
                        }

        for kwarg in kwargs:
            if kwarg not in allowed_args:
                LOGGER.warn("Unidentified argument {}, ignore it!".format(kwarg))
        name = kwargs.get('name')
        dtype = kwargs.get('dtype')
        super(AutoGrowingCascadeLayer, self).__init__(name=name, dtype=dtype)
        self.name = name
        self.early_stop_rounds = kwargs.get('early_stop_rounds', 4)
        self.max_layers = kwargs.get('max_layers', 0)
        self.n_classes = kwargs.get('n_classes')
        # if look_index_cycle is None, you need set look_index_cycle in fit / fit_transform
        self.look_index_cycle = kwargs.get('look_index_cycle')
        self.data_save_rounds = kwargs.get('data_save_rounds')
        self.data_save_dir = kwargs.get('data_save_dir')
        check_dir(self.data_save_dir)  # check data save dir, if not exists, create the dir
        self.keep_in_mem = kwargs.get('keep_in_mem', True)
        self.stop_by_test = kwargs.get('stop_by_test', False)
        self.layer_fit_cascades = []
        self.n_layers = 0
        self.opt_layer_id = 0

    def _create_cascade_layer(self, kwargs):
        return CascadeLayer(est_configs=self.est_configs,
                            kwargs=kwargs)

    def call(self, x_trains):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def fit(self, x_trains, y_train):
        if not isinstance(x_trains, (list, tuple)):
            x_trains = [x_trains]
        n_groups_train = len(x_trains)
        n_trains = len(y_train)
        # Initialize the groups
        x_train_group = np.zeros((n_trains, 0), dtype=x_trains[0].dtype)
        group_starts, group_ends, group_dims = [], [], []
        # train set
        for i, x_train in enumerate(x_trains):
            assert x_train.shape[0] == n_trains
            x_train = x_train.reshape(n_trains, -1)
            group_dims.append(x_train.shape[1])
            group_starts.append(i if i == 0 else group_ends[i - 1])
            group_ends.append(group_starts[i] + group_dims[i])
            x_train_group = np.hstack((x_train_group, x_train))

        LOGGER.info('group_starts={}'.format(group_dims))
        LOGGER.info('group_dims={}'.format(group_dims))
        LOGGER.info('X_train_group={}'.format(x_train_group.shape))

        if self.look_index_cycle is None:
            self.look_index_cycle = [[i, ] for i in range(n_groups_train)]
        x_cur_train = np.zeros((n_trains, 0), dtype=np.float32)
        x_proba_train = np.zeros((n_trains, 0), dtype=np.float32)
        layer_id = 0
        layer_metric_list = []
        opt_data = [None, None]
        try:
            while True:
                if layer_id >= self.max_layers > 0:
                    break
                train_ids = self.look_index_cycle[layer_id % n_groups_train]
                for gid in train_ids:
                    x_cur_train = np.hstack((x_cur_train, x_train_group[:, group_starts[gid]:group_ends[gid]]))
                x_cur_train = np.hstack((x_cur_train, x_proba_train))
                kwargs = {
                    'n_classes': self.n_classes,
                    'data_save_dir': osp.join(self.data_save_dir, 'cascade_layer_{}'.format(layer_id)),
                    'layer_id': layer_id,
                    'keep_in_mem': self.keep_in_mem,
                    'dtype': self.dtype,
                }
                cascade = self._create_cascade_layer(kwargs=kwargs)
                x_proba_train, x_proba_test = cascade.fit_transform(x_cur_train, y_train)
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
        n_groups_train = len(x_trains)
        n_groups_test = len(x_tests)
        n_trains = len(y_train)
        n_tests = x_tests[0].shape[0]
        if y_test is None and self.stop_by_test is True:
            raise ValueError("Since y_test is None so you cannot set stop_by_test True!")
        assert n_groups_train == n_groups_test, 'n_group_train must equal to n_group_test now! Sorry about that!'
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
                kwargs = {
                    'n_classes': self.n_classes,
                    'data_save_dir': osp.join(self.data_save_dir, 'cascade_layer_{}'.format(layer_id)),
                    'layer_id': layer_id,
                    'keep_in_mem': self.keep_in_mem,
                    'dtype': self.dtype,
                }
                cascade = self._create_cascade_layer(kwargs=kwargs)
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
            if y_test is None and cascade is not None:
                self.save_test_result(x_proba_test=cascade.eval_proba_test)
            return x_cur_train, x_cur_test
        except KeyboardInterrupt:
            pass

    def predict(self, inputs):
        pass

    def transform(self, inputs, labels=None):
        pass

    def evaluate(self, inputs, labels):
        pass

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


def calc_accuracy(y_true, y_pred, name, prefix=""):
    acc = 100. * np.sum(np.asarray(y_true) == y_pred) / len(y_true)
    LOGGER.info('{}Accuracy({})={:.2f}%'.format(prefix, name, acc))
    return acc


def calc_auc(y_true, y_pred, name, prefix=""):
    auc_res = auc(y_true, y_pred)
    LOGGER.info('{}Accuracy({})={:.2f}%'.format(prefix, name, auc_res*100.0))
    return auc_res


def get_opt_layer_id(acc_list):
    """ Return layer id with max accuracy on training data """
    opt_layer_id = np.argsort(-np.asarray(acc_list), kind='mergesort')[0]
    return opt_layer_id


def get_opt_layer_id_best(acc_list_train, acc_list_test):
    """ Return layer id with max accuracy on training and testing data"""
    opt_layer_id_train = np.argsort(-np.asarray(acc_list_train), kind='mergesort')[0]
    opt_layer_id_test = np.argsort(-np.asarray(acc_list_test), kind='mergesort')[0]
    opt_layer_id = opt_layer_id_train if opt_layer_id_train > opt_layer_id_test else opt_layer_id_test
    return opt_layer_id
