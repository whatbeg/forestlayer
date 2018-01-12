# -*- coding:utf-8 -*-
"""
DAG definition
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from ..utils.log_utils import get_logger
from ..utils.metrics import Accuracy, MSE
from ..layers.layer import Layer
from ..utils.layer_utils import print_summary
import copy
import numpy as np
import time


class Graph(object):
    """
    A Graph is composed of a sequence of standalone, fully-configurable modules, which is called layers,
     that can be combined together with as little restrictions as possible.
    Now a graph defines a stacking structure of layers, do fit, fit_transform layer by layer, which has not complex
     graph building mechanism.
    A graph is stateful, that's to say, after fit, the graph has states, which contains fitted layers. If you
     dynamically add layers to it, you need to re-fit the whole graph. Later we shall consider how to support
     dynamic graph construction and execution.
    """
    def __init__(self, task='classification'):
        """
        Initialize a graph for specific task.

        :param task:
        """
        self.LOGGER = get_logger('forestlayer.layers.graph')
        self.layers = []
        self.task = task
        self.FIT = False

    def call(self):
        pass

    def __call__(self, *args, **kwargs):
        self.call()

    def _add(self, layer):
        """
        layer add inner method, just add one layer.

        :param layer:
        :return:
        """
        if layer is None or not isinstance(layer, Layer):
            self.LOGGER.info('layer [{}] is invalid!'.format(layer))
            return
        self.layers.append(layer)

    def add(self, layer, *layers):
        """
        Add one or more layers.

        :param layer: at least one layer to be add
        :param layers: additional layers, optional
        :return:
        """
        self._add(layer)
        for lay in layers:
            self._add(lay)

    def build(self):
        """
        Print the graph.

        :return:
        """
        self.LOGGER.info("graph build finished!")
        self.LOGGER.info(self.to_debug_string())

    def summary(self):
        print_summary(self)

    def fit(self, x_trains, y_trains):
        """
        Fit with x_trians, y_trains.

        :param x_trains:
        :param y_trains:
        :return: self
        """
        start_time = time.time()
        self.build()
        if not isinstance(x_trains, (list, tuple)):
            inputs = [x_trains]
        else:
            inputs = copy.deepcopy(x_trains)
        if not isinstance(y_trains, (list, tuple)):
            labels = [y_trains]
        else:
            labels = copy.deepcopy(y_trains)
        for layer in self.layers:
            self.LOGGER.info(" -------------- Now fitting layer [{}] --------------".format(layer))
            inputs = layer.fit(inputs, labels)
        time_cost = time.time() - start_time
        self.LOGGER.info("graph fit finished! Time Cost: {} s".format(time_cost))
        self.FIT = True
        return self

    def fit_transform(self, x_trains, y_trains, x_tests=None, y_tests=None):
        """
        Fit transform the inputs: x_trains, y_trains, x_tests, y_tests.
        x_tests and y_tests is optional.
        If x_tests is None, we do not evaluate test metrics.
        If x_tests is not None but y_tests is None, we consider that user wants to train on training data and predict
         on testing data by the way, so we will save predict results for tests.

        :param x_trains: numpy ndarray or list of numpy ndarray.
        :param y_trains: numpy ndarray or list of numpy ndarray.
        :param x_tests: optional. numpy ndarray or list of numpy ndarray.
        :param y_tests: optional. numpy ndarray or list of numpy ndarray.
        :return:
        """
        start_time = time.time()
        self.build()
        if not isinstance(x_trains, (list, tuple)):
            x_trains = [x_trains]
        if not isinstance(y_trains, (list, tuple)):
            y_trains = [y_trains]
        if x_tests is not None and not isinstance(x_tests, (list, tuple)):
            x_tests = [x_tests]
        if y_tests is not None and not isinstance(y_tests, (list, tuple)):
            y_tests = [y_tests]
        for li, layer in enumerate(self.layers):
            self.LOGGER.info(" -------------- Now fitting layer - [{}] [{}] --------------".format(li, layer))
            x_trains, x_tests = layer.fit_transform(x_trains, y_trains, x_tests, y_tests)
        time_cost = time.time() - start_time
        self.LOGGER.info("graph fit_transform finished! Time Cost: {} s".format(time_cost))
        self.FIT = True
        return x_trains, x_tests

    def transform(self, inputs):
        """
        Transform inputs.

        :param inputs:
        :return:
        """
        if self.FIT is False:
            raise RuntimeError('You must fit the graph before predict')
        if not isinstance(inputs, (list, tuple)):
            X = [inputs]
        else:
            X = inputs
        for layer in self.layers:
            X = layer.transform(X)
        self.LOGGER.info("graph transform finished!")
        return X

    def predict(self, inputs):
        """
        Predict outputs of the inputs.

        :param inputs:
        :return:
        """
        if self.FIT is False:
            raise RuntimeError('You must fit the graph before predict')
        X = self.predict_proba(inputs)
        return np.argmax(X.reshape((-1, self.layers[-1].n_classes)), axis=1)

    def predict_proba(self, inputs):
        """
        Predict probability outputs of the inputs.

        :param inputs:
        :return:
        """
        if self.FIT is False:
            raise RuntimeError('You must fit the graph before predict')
        if not isinstance(inputs, (list, tuple)):
            X = [inputs]
        else:
            X = inputs
        for layer in self.layers[:-1]:
            X = layer.transform(X)
        X = self.layers[-1].predict_proba(X)
        return X

    @property
    def is_classification(self):
        """
        True if the task is classification.

        :return:
        """
        return self.task == 'classification'

    def evaluate(self, inputs, labels, eval_metrics=None):
        """
        Evaluate inputs with labels.

        :param inputs:
        :param labels:
        :param eval_metrics:
        :return:
        """
        if eval_metrics is None:
            if self.is_classification:
                eval_metrics = [Accuracy('evaluate')]
            else:
                eval_metrics = [MSE('evaluate')]
        # make eval_metrics iterative
        if not isinstance(eval_metrics, (list, tuple)):
            eval_metrics = [eval_metrics]
        for metric in eval_metrics:
            metric(labels, self.predict_proba(inputs), prefix='', logger=self.LOGGER)

    def to_debug_string(self):
        """
        To debug string, to see the graph structure.

        :return:
        """
        debug_str = '\n'
        for i, layer in enumerate(self.layers):
            debug_str += "{}".format(layer)
            if i != len(self.layers)-1:
                debug_str += " -> "
        return debug_str







