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
import copy
import numpy as np

LOGGER = get_logger('forestflow.layers.graph')


class Graph(object):
    def __init__(self, task='classification'):
        self.layers = []
        self.task = task
        self.FIT = False

    def call(self):
        pass

    def __call__(self, *args, **kwargs):
        self.call()

    def _add(self, layer):
        if layer is None or not isinstance(layer, Layer):
            LOGGER.info('layer [{}] is invalid!'.format(layer))
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
        LOGGER.info("graph build finished!")
        LOGGER.info(self.to_debug_string())

    def fit(self, x_trains, y_trains):
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
            LOGGER.info(" -------------- Now fitting layer [{}] --------------".format(layer))
            inputs = layer.fit(inputs, labels)
        LOGGER.info("graph fit finished!")
        self.FIT = True
        return self

    def fit_transform(self, x_trains, y_trains, x_tests=None, y_tests=None):
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
            LOGGER.info(" -------------- Now fitting layer - [{}] [{}] --------------".format(li, layer))
            x_trains, x_tests = layer.fit_transform(x_trains, y_trains, x_tests, y_tests)
        LOGGER.info("graph fit_transform finished!")
        self.FIT = True
        return x_trains, x_tests

    def transform(self, inputs):
        if self.FIT is False:
            raise RuntimeError('You must fit the graph before predict')
        if not isinstance(inputs, (list, tuple)):
            X = [inputs]
        else:
            X = inputs
        for layer in self.layers:
            X = layer.transform(X)
        LOGGER.info("graph transform finished!")
        return X

    def predict(self, inputs):
        if self.FIT is False:
            raise RuntimeError('You must fit the graph before predict')
        X = self.predict_proba(inputs)
        return np.argmax(X.reshape((-1, self.layers[-1].n_classes)), axis=1)

    def predict_proba(self, inputs):
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
        return self.task == 'classification'

    def evaluate(self, inputs, labels, eval_metrics=None):
        if eval_metrics is None:
            if self.is_classification:
                eval_metrics = [Accuracy('evaluate')]
            else:
                eval_metrics = [MSE('evaluate')]
        # make eval_metrics iterative
        if not isinstance(eval_metrics, (list, tuple)):
            eval_metrics = [eval_metrics]
        for metric in eval_metrics:
            metric(labels, self.predict_proba(inputs), prefix='', logger=LOGGER)

    def to_debug_string(self):
        debug_str = '\n'
        for i, layer in enumerate(self.layers):
            debug_str += "{}".format(layer)
            if i != len(self.layers)-1:
                debug_str += " -> "
        return debug_str







