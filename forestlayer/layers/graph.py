# -*- coding:utf-8 -*-
"""
DAG definition
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from ..utils import get_logger
from ..backend import pb2pred

LOGGER = get_logger('forestflow.layers.graph')


class Graph(object):
    def __init__(self):
        self.layers = []
        self.FIT = False

    def call(self):
        pass

    def __call__(self, *args, **kwargs):
        self.call()

    def add(self, layer):
        self.layers.append(layer)
        if len(self.layers) <= 1:
            layer.input_layer = None
        else:
            layer.input_layer = self.layers[-2]
            self.layers[-2].output_layer = layer

    def build(self):
        LOGGER.info("graph build finished!")
        LOGGER.info(self.to_debug_string())

    def fit(self, x_trains, y_trains):
        self.build()
        if not isinstance(x_trains, (list, tuple)):
            inputs = [x_trains]
        else:
            inputs = x_trains
        for layer in self.layers:
            inputs = layer.fit(inputs, y_trains)
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
        for layer in self.layers:
            LOGGER.info(" -------------- Now fitting layer [{}] --------------".format(layer))
            x_trains, x_tests = layer.fit_transform(x_trains, y_trains, x_tests, y_tests)
        LOGGER.info("graph fit_transform finished!")
        self.FIT = True
        return x_trains, x_tests

    def transform(self, inputs):
        # TODO
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        for layer in self.layers:
            prev_inputs = inputs
            inputs = layer.transform(prev_inputs)
            del prev_inputs
        LOGGER.info("graph fit_transform finished!")
        self.FIT = True
        return inputs

    def predict(self, inputs):
        # TODO
        inputs = self.predict_proba(inputs)
        for i, inp in enumerate(inputs):
            inputs[i] = pb2pred(inp)
        if len(inputs) == 1:
            return inputs[0]
        return inputs

    def predict_proba(self, inputs):
        if self.FIT is False:
            raise RuntimeError('You must fit the graph before predict')
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        for layer in self.layers:
            prev_inputs = inputs
            inputs = layer.transform(prev_inputs)
            del prev_inputs
        return inputs

    def evaluate(self, eval_metrics, inputs, labels):
        # TODO
        # make eval_metrics iterative
        if isinstance(eval_metrics, (list, tuple)) is not True:
            eval_metrics = [eval_metrics]
        metric_result = []
        for metric in eval_metrics:
            res = metric(labels, self.predict(inputs))
            metric_result.append(res)
        return metric_result

    def to_debug_string(self):
        debug_str = '\n'
        for i, layer in enumerate(self.layers):
            debug_str += "{}".format(layer)
            if i != len(self.layers)-1:
                debug_str += " -> "
        return debug_str







