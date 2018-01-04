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
        self.inputs_shape = []
        self.outputs_shape = []
        self.input_tensors = []
        self.output_tensors = []
        self.built = False
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
        for layer in self.layers:
            if layer.input_layer is not None:
                assert layer.input_layer.output_shape == layer.input_shape
            if layer.output_layer is not None:
                assert layer.output_layer.input_shape == layer.output_shape
        self.built = True
        LOGGER.info("graph build finished!")
        LOGGER.info(self.to_debug_string())

    def fit(self, inputs, labels):
        if self.built is False:
            raise RuntimeError('You must build the graph before fit')
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        self.input_tensors = inputs
        for layer in self.layers:
            prev_inputs = inputs
            inputs = layer.fit_transform(prev_inputs, labels)
            del prev_inputs
        LOGGER.info("graph fit finished!")
        self.FIT = True

    def fit_transform(self, inputs, labels):
        if self.built is False:
            raise RuntimeError('You must build the graph before fit')
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        for layer in self.layers:
            prev_inputs = inputs
            inputs = layer.fit_transform(prev_inputs, labels)
            del prev_inputs
        LOGGER.info("graph fit_transform finished!")
        self.FIT = True
        return inputs

    def transform(self, inputs):
        if self.built is False:
            raise RuntimeError('You must build the graph before fit')
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
        inputs = self.predict_proba(inputs)
        for i, inp in enumerate(inputs):
            inputs[i] = pb2pred(inp)
        if len(inputs) == 1:
            return inputs[0]
        return inputs

    def predict_proba(self, inputs):
        if self.built is False:
            raise RuntimeError('You must build and fit the graph before predict')
        if self.fit is False:
            raise RuntimeError('You must fit the graph before predict')
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        for layer in self.layers:
            prev_inputs = inputs
            inputs = layer.transform(prev_inputs)
            del prev_inputs
        return inputs

    def evaluate(self, eval_metrics, inputs, labels):
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
        debug_str += '\n'
        return debug_str







