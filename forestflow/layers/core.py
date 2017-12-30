# -*- coding:utf-8 -*-
"""
Base layers definition
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import random as rnd
from .. import backend as F


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

    def transform(self, inputs):
        return inputs

    def predict(self, inputs):
        raise NotImplementedError

    def evaluate(self, inputs, labels):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__
