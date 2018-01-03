# -*- coding:utf-8 -*-
"""
Factory methods to Layers.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from .layer import *


def Input(shape=None, input_tensors=None, name=None, dtype=None):
    if not dtype:
        if input_tensors is not None:
            if isinstance(input_tensors, list):
                dtype = input_tensors[0].dtype
            else:
                dtype = input_tensors.dtype
        else:
            dtype = F.floatx()
    return InputLayer(input_shape=shape, name=name, dtype=dtype)
