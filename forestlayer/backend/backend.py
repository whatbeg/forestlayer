# -*- coding:utf-8 -*-
"""
Scikit-learn and Ray as backend.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from sklearn import metrics
import ray
import numpy as np


def pb2pred(y_proba):
    y_pred = np.argmax(y_proba.reshape((-1, y_proba.shape[-1])), 1)
    return y_pred

