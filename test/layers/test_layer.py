# -*- coding:utf-8 -*-
"""
Test Suites of layers.layer.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import numpy as np
from forestflow.layers.window import Window, Pooling
from forestflow.layers.layer import MultiGrainScanLayer
from keras.datasets import mnist


# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = np.reshape(x_train, (60000, -1, 28, 28))

windows = [Window(X, 7, 7, 2, 2, 0, 0), Window(X, 11, 11, 2, 2, 0, 0)]
est_for_windows = [[EST1, EST2], [EST1, EST2]]
pools = [[Pooling(2, 2, "mean"), Pooling(2, 2, "mean")], [Pooling(2, 2, "mean"), Pooling(2, 2, "mean")]]

mgs = MultiGrainScanLayer(X.shape, None, dtype=np.float32, windows=windows, est_for_windows=est_for_windows,
                          pools=pools, n_class=10)

res = mgs.fit_transform(X, y_train)

print(res.shape)

