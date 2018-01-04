# -*- coding:utf-8 -*-
"""
MNIST Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function

from forestlayer.layers import Input
from forestlayer.layers import Graph
from keras.datasets import mnist


# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x = Input(x_train.shape, name='input')
print(x)

model = Graph()
model.add(x)
model.build()
model.fit(x_train, y_train)
model.predict(x_test)
