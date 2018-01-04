# -*- coding:utf-8 -*-
"""
UCI_ADULT Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.datasets import uci_adult
from forestlayer.layers import Input, Graph
from forestlayer.utils.metrics import accuracy_pb
import time

start_time = time.time()
(x_train, y_train) = uci_adult.load_data("adult.data")
(x_test, y_test) = uci_adult.load_data("adult.test")

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train.shape[1], 'features')

end_time = time.time()
print('time cost: {}'.format(end_time - start_time))

x = Input(x_train.shape, name='input')
print(x)

model = Graph()
model.add(x)
model.build()
model.fit(x_train, y_train)
pred = model.predict(x_test)
eval_ans = model.evaluate(accuracy_pb, x_test, y_test)
print(eval_ans)
