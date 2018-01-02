# -*- coding:utf-8 -*-
"""
UCI_ADULT Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestflow.datasets import uci_adult
import ray
import time

ray.init()


@ray.remote(num_cpus=2)
def load_uci_data(filename):
    return uci_adult.load_data(filename)


start_time = time.time()

data = [load_uci_data.remote(f) for f in ("adult_4x.data", "adult_4x.test")]
(x_train, y_train), (x_test, y_test) = ray.get(data)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train.shape[1], 'features')

end_time = time.time()
print('time cost: {}'.format(end_time - start_time))
