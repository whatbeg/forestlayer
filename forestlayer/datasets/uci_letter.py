# -*- coding:utf-8 -*-
"""
UCI_LETTER dataset loading.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from .dataset import get_data_base
import numpy as np
import os.path as osp


def load_data():
    data_path = osp.join(get_data_base(), 'uci_letter', "letter-recognition.data")
    with open(data_path) as f:
        rows = [row.strip().split(',') for row in f.readlines()]
    n_datas = len(rows)
    X = np.zeros((n_datas, 16), dtype=np.float32)
    y = np.zeros(n_datas, dtype=np.int32)
    for i, row in enumerate(rows):
        X[i, :] = list(map(float, row[1:]))
        y[i] = ord(row[0]) - ord('A')
    X_train, y_train = X[:16000], y[:16000]
    X_test, y_test = X[16000:], y[16000:]
    return X_train, y_train, X_test, y_test

