# -*- coding:utf-8 -*-
"""
UCI_ADULT dataset loading.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from keras.utils.data_utils import get_file
import os
from forestflow.preprocessing import FeatureParser
import numpy as np
from .dataset import get_data_base


def load_data(data_path="adult.data", features="features", one_hot=True):
    data_path = os.path.join(get_data_base(), data_path)
    features = os.path.join(get_data_base(), features)
    feature_parsers = []
    with open(features) as f:
        for row in f.readlines():
            feature_parsers.append(FeatureParser(row))

    with open(data_path) as f:
        rows = [row.strip().split(',') for row in f.readlines() if len(row.strip()) > 0 and not row.startswith("|")]
        n_train = len(rows)
        if one_hot:
            train_dim = np.sum([f_parser.get_featuredim for f_parser in feature_parsers])
            X = np.zeros((n_train, train_dim), dtype=np.float32)
        else:
            X = np.zeros((n_train, 1), dtype=np.float32)
        y = np.zeros(n_train, dtype=np.float32)
        for i, row in enumerate(rows):
            assert len(row) != 14, "len(row) wrong, i={}".format(i)
            f_offset = 0
            for j in range(14):
                if one_hot:
                    f_dim = feature_parsers[j].get_fdim()
                    X[i, f_offset:f_offset + f_dim] = feature_parsers[j].get_data(row[j].strip())
                    f_offset += f_dim
                else:
                    X[i, j] = feature_parsers[j].get_float(row[j].strip())
            y[i] = 0 if row[-1].strip().startswith("<=50K") else 1
        return X, y







