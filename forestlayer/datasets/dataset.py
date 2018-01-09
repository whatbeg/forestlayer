# -*- coding:utf-8 -*-
"""
Base dataset.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import os.path as osp
import os
from ..utils.log_utils import get_logger
from ..backend.backend import get_base_dir

LOGGER = get_logger('datasets.dataset')

_DATASET_DIR = osp.join(get_base_dir(), 'data')
_DATA_CACHE_DIR = osp.join(get_base_dir(), 'data-cache')


def get_data_base():
    global _DATASET_DIR
    _DATASET_DIR = osp.join(get_base_dir(), 'data')
    if not osp.exists(_DATASET_DIR):
        os.makedirs(_DATASET_DIR)
    return _DATASET_DIR


def set_data_dir(dir_path):
    global _DATASET_DIR
    _DATASET_DIR = dir_path
    if not osp.exists(_DATASET_DIR):
        os.makedirs(_DATASET_DIR)


def get_data_cache_base():
    global _DATA_CACHE_DIR
    _DATA_CACHE_DIR = osp.join(get_base_dir(), 'data-cache')
    if not osp.exists(_DATA_CACHE_DIR):
        os.makedirs(_DATA_CACHE_DIR)
    return _DATA_CACHE_DIR


def set_data_cache_base(dir_path):
    global _DATA_CACHE_DIR
    _DATA_CACHE_DIR = dir_path
    if not osp.exists(_DATA_CACHE_DIR):
        os.makedirs(_DATA_CACHE_DIR)
