# -*- coding:utf-8 -*-
"""
Storage Utilities, include cache utilities.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import os.path as osp
import os
from ..backend.backend import get_base_dir
import pickle


_DATA_SAVE_BASE = osp.join(get_base_dir(), 'run_data')

_MODEL_SAVE_BASE = osp.join(get_base_dir(), 'run_model')


def name2path(name):
    """
    Replace '/' in name by '-'
    """
    return name.replace("/", "-")


def is_path_exists(path):
    """
    Judge if path exists.
    :param path:
    :return:
    """
    return path is not None and osp.exists(path)


def check_dir(path):
    """
    Check directory existence, if not, create the directory.
    :param path:
    :return:
    """
    if path is None:
        return
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        os.makedirs(d)


def get_data_save_base():
    """
    Get data save base dir.
    :return:
    """
    global _DATA_SAVE_BASE
    _DATA_SAVE_BASE = osp.join(get_base_dir(), 'run_data')
    return _DATA_SAVE_BASE


def set_data_save_base(dir_path):
    """
    Set data save base dir.
    :param dir_path:
    :return:
    """
    global _DATA_SAVE_BASE
    _DATA_SAVE_BASE = dir_path
    check_dir(_DATA_SAVE_BASE)


def get_model_save_base():
    """
    Get model save base dir.
    :return:
    """
    global _MODEL_SAVE_BASE
    _MODEL_SAVE_BASE = osp.join(get_base_dir(), 'run_model')
    return _MODEL_SAVE_BASE


def set_model_save_base(dir_path):
    """
    Set model save base dir.
    """
    global _MODEL_SAVE_BASE
    _MODEL_SAVE_BASE = dir_path
    check_dir(_MODEL_SAVE_BASE)

