# -*- coding:utf-8 -*-
"""
Storage Utilities, include cache utilities.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import os.path as osp
import os
import pickle


def name2path(name):
    """
    Replace '/' in name by '-'
    """
    return name.replace("/", "-")


def is_path_exists(path):
    return path is not None and osp.exists(path)


def check_dir(path):
    if path is None:
        return
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        os.makedirs(d)


def get_data_save_base():
    return osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, "run_data"))


def get_model_save_base():
    return osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, "run_model"))
