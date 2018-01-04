# -*- coding:utf-8 -*-
"""
Base dataset.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import os.path as osp


def get_data_base():
    return osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, "data"))


def get_data_cache_base():
    return osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, "data-cache"))
