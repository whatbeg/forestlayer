# -*- coding:utf-8 -*-
"""
Base utilities.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import os.path as osp


def get_log_base():
    return osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, "log"))

