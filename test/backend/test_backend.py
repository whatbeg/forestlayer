# -*- coding:utf-8 -*-
"""
Test Suite of forestlayer.backend.backend.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import os.path as osp
from forestlayer.backend.backend import set_base_dir, get_base_dir


def test_base_dir():
    print("base dir: " + get_base_dir())
    set_base_dir(osp.expanduser(osp.join('~', 'forestlayer')))
    print('after set, base dir = {}'.format(get_base_dir()))


test_base_dir()



