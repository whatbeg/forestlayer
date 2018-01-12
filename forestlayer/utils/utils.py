# -*- coding:utf-8 -*-
"""
Base utilities.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0


def check_list_depth(lis):
    if lis is None:
        return 0
    depth = 0
    tmp = lis
    while isinstance(tmp, (list, tuple)):
        depth += 1
        tmp = tmp[0]
    return depth

