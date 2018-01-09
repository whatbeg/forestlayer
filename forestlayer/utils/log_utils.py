# -*- coding:utf-8 -*-
import os
import os.path as osp
import time
import logging
from ..backend.backend import get_base_dir


def get_log_base():
    return osp.join(get_base_dir(), 'log')


logging.basicConfig(format="[ %(asctime)s][%(module)s.%(funcName)s] %(message)s")

DEFAULT_LEVEL = logging.INFO
DEFAULT_LOGGING_DIR = osp.join(get_log_base(), "forestlayer")
fh = None


def strftime(t=None):
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(t or time.time()))


def init_fh():
    global fh
    if fh is not None:
        return
    if DEFAULT_LOGGING_DIR is None:
        return
    if not osp.exists(DEFAULT_LOGGING_DIR):
        os.makedirs(DEFAULT_LOGGING_DIR)
    logging_path = osp.join(DEFAULT_LOGGING_DIR, strftime() + ".log")
    fh = logging.FileHandler(logging_path)
    fh.setFormatter(logging.Formatter("[ %(asctime)s][%(module)s.%(funcName)s] %(message)s"))


def update_default_level(defalut_level):
    global DEFAULT_LEVEL
    DEFAULT_LEVEL = defalut_level


def update_default_logging_dir(default_logging_dir):
    global DEFAULT_LOGGING_DIR
    DEFAULT_LOGGING_DIR = default_logging_dir


def get_logger(name="forestlayer", level=None):
    level = level or DEFAULT_LEVEL
    logger = logging.getLogger(name)
    logger.setLevel(level)
    init_fh()
    if fh is not None:
        logger.addHandler(fh)
    return logger


def list2str(lis, dim):
    result = "["
    for l in lis:
        if dim == 1:
            result += '{},'.format(l.shape)
        elif dim == 2:
            result += '['
            for j in l:
                result += '{} '.format(j.shape)
            result += '], '
        elif dim == 3:
            result += '['
            for j in l:
                result += '['
                for k in j:
                    result += '{} '.format(k.shape)
                result += '], '
            result += '], '
        else:
            raise NotImplementedError
    result += ']'
    return result
