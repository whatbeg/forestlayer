# -*- coding:utf-8 -*-
"""
Metrics.
"""

import numpy as np
from sklearn import metrics


class Metrics(object):
    def __init__(self, name=''):
        self.name = name

    def __call__(self, y_true, y_pred, prefix='', logger=None):
        if y_true is None or y_pred is None:
            return
        if not isinstance(y_pred, type(np.array)):
            y_pred = np.asarray(y_pred)
        if y_pred.shape[1] > 1:
            return self.calc_proba(y_true, y_pred, prefix=prefix, logger=logger)
        elif y_pred.shape[1] == 1:
            return self.calc(y_true, y_pred, prefix=prefix, logger=logger)
        else:
            raise ValueError('y_pred.shape={} does not confirm the restriction!'.format(y_pred.shape))

    def calc(self, y_true, y_pred, prefix='', logger=None):
        raise NotImplementedError

    def calc_proba(self, y_true, y_proba, prefix='', logger=None):
        raise NotImplementedError


class Accuracy(Metrics):
    def __init__(self, name=''):
        super(Accuracy, self).__init__(name)

    def calc(self, y_true, y_pred, prefix='', logger=None):
        if y_true is None or y_pred is None:
            return
        acc = 100. * np.sum(np.asarray(y_true) == y_pred) / len(y_true)
        if logger is not None:
            logger.info('{} Accuracy({}) = {:.2f}%'.format(prefix, self.name, acc))
        return acc

    def calc_proba(self, y_true, y_proba, prefix='', logger=None):
        y_true = y_true.reshape(-1)
        y_pred = np.argmax(y_proba.reshape((-1, y_proba.shape[-1])), 1)
        acc = 100. * np.sum(y_true == y_pred) / len(y_true)
        if logger is not None:
            logger.info('{} Accuracy({}) = {:.2f}%'.format(prefix, self.name, acc))
        return acc


class AUC(Metrics):
    def __init__(self, name=''):
        super(AUC, self).__init__(name)

    def calc(self, y_true, y_pred, prefix='', logger=None):
        assert y_pred.shape[1] == 2, 'auc metric is restricted to the binary classification task!'
        return self.calc_proba(y_true, y_pred, prefix, logger)

    def calc_proba(self, y_true, y_proba, prefix='', logger=None):
        assert y_proba.shape[1] == 2, 'auc metric is restricted to the binary classification task!'
        y_true = y_true.reshape(-1)
        auc_result = auc(y_true, y_proba)
        if logger is not None:
            logger.info('{} AUC({}) = {:.4f}'.format(prefix, self.name, auc_result))
        return auc_result


class MSE(Metrics):
    def __init__(self, name=''):
        super(MSE, self).__init__(name)

    def __call__(self, y_true, y_pred, prefix='', logger=None):
        return self.calc(y_true, y_pred, prefix, logger)

    def calc(self, y_true, y_pred, prefix='', logger=None):
        return metrics.mean_squared_error(y_true, y_pred)

    def calc_proba(self, y_true, y_proba, prefix='', logger=None):
        return metrics.mean_squared_error(y_true, y_proba)


def accuracy(y_true, y_pred):
    return 1.0 * np.sum(np.asarray(y_true) == y_pred) / len(y_true)


def accuracy_pb(y_true, y_proba):
    y_true = y_true.reshape(-1)
    y_pred = np.argmax(y_proba.reshape((-1, y_proba.shape[-1])), 1)
    return 1.0 * np.sum(y_true == y_pred) / len(y_true)


def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y))], dtype=np.float)
    g = g[np.lexsort((g[:, 2], -1 * g[:, 1]))]
    gs = g[:, 0].cumsum().sum() / g[:, 0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)


def gini_nor(y_true, y_proba):
    y_proba = [item[1] for item in y_proba]
    y_proba = np.array(y_proba)
    return gini(y_true, y_proba) / gini(y_true, y_true)


def auc(y_true, y_proba):
    y_proba = [item[1] for item in y_proba]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_proba, pos_label=1)
    return metrics.auc(fpr, tpr)
