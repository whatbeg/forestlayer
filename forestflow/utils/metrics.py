# -*- coding:utf-8 -*-
import numpy as np
from sklearn import metrics


def accuracy(y_true, y_pred):
    return 1.0 * np.sum(y_true == y_pred) / len(y_true)


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
