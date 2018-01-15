# -*- coding:utf-8 -*-
"""
==============================================================
scikit-learn based Classifier definition
==============================================================
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from .sklearn_estimator import SKLearnBaseEstimator, forest_predict_batch_size


class FLRFClassifier(SKLearnBaseEstimator):
    """
    Random Forest Classifier
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import RandomForestClassifier
        super(FLRFClassifier, self).__init__('classification', RandomForestClassifier, name, kwargs)

    def _default_predict_batch_size(self, est, X, task='classification'):
        return forest_predict_batch_size(est, X, task)


class FLCRFClassifier(SKLearnBaseEstimator):
    """
    Completely Random Forest Classifier
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import ExtraTreesClassifier
        super(FLCRFClassifier, self).__init__('classification', ExtraTreesClassifier, name, kwargs)

    def _default_predict_batch_size(self, est, X, task='classification'):
        return forest_predict_batch_size(est, X,  task)


class FLGBDTClassifier(SKLearnBaseEstimator):
    """
    Gradient Boosting Decision Tree Classifier.
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import GradientBoostingClassifier
        super(FLGBDTClassifier, self).__init__('classification', GradientBoostingClassifier, name, kwargs)


class FLXGBoostClassifier(SKLearnBaseEstimator):
    """
    XGBoost Classifier using Sklearn interfaces.
    """
    def __init__(self, name, kwargs):
        from xgboost import XGBClassifier
        super(FLXGBoostClassifier, self).__init__('classification', XGBClassifier, name, kwargs)


class FLLGBMClassifier(SKLearnBaseEstimator):
    """
    LightGBM Classifier using Sklearn interfaces.
    """
    def __init__(self, name, kwargs):
        from lightgbm.sklearn import LGBMClassifier
        super(FLLGBMClassifier, self).__init__('classifier', LGBMClassifier, name, kwargs)

