# -*- coding:utf-8 -*-
"""
==============================================================
scikit-learn based Regressor definition
==============================================================
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from .sklearn_estimator import SKLearnBaseEstimator, forest_predict_batch_size


class FLRFRegressor(SKLearnBaseEstimator):
    """
    Random Forest Regressor.
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import RandomForestRegressor
        super(FLRFRegressor, self).__init__('regression', RandomForestRegressor, name, kwargs)

    def _default_predict_batch_size(self, est, X, task='regression'):
        return forest_predict_batch_size(est, X, task)


class FLCRFRegressor(SKLearnBaseEstimator):
    """
    Completely Random Forest Regressor.
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import ExtraTreesRegressor
        super(FLCRFRegressor, self).__init__('regression', ExtraTreesRegressor, name, kwargs)

    def _default_predict_batch_size(self, est, X, task='regression'):
        return forest_predict_batch_size(est, X, task)


class FLGBDTRegressor(SKLearnBaseEstimator):
    """
    Gradient Boosting Decision Tree Regressor.
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import GradientBoostingRegressor
        super(FLGBDTRegressor, self).__init__('regression', GradientBoostingRegressor, name, kwargs)


class FLXGBoostRegressor(SKLearnBaseEstimator):
    """
    XGBoost Regressor using Sklearn interfaces.
    """
    def __init__(self, name, kwargs):
        from xgboost.sklearn import XGBRegressor
        super(FLXGBoostRegressor, self).__init__('regression', XGBRegressor, name, kwargs)


class FLLGBMRegressor(SKLearnBaseEstimator):
    """
    LightGBM Regressor using Sklearn interfaces.
    """
    def __init__(self, name, kwargs):
        from lightgbm.sklearn import LGBMRegressor
        super(FLLGBMRegressor, self).__init__('regression', LGBMRegressor, name, kwargs)


