# -*- coding:utf-8 -*-
"""
Scikit-learn Estimators Definition.
"""

from base_estimator import *
from ..utils.log_utils import get_logger
from sklearn.externals import joblib

LOGGER = get_logger('estimators.sklearn_estimator')


def forest_predict_batch_size(clf, X, task):
    """
    Decide predict batch size by calculating memory occupation.
    :param clf:
    :param X:
    :param task:
    :return:
    """
    import psutil
    free_memory = psutil.virtual_memory().free
    if free_memory < 2e9:
        free_memory = int(2e9)
    max_mem_size = max(int(free_memory * 0.5), int(8e10))
    if task == 'regression':
        mem_size_1 = clf.n_estimators * 16
    else:
        mem_size_1 = clf.n_classes_ * clf.n_estimators * 16
    batch_size = (max_mem_size - 1) / mem_size_1 + 1
    if batch_size < 10:
        batch_size = 10
    if batch_size >= X.shape[0]:
        return 0
    return batch_size


class SKlearnBaseEstimator(BaseEstimator):
    """
    SKlearn base estimators inherited from BaseEstimator.
    """
    def _save_model_to_disk(self, est, cache_path):
        """
        Save model to disk using joblib.
        :param est:
        :param cache_path:
        :return:
        """
        joblib.dump(est, cache_path)

    def _load_model_from_disk(self, cache_path):
        """
        Load model from disk using joblib.
        :param cache_path:
        :return:
        """
        return joblib.load(cache_path)

    def copy(self):
        """
        copy
        :return:
        """
        return SKlearnBaseEstimator(est_class=self.est_class, **self.est_args)


class RFClassifier(SKlearnBaseEstimator):
    """
    Random Forest Classifier
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import RandomForestClassifier
        super(RFClassifier, self).__init__('classification', RandomForestClassifier, name, kwargs)

    def _default_predict_batch_size(self, est, X, task='classification'):
        return forest_predict_batch_size(est, X, task)


class CompletelyRFClassifier(SKlearnBaseEstimator):
    """
    Completely Random Forest Classifier
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import ExtraTreesClassifier
        super(CompletelyRFClassifier, self).__init__('classification', ExtraTreesClassifier, name, kwargs)

    def _default_predict_batch_size(self, est, X, task='classification'):
        return forest_predict_batch_size(est, X,  task)


class GBDTClassifier(SKlearnBaseEstimator):
    """
    Gradient Boosting Decision Tree Classifier.
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import GradientBoostingClassifier
        super(GBDTClassifier, self).__init__('classification', GradientBoostingClassifier, name, kwargs)


class SKXGBoostClassifier(SKlearnBaseEstimator):
    """
    XGBoost Classifier using Sklearn interfaces.
    """
    def __init__(self, name, kwargs):
        from xgboost import XGBClassifier
        super(SKXGBoostClassifier, self).__init__('classification', XGBClassifier, name, kwargs)


class RFRegressor(SKlearnBaseEstimator):
    """
    Random Forest Regressor.
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import RandomForestRegressor
        super(RFRegressor, self).__init__('regression', RandomForestRegressor, name, kwargs)

    def _default_predict_batch_size(self, est, X, task='regression'):
        return forest_predict_batch_size(est, X, task)


class CompletelyRFRegressor(SKlearnBaseEstimator):
    """
    Completely Random Forest Regressor.
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import ExtraTreesRegressor
        super(CompletelyRFRegressor, self).__init__('regression', ExtraTreesRegressor, name, kwargs)

    def _default_predict_batch_size(self, est, X, task='regression'):
        return forest_predict_batch_size(est, X, task)


class GBDTRegressor(SKlearnBaseEstimator):
    """
    Gradient Boosting Decision Tree Regressor.
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import GradientBoostingRegressor
        super(GBDTRegressor, self).__init__('regression', GradientBoostingRegressor, name, kwargs)


class SKXGBoostRegressor(SKlearnBaseEstimator):
    """
    XGBoost Regressor using Sklearn interfaces.
    """
    def __init__(self, name, kwargs):
        from xgboost import XGBRegressor
        super(SKXGBoostRegressor, self).__init__('regression', XGBRegressor, name, kwargs)


