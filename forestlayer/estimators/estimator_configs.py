# -*- coding:utf-8 -*-
"""
Estimator Configuration Class Definition.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from ..utils.log_utils import get_logger


class EstimatorConfig(object):
    """
    Estimator Argument is a class describes an estimator and its specific arguments, which is used to create concrete
    estimators in training.
    Every time we create a multi-grain scan layer or a cascade layer, we don't put concrete estimator instances into
    initialization parameters, but put the EstimatorConfig that describes the estimator.
    """
    def __init__(self):
        self.LOGGER = get_logger('estimator.argument')
        self.est_args = {}

    def get_est_args(self):
        """
        Get estimator argument.

        :return: estimator argument
        """
        return self.est_args.copy()


class MultiClassXGBConfig(EstimatorConfig):
    """
    Multi-class XGBoost Classifier Argument.
    """
    def __init__(self, n_folds=3, nthread=-1, booster='gbtree', scale_pos_weight=1, num_class=None, silent=True,
                 objective="multi:softprob", eval_metric="merror", eta=0.03, subsample=0.9, num_boost_round=60,
                 early_stopping_rounds=20, colsample_bytree=0.85, colsample_bylevel=0.9, max_depth=6,
                 verbose_eval=False, learning_rates=None):
        """
        Multi-class XGBoost Classifier Argument describes arguments of multi-class xgboost classifier.
        Parameters can refer to xgboost document: http://xgboost.readthedocs.io/en/latest/python/python_api.html

        :param n_folds: how many folds to execute in cross validation
        :param nthread: number of threads to execute
        :param booster: booster, default is 'gbtree'
        :param scale_pos_weight: used to handle class imbalance
        :param num_class: number of classes to classify
        :param silent:
        :param objective: objective, default is 'multi:softprob'
        :param eval_metric: evaluation metrics, default is 'merror'
        :param eta:
        :param subsample:
        :param num_boost_round:
        :param early_stopping_rounds:
        :param colsample_bytree:
        :param colsample_bylevel:
        :param max_depth:
        :param verbose_eval:
        :param learning_rates:
        """
        super(MultiClassXGBConfig, self).__init__()
        assert num_class is not None, 'You must set number of classes!'
        self.est_args = {
            'est_type': 'FLXGB',
            'n_folds': n_folds,
            'nthread': nthread,
            'booster': booster,
            'scale_pos_weight': scale_pos_weight,
            'num_class': num_class,
            'silent': silent,
            'objective': objective,
            'eval_metric': eval_metric,
            'eta': eta,
            'subsample': subsample,
            'num_boost_round': num_boost_round,
            'early_stopping_rounds': early_stopping_rounds,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'max_depth': max_depth,
            'verbose_eval': verbose_eval,
            'learning_rates': learning_rates
        }


class BinClassXGBConfig(EstimatorConfig):
    """
    Binary Class XGBoost Classifier.
    """
    def __init__(self, n_folds=3, nthread=-1, booster='gbtree', scale_pos_weight=1, num_class=2, silent=True,
                 objective="binary:logistic", eval_metric="auc", eta=0.03, subsample=0.9, num_boost_round=160,
                 early_stopping_rounds=30, colsample_bytree=0.85, colsample_bylevel=0.9, max_depth=6,
                 verbose_eval=False, learning_rates=None):
        """
        Binary-class XGBoost Classifier Argument describes arguments of multi-class xgboost classifier.
        Parameter can refer to xgboost document: http://xgboost.readthedocs.io/en/latest/python/python_api.html

        :param n_folds: how many folds to execute in cross validation
        :param nthread: number of threads to execute
        :param booster: booster, default is 'gbtree'
        :param scale_pos_weight: used to handle class imbalance
        :param num_class: number of classes to classify
        :param silent:
        :param objective: objective, default is 'multi:softprob'
        :param eval_metric: evaluation metrics, default is 'merror'
        :param eta:
        :param subsample:
        :param num_boost_round:
        :param early_stopping_rounds:
        :param colsample_bytree:
        :param colsample_bylevel:
        :param max_depth:
        :param verbose_eval:
        :param learning_rates:
        """
        super(BinClassXGBConfig, self).__init__()
        assert num_class is not None, 'You must set number of classes!'
        self.est_args = {
            'est_type': 'FLXGB',
            'n_folds': n_folds,
            'nthread': nthread,
            'booster': booster,
            'scale_pos_weight': scale_pos_weight,
            'silent': silent,
            'objective': objective,
            'eval_metric': eval_metric,
            'eta': eta,
            'subsample': subsample,
            'num_boost_round': num_boost_round,
            'early_stopping_rounds': early_stopping_rounds,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'max_depth': max_depth,
            'verbose_eval': verbose_eval,
            'learning_rates': learning_rates
        }


class RandomForestConfig(EstimatorConfig):
    """
    Random Forest Argument.
    """
    def __init__(self, n_folds=3, n_estimators=500, max_depth=100, n_jobs=-1, max_features='sqrt',
                 min_samples_leaf=1, random_state=None):
        """
        Random Forest Argument.

        :param n_folds:
        :param n_estimators:
        :param max_depth:
        :param n_jobs:
        :param max_features:
        :param min_samples_leaf:
        :param random_state:
        """
        super(RandomForestConfig, self).__init__()
        self.est_args = {
            'est_type': 'FLRF',
            'n_folds': n_folds,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'n_jobs': n_jobs,
            'max_features': max_features,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state
        }


class ExtraRandomForestConfig(EstimatorConfig):
    """
    Extremely Random Forest Argument.
    """
    def __init__(self, n_folds=3, n_estimators=500, max_depth=100, n_jobs=-1, max_features=1,
                 min_samples_leaf=1, random_state=None):
        """
        Completely Random Forest Argument, also called Extremely Random Forest.

        :param n_folds: number of folds for cross validation
        :param n_estimators: number of tree estimators
        :param max_depth: max depth of a tree
        :param n_jobs:
        :param max_features:
        :param min_samples_leaf:
        :param random_state:
        """
        super(ExtraRandomForestConfig, self).__init__()
        self.est_args = {
            'est_type': 'FLCRF',
            'n_folds': n_folds,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'n_jobs': n_jobs,
            'max_features': max_features,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state
        }


class GBDTConfig(EstimatorConfig):
    """
    Gradient Boosting Decision Tree Argument.
    """
    def __init__(self, n_folds=3, loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='auto'):
        """
        Gradient Boosting Decision Tree Argument.

        :param n_folds: how many folds to execute in cross validation
        :param loss:
        :param learning_rate:
        :param n_estimators:
        :param subsample:
        :param criterion:
        :param min_samples_split:
        :param min_samples_leaf:
        :param min_weight_fraction_leaf:
        :param max_depth:
        :param min_impurity_decrease:
        :param min_impurity_split:
        :param init:
        :param random_state:
        :param max_features:
        :param alpha:
        :param verbose:
        :param max_leaf_nodes:
        :param warm_start:
        :param presort:
        """
        super(GBDTConfig, self).__init__()
        self.est_args = {
            'est_type': 'FLGBDT',
            'n_folds': n_folds,
            'loss': loss,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'criterion': criterion,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'min_weight_fraction_leaf': min_weight_fraction_leaf,
            'max_depth': max_depth,
            'init': init,
            'subsample': subsample,
            'max_features': max_features,
            'min_impurity_decrease': min_impurity_decrease,
            'min_impurity_split': min_impurity_split,
            'random_state': random_state,
            'alpha': alpha,
            'verbose': verbose,
            'max_leaf_nodes': max_leaf_nodes,
            'warm_start': warm_start,
            'presort': presort
        }


class XGBRegressorConfig(EstimatorConfig):
    """
    Sklearn based xgboost regressor.
    """
    def __init__(self, n_folds=3,  max_depth=3, learning_rate=0.1, n_estimators=100,
                 silent=True, objective="reg:linear",
                 nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None):
        """
        XGBoost Regressor Argument describes arguments of XGBoost Regressor.
        Parameter can refer to xgboost document: http://xgboost.readthedocs.io/en/latest/python/python_api.html.

        :param n_folds: how many folds to execute in cross validation
        :param nthread: number of threads to execute
        :param scale_pos_weight: used to handle class imbalance
        :param n_estimators:
        :param silent:
        :param reg_lambda:
        :param reg_alpha:
        :param gamma:
        :param min_child_weight: Defines the minimum sum of weights of all observations required in a child.
                                 Used to control over-fitting. Higher values prevent a model from learning relations
                                  which might be highly specific to the particular sample selected for a tree.
                                 Too high values can lead to under-fitting hence, it should be tuned using CV.
        :param base_score:
        :param objective: objective, default is 'reg:linear'
        :param subsample: default=1
        :param colsample_bytree:
        :param colsample_bylevel:
        :param max_depth: default=3, The maximum depth of a tree, same as GBM.
                          Used to control over-fitting as higher depth will allow model to learn relations
                           very specific to a particular sample.
        :param learning_rate: default=0.1
        :param seed:
        """
        super(XGBRegressorConfig, self).__init__()
        self.est_args = {
            'est_type': 'FLXGB',
            'n_folds': n_folds,
            'nthread': nthread,
            'scale_pos_weight': scale_pos_weight,
            'silent': silent,
            'objective': objective,
            'max_delta_step': max_delta_step,
            'n_estimators': n_estimators,
            'reg_lambda': reg_lambda,
            'reg_alpha': reg_alpha,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'base_score': base_score,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'seed': seed,
            'missing': missing
        }


class LGBMRegressorConfig(EstimatorConfig):
    """
    Sklearn based LightGBM regressor.
    """
    def __init__(self, n_folds=3, boosting_type="gbdt", num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=100,
                 subsample_for_bin=200000, objective=None, class_weight=None,
                 min_split_gain=0., min_child_weight=1e-3, min_child_samples=20,
                 subsample=1., subsample_freq=1, colsample_bytree=1.,
                 reg_alpha=0., reg_lambda=0., random_state=None,
                 n_jobs=-1, silent=True):
        """
        Initialize FLLGBMRegressor argument.

        :param n_folds:
        :param boosting_type:
        :param num_leaves:
        :param max_depth:
        :param learning_rate:
        :param n_estimators:
        :param subsample_for_bin:
        :param objective:
        :param class_weight:
        :param min_split_gain:
        :param min_child_weight:
        :param min_child_samples:
        :param subsample:
        :param subsample_freq:
        :param colsample_bytree:
        :param reg_alpha:
        :param reg_lambda:
        :param random_state:
        :param n_jobs:
        :param silent:
        """
        super(LGBMRegressorConfig, self).__init__()
        self.est_args = {
            'est_type': 'FLLGBM',
            'n_folds': n_folds,
            'boosting_type': boosting_type,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample_for_bin': subsample_for_bin,
            'objective': objective,
            'class_weight': class_weight,
            'min_split_gain': min_split_gain,
            'min_child_weight': min_child_weight,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'subsample_freq': subsample_freq,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'silent': silent
        }


def Basic4x2(n_folds=3, n_estimators=500, max_depth=100, n_jobs=-1, min_samples_leaf=1):
    crf = ExtraRandomForestConfig(n_folds=n_folds,
                                  n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  n_jobs=n_jobs,
                                  min_samples_leaf=min_samples_leaf)
    rf = RandomForestConfig(n_folds=n_folds,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            n_jobs=n_jobs,
                            min_samples_leaf=min_samples_leaf)
    est_configs = [
        crf, crf, crf, crf,
        rf, rf, rf, rf
    ]
    return est_configs
