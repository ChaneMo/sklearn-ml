# -*- coding: utf-8 -*-
"""
@author: ChaneMo
@date: 2024-10-22
"""

# import necessary libraries
from utils.binary_classification import biClassification
from utils.multi_classification import multiClassification
from utils.regression import regression
from utils.clustering import cluster
from training.train_classification import train_classification
from training.train_regression import train_regression
from training.train_clustering import train_clustering


__all__ = ['model_comparison']
# define class
class model_comparison():
    def __init__(self):
        # missions and models for choose
        self.binary_classification_models = ['lr', 'svc', 'knn_c', 'dt_c', 'gbdt_c', 'adaboost_c', 'rf_c', 'mlp_c']
        self.multiclass_classification_models = ['knn_c', 'dt_c', 'gbdt_c', 'adaboost_c', 'rf_c', 'mlp_c']
        self.regression_models = ['svr', 'knn_r', 'dt_r', 'gbdt_r', 'adaboost_r', 'rf_r', 'mlp_r']
        self.cluster_models = ['kmeans', 'dbscan']
        print('-'*130)
        print('Missions for choose:')
        print(['binary_classification', 'multiclass_classification', 'regression', 'clustering'], sep='')
        print('default=multiclass_classification')
        print('-' * 130)
        print('Model for Missions:')
        print('Binary classification models:', self.binary_classification_models)
        print('Multiclass classification models:', self.multiclass_classification_models)
        print('Regression models:', self.regression_models)
        print('Clustering models:', self.cluster_models)
        print('-' * 130)
        print('functions:')
        print('initialize_models(self, model_names=None, mission=None, seed=None, param=None)')
        print("fit_models(self, models_container=None, data=[], label=[], stratify=None, mission=None,  test_size=0.3, use_normalization='none')")
        print('-' * 130)

    # define function to initialize models
    def initialize_models(self, model_names=None, mission=None, seed=None, param=None):
        '''
        :param model_names: models for mission, only support models in initialize lists
        :param mission: binary_classification, classification_classification, regression or clustering
        :param seed: random seed
        :param param: custom params for specific models, dict type, the key must be strings in initialize lists, values are the params for the specific
                    models, params that could be used for a model are the same as scikit-learn's definition
        :return: initialized models
        '''
        if not model_names:
            print('Please choose models!')
            return
        if not mission:
            print('Please choose a mission!')
            return

        # binary classification models
        if mission=='binary_classification':
            models_container =  biClassification(model_names, param, seed)

        # multiclass classification models
        if mission=='multiclass_classification':
            models_container = multiClassification(model_names, param, seed)

        # regression models
        if mission=='regression':
            models_container = regression(model_names, param, seed)

        # cluster models
        if mission=='clustering':
            models_container = cluster(model_names, param, seed)

        return models_container

    # models fitting
    def fit_models(self, models_container=None, data=[], label=[], mission=None, stratify=None, test_size=0.3, use_normalization='none'):
        '''
        :param models_container: scikit-learn models initialized by function initialize_models() or by user
        :param data: data for model fit
        :param label: label for model fit, one dimensional list type
        :param mission: binary_classification, classification_classification, regression or clustering
        :param test_size: test set size for train_test_split function
        :param use_normalization: normalization for data, could be 'minmax', 'standard' or a list that define the normalization type
                                for each model
        :return: models fitted
        '''

        if not mission:
            print('Please choose your mission!')
            return
        if not models_container:
            print('Please input your models!')
            return
        data = list(data)
        label = list(label)
        if (not data and not label) or (not data):
            print('Please input your data and label!')
            return


        # fit classification models
        if mission=='multiclass_classification' or mission=='binary_classification':
            models_fitted, score_df = train_classification(data, label, use_normalization, stratify, test_size, models_container)

            return models_fitted, score_df

        # fit regression models
        if mission=='regression':
            models_fitted, score_df = train_regression(data, label, use_normalization, stratify, test_size, models_container)

            return models_fitted, score_df

        # fit cluster models
        if mission=='clustering':
            models_fitted, score_df = train_clustering(data, use_normalization, models_container)

            return models_fitted, score_df


# model_comparison = model_comparison()


if __name__ == '__main__':
    import warnings
    # 或者忽略所有警告
    warnings.filterwarnings("ignore")

    from sklearn.datasets import load_iris, load_breast_cancer
    func = model_comparison()
    models_container = func.initialize_models(model_names=['knn_c', 'dt_c'], mission='multiclass_classification', seed=2023)
    # print(models_container)
    data = load_iris()
    # data = load_breast_cancer()
    X = data.data
    y = data.target
    # print(X.shape)
    # print(y.shape)
    models, result = func.fit_models(data=X, label=y, models_container=models_container, mission='multiclass_classification')
    # print(models)
    # m1 = models[0]
    # print(m1)
    # data = load_iris()
    # y_pred = m1.predict(X)
    # print(precision_score(y, y_pred, average='macro'))


