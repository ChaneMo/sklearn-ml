# -*- coding: utf-8 -*-
"""
@author: ChaneMo
@date: 2024-04-28
"""

# import necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error, silhouette_score, calinski_harabasz_score, davies_bouldin_score)


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
            models_container = []
            if 'lr' in model_names:
                if param!=None and 'lr' in param:
                    lr = LogisticRegression(random_state=seed, **param['lr'])
                else:
                    lr = LogisticRegression(random_state=seed)
                models_container.append(lr)
            if 'svc' in model_names:
                if param!=None and 'svc' in param:
                    svc = SVC(random_state=seed, **param['svc'])
                else:
                    svc = SVC(random_state=seed)
                models_container.append(svc)
            if 'knn_c' in model_names:
                if param!=None and 'knn_c' in param:
                    knn_c = KNeighborsClassifier(**param['knn_c'])
                else:
                    knn_c = KNeighborsClassifier()
                models_container.append(knn_c)
            if 'dt_c' in model_names:
                if param!=None and 'dt_c' in param:
                    dt_c = DecisionTreeClassifier(random_state=seed, **param['dt_c'])
                else:
                    dt_c = DecisionTreeClassifier(random_state=seed)
                models_container.append(dt_c)
            if 'gbdt_c' in model_names:
                if param!=None and 'gbdt_c' in param:
                    gbdt_c = GradientBoostingClassifier(random_state=seed, **param['gbdt_c'])
                else:
                    gbdt_c = GradientBoostingClassifier(random_state=seed)
                models_container.append(gbdt_c)
            if 'adaboost_c' in model_names:
                if param!=None and 'adaboost_c' in param:
                    adaboost_c = AdaBoostClassifier(random_state=seed, **param['adaboost_c'])
                else:
                    adaboost_c = AdaBoostClassifier(random_state=seed)
                models_container.append(adaboost_c)
            if 'rf_c' in model_names:
                if param!=None and 'rf_c' in param:
                    rf_c = RandomForestClassifier(random_state=seed, **param['rf_c'])
                else:
                    rf_c = RandomForestClassifier(random_state=seed)
                models_container.append(rf_c)
            if 'mlp_c' in model_names:
                if param!=None and 'mlp_c' in param:
                    mlp_c = MLPClassifier(random_state=seed, **param['mlp_c'])
                else:
                    mlp_c = MLPClassifier(random_state=seed)
                models_container.append(mlp_c)

        # multiclass classification models
        if mission=='multiclass_classification':
            models_container = []
            if 'knn_c' in model_names:
                if param != None and 'knn_c' in param:
                    knn_c = KNeighborsClassifier(**param['knn_c'])
                else:
                    knn_c = KNeighborsClassifier()
                models_container.append(knn_c)
            if 'dt_c' in model_names:
                if param != None and 'dt_c' in param:
                    dt_c = DecisionTreeClassifier(random_state=seed, **param['dt_c'])
                else:
                    dt_c = DecisionTreeClassifier(random_state=seed)
                models_container.append(dt_c)
            if 'gbdt_c' in model_names:
                if param != None and 'gbdt_c' in param:
                    gbdt_c = GradientBoostingClassifier(random_state=seed, **param['gbdt_c'])
                else:
                    gbdt_c = GradientBoostingClassifier(random_state=seed)
                models_container.append(gbdt_c)
            if 'adaboost_c' in model_names:
                if param != None and 'adaboost_c' in param:
                    adaboost_c = AdaBoostClassifier(random_state=seed, **param['adaboost_c'])
                else:
                    adaboost_c = AdaBoostClassifier(random_state=seed)
                models_container.append(adaboost_c)
            if 'rf_c' in model_names:
                if param != None and 'rf_c' in param:
                    rf_c = RandomForestClassifier(random_state=seed, **param['rf_c'])
                else:
                    rf_c = RandomForestClassifier(random_state=seed)
                models_container.append(rf_c)
            if 'mlp_c' in model_names:
                if param != None and 'mlp_c' in param:
                    mlp_c = MLPClassifier(random_state=seed, **param['mlp_c'])
                else:
                    mlp_c = MLPClassifier(random_state=seed)
                models_container.append(mlp_c)

        # regression models
        if mission=='regression':
            models_container = []
            if 'svr' in model_names:
                if param!=None and 'svr' in param:
                    svr = SVR(**param['svr'])
                else:
                    svr = SVR()
                models_container.append(svr)
            if 'knn_r' in model_names:
                if param!=None and 'knn_r' in param:
                    knn_r = KNeighborsRegressor(**param['knn_r'])
                else:
                    knn_r = KNeighborsRegressor()
                models_container.append(knn_r)
            if 'dt_r' in model_names:
                if param!=None and 'dt_r' in param:
                    dt_r = DecisionTreeRegressor(random_state=seed, **param['dt_r'])
                else:
                    dt_r = DecisionTreeRegressor(random_state=seed)
                models_container.append(dt_r)
            if 'gbdt_r' in model_names:
                if param!=None and 'gbdt_r' in param:
                    gbdt_r = GradientBoostingRegressor(random_state=seed, **param['ggbdt_r'])
                else:
                    gbdt_r = GradientBoostingRegressor(random_state=seed)
                models_container.append(gbdt_r)
            if 'adaboost_r' in model_names:
                if param!=None and 'adaboost_r' in param:
                    adaboost_r = AdaBoostRegressor(random_state=seed, **param['adaboost_r'])
                else:
                    adaboost_r = AdaBoostRegressor(random_state=seed)
                models_container.append(adaboost_r)
            if 'rf_r' in model_names:
                if param!=None and 'rf_r' in param:
                    rf_r = RandomForestRegressor(random_state=seed, **param['rf_r'])
                else:
                    rf_r = RandomForestRegressor(random_state=seed)
                models_container.append(rf_r)
            if 'mlp_r' in model_names:
                if param!=None and 'mlp_r' in param:
                    mlp_r = MLPRegressor(random_state=seed, **param['mlp_r'])
                else:
                    mlp_r = MLPRegressor(random_state=seed)
                models_container.append(mlp_r)

        # cluster models
        if mission=='clustering':
            models_container = []
            if 'kmeans' in model_names:
                if param!=None and 'kmeans' in param:
                    kmeans = KMeans(random_state=seed, **param['kmeans'])
                else:
                    kmeans = KMeans(random_state=seed)
                models_container.append(kmeans)
            if 'dbscan' in model_names:
                if param!=None and 'dbscan' in param:
                    dbscan = DBSCAN(**param['dbscan'])
                else:
                    dbscan = DBSCAN()
                models_container.append(dbscan)

        return models_container

    # models fitting
    def fit_models(self, models_container=None, data=[], label=[], mission=None, stratify='none', test_size=0.3, use_normalization='none'):
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
            if not label:
                print('Please make sure that your label are inputted!')
                return
            X_train, X_test, y_train, y_test = train_test_split(data, label, stratify=stratify, test_size=test_size)

            # standardization
            if use_normalization=='minmax':
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.fit_transform(X_test)
            if use_normalization=='standard':
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.fit_transform(X_test)

            # train models and return
            score_df = []
            models_fitted = []
            for idx, model in enumerate(models_container):
                normalize_signal = 0
                # custom standardization
                if use_normalization!='none':
                    if type(use_normalization)=='list':
                        if use_normalization[idx]=='minmax':
                            scaler = MinMaxScaler()
                            X_train_normalize = scaler.fit_transform(X_train)
                            X_test_normalize = scaler.fit_transform(X_test)
                            normalize_signal = 1
                        if use_normalization[idx]=='standard':
                            scaler = StandardScaler()
                            X_train_normalize = scaler.fit_transform(X_train)
                            X_test_normalize = scaler.fit_transform(X_test)
                            normalize_signal = 1
                if normalize_signal==0:
                    model.fit(X_train, y_train)
                    models_fitted.append(model)
                    y_pred = model.predict(X_test)
                elif normalize_signal==1:
                    model.fit(X_train_normalize, y_train)
                    models_fitted.append(model)
                    y_pred = model.predict(X_test_normalize)

                # binary classification metrics
                if len(set(label))==2:
                    prec = precision_score(y_test, y_pred)
                    rec = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    # print('Precision:', prec)
                    # print('Recall:', rec)
                    # print('F1 score:', f1)
                else:
                    # multiclass classification metrics
                    prec = precision_score(y_test, y_pred, average='macro')
                    rec = recall_score(y_test, y_pred, average='macro')
                    f1 = f1_score(y_test, y_pred, average='macro')
                    # print('Precision:', precision_score(y_test, y_pred, average='macro'))
                    # print('Recall:', recall_score(y_test, y_pred, average='macro'))
                    # print('F1 score:', f1_score(y_test, y_pred, average='macro'))
                score_df.append([str(model).split('(')[0], prec, rec, f1])

            # print results
            score_df = pd.DataFrame(score_df)
            score_df.columns = ['Model', 'Precision', 'Recall', 'F1']
            print('Model results:')
            print('-'*60)
            print(score_df)
            print('-'*60)

            return models_fitted, score_df

        # fit regression models
        if mission=='regression':
            if not label:
                print('Please make sure that your label are inputted!')
                return
            X_train, X_test, y_train, y_test = train_test_split(data, label, stratify=stratify, test_size=test_size)

            # standardization
            if use_normalization=='minmax':
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.fit_transform(X_test)
            if use_normalization=='standard':
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.fit_transform(X_test)

            # train models and return
            score_df = []
            models_fitted = []
            for idx, model in enumerate(models_container):
                normalize_signal = 0
                # custom standardization
                if use_normalization!='none':
                    if type(use_normalization)=='list':
                        if use_normalization[idx]=='minmax':
                            scaler = MinMaxScaler()
                            X_train_normalize = scaler.fit_transform(X_train)
                            X_test_normalize = scaler.fit_transform(X_test)
                            normalize_signal = 1
                        if use_normalization[idx]=='standard':
                            scaler = StandardScaler()
                            X_train_normalize = scaler.fit_transform(X_train)
                            X_test_normalize = scaler.fit_transform(X_test)
                            normalize_signal = 1
                if normalize_signal==0:
                    model.fit(X_train, y_train)
                    models_fitted.append(model)
                    y_pred = model.predict(X_test)
                elif normalize_signal==1:
                    model.fit(X_train_normalize, y_train)
                    models_fitted.append(model)
                    y_pred = model.predict(X_test_normalize)

                # regression metrics
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                score_df.append([str(model).split('(')[0], r2, mse, mae, mape])

            # print results
            score_df = pd.DataFrame(score_df)
            score_df.columns = ['Model', 'r2', 'mse', 'mae', 'mape']
            print('Model results:')
            print('-'*60)
            print(score_df)
            print('-'*60)

            return models_fitted, score_df

        # fit cluster models
        if mission=='clustering':
            # standardization
            if use_normalization=='minmax':
                scaler = MinMaxScaler()
                data = scaler.fit_transform(data)
            if use_normalization=='standard':
                scaler = StandardScaler()
                data = scaler.fit_transform(data)

            # train models and return
            score_df = []
            models_fitted = []
            for idx, model in enumerate(models_container):
                normalize_signal = 0
                # custom standardization
                if use_normalization!='none':
                    if type(use_normalization)=='list':
                        if use_normalization[idx]=='minmax':
                            scaler = MinMaxScaler()
                            data_normalize = scaler.fit_transform(data)
                            normalize_signal = 1
                        if use_normalization[idx]=='standard':
                            scaler = StandardScaler()
                            data_normalize = scaler.fit_transform(data)
                            normalize_signal = 1
                if normalize_signal==0:
                    y_pred = model.fit_predict(data)
                    models_fitted.append(model)
                elif normalize_signal==1:
                    y_pred = model.fit_predict(data_normalize)
                    models_fitted.append(model)

                # cluster metrics
                sil_score = silhouette_score(data, y_pred)
                ch_score = calinski_harabasz_score(data, y_pred)
                dbi_score = davies_bouldin_score(data, y_pred)
                score_df.append([str(model).split('(')[0], sil_score, ch_score, dbi_score])

            # print results
            score_df = pd.DataFrame(score_df)
            score_df.columns = ['Model', 'silhouette', 'calinski_harabasz', 'davies_bouldin']
            print('Model results:')
            print('-'*60)
            print(score_df)
            print('-'*60)

            return models_fitted, score_df


# model_comparison = model_comparison()


if __name__ == '__main__':
    import warnings
    # 或者忽略所有警告
    warnings.filterwarnings("ignore")

    from sklearn.datasets import load_iris, load_breast_cancer
    func = model_comparison()
    models_container = func.initialize_models(model_names=['knn_c', 'dt_c', 'gbdt_c', 'adaboost_c', 'rf_c', 'mlp_c'], mission='multiclass_classification', seed=2023)
    # print(models_container)
    data = load_iris()
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

