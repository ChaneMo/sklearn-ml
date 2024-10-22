from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN
import inspect


# 创建一个字典来映射模型名称和对应的类
model_classes = {
    'lr': LogisticRegression,
    'svc': SVC,
    'knn_c': KNeighborsClassifier,
    'dt_c': DecisionTreeClassifier,
    'gbdt_c': GradientBoostingClassifier,
    'adaboost_c': AdaBoostClassifier,
    'rf_c': RandomForestClassifier,
    'mlp_c': MLPClassifier
}

def initialize_model(model_name, param, seed):
    """初始化模型，如果提供了参数则使用，否则使用默认参数。"""
    if param and model_name in param:
        if 'random_state' in [pname for pname, pval in inspect.signature(model_classes[model_name].__init__).parameters.items()]:
            return model_classes[model_name](random_state=seed, **param[model_name])
        else:
            return model_classes[model_name](**param[model_name])
    else:
        if 'random_state' in [pname for pname, pval in inspect.signature(model_classes[model_name].__init__).parameters.items()]:
            return model_classes[model_name](random_state=seed)
        else:
            return model_classes[model_name]()

def biClassification(model_names, param, seed):
    models_container = [
        initialize_model(model_name, param, seed)
        for model_name in model_names
        if model_name in model_classes
    ]
    return models_container
