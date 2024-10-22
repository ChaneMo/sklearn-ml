from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN
import inspect


from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# 创建一个字典来映射模型名称和对应的类
model_classes = {
    'svr': SVR,
    'knn_r': KNeighborsRegressor,
    'dt_r': DecisionTreeRegressor,
    'gbdt_r': GradientBoostingRegressor,
    'adaboost_r': AdaBoostRegressor,
    'rf_r': RandomForestRegressor,
    'mlp_r': MLPRegressor
}

def initialize_model(model_name, param, seed):
    """初始化模型，如果提供了参数则使用，否则使用默认参数。"""
    if param and model_name in param:
        # 某些模型需要random_state参数
        if model_name in ['dt_r', 'gbdt_r', 'adaboost_r', 'rf_r', 'mlp_r']:
            if 'random_state' in [pname for pname, pval in inspect.signature(model_classes[model_name].__init__).parameters.items()]:
                return model_classes[model_name](random_state=seed, **param[model_name])
            else:
                return model_classes[model_name](**param[model_name])
        else:
            return model_classes[model_name](**param[model_name])
    else:
        # 某些模型需要random_state参数
        if model_name in ['dt_r', 'gbdt_r', 'adaboost_r', 'rf_r', 'mlp_r']:
            if 'random_state' in [pname for pname, pval in inspect.signature(model_classes[model_name].__init__).parameters.items()]:
                return model_classes[model_name](random_state=seed)
            else:
                return model_classes[model_name]()
        else:
            return model_classes[model_name]()

def regression(model_names, param, seed):
    models_container = [
        initialize_model(model_name, param, seed)
        for model_name in model_names
        if model_name in model_classes
    ]
    return models_container
