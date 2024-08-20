# sklearn-ml
This project is a description for the pypi package "sklearn-ml", which is for machine learning missions.
# installation
    pip install sklearn-ml
# usage
## import package
    from sklearn_ml import model_comparison
    # This will tell you how to use the package.
    mc = model_comparison()
## initialize models from sklearn
sklearn-ml supports missions including "binary_classification", "multiclass_classification", "regression" and "clustering", you could choose your mission and models by using the following sample code:

    models_container = mc.initialize_models(model_names=['knn_c', 'dt_c', 'gbdt_c'], mission='multiclass_classification', seed=2023)

The function returns initialized models that in model_names, you could also customize the model parameters by use parameter "param={model_name:model_parameters}", for example:

    params = {'knn_c':{'n_neighbors':15}, 'dt_c':{'max_depth':4}}
    models_container = mc.initialize_models(model_names=['knn_c', 'dt_c', 'gbdt_c'], mission='multiclass_classification', param=params, seed=2023)
## train models
You could train models that are already initiialized in initialized_models by using the following code:

    fitted_models, test_results = mc.fit_models(data=X, label=y, models_container=models_container, mission='multiclass_classification')

The function returns the trained models and the test result after using the sklearn function train_test_split() to train and test. Besides, the label y must be categorical, such as [0, 1, 1, 0, ......].

You could also use normalization by setting the "use_normalization" parameter: "use_normalization=minmax" represents MinMaxScaler() for all models, "use_normalization=standard" represents StandardScaler() for all models, a list such as "use_normalization=['minmax', 'standard', 'none', ...]" for models customization is also available.
# example
Here, we use iris dataset from sklearn as an example:

    from sklearn.datasets import load_iris
    from sklearn_ml import model_comparison
    mc = model_comparison()
    models_container = mc.initialize_models(model_names=['knn_c', 'dt_c', 'gbdt_c'], mission='multiclass_classification', seed=2023)
    # print(models_container)
    data = load_iris()
    X = data.data
    y = data.target
    models_fitted, result_df = mc.fit_models(data=X, label=y, stratify=y, test_size=0.2, models_container=models_container, mission='multiclass_classification')
Once you import sklearn_ml, it will show the usage tips:

![1697384201769](https://github.com/ChaneMo/sklearn-ml/assets/91654630/bd1de6bd-8308-4429-a021-4059871b1bfd)
We use train_test_split() function to devide training set and testing set. After training the models, we test on the testing set and print the test results

![1697384586237](https://github.com/ChaneMo/sklearn-ml/assets/91654630/73186cb5-5f30-4864-b323-40fecd7a9142)
