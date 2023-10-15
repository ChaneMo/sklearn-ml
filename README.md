# sklearn-ml
This project is a description for the pypi package "sklearn-ml", which is for machine learning missions.
# installation
    pip install sklearn-ml
# usage
## import package
    import sklearn_ml.model_comparison as mc
## initialize models from sklearn
sklearn-ml supports missions including "binary_classification", "multiclass_classification", "regression" and "clustering", you could choose your mission and models by using the following sample code:

    initialized_models = mc.initialize_models(model_names=['knn_c', 'dt_c', 'gbdt_c'], mission='multiclass_classification', seed=2023)

The function returns initialized models that in model_names, you could also customize the parameters for each model as a dict and send them to the function by use parameter "param={model_name:model_parameters}"
## train models
You could train models that are already initiialized in initialized_models by using the following code:

    fitted_models, test_results = fit_models(data=X, label=y, models_container=models_container, mission='multiclass_classification')

The function returns the models that trained on your data and the test result after using the sklearn function train_test_split() to train and test these models. Besides, the label y must be categorical, such as [0, 1, 1, 0, ......].

By using the fit_models function, it will print the test results automatically.
# example
Here, we use iris dataset from sklearn as an example:

    from sklearn.datasets import load_iris
    import sklearn_ml.model_comparison as mc 
    models_container = mc.initialize_models(model_names=['knn_c', 'dt_c', 'gbdt_c'], mission='multiclass_classification', seed=2023)
    # print(models_container)
    data = load_iris()
    X = data.data
    y = data.target
    models_fitted, result_df = mc.fit_models(data=X, label=y, models_container=models_container, mission='multiclass_classification')
Once you import sklearn_ml, it will show the usage tips:

![1697384201769](https://github.com/ChaneMo/sklearn-ml/assets/91654630/bd1de6bd-8308-4429-a021-4059871b1bfd)
We use train_test_split() function to devide training set and testing set, after training the models, we test them by using testing set and print the test results

![1697384586237](https://github.com/ChaneMo/sklearn-ml/assets/91654630/73186cb5-5f30-4864-b323-40fecd7a9142)
