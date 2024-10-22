from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


def train_classification(data, label, use_normalization, stratify, test_size, models_container):
    if not label:
        print('Please make sure that your label are inputted!')
        return
    X_train, X_test, y_train, y_test = train_test_split(data, label, stratify=stratify, test_size=test_size)

    # standardization
    if use_normalization == 'minmax':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    if use_normalization == 'standard':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # train models and return
    score_df = []
    models_fitted = []
    for idx, model in enumerate(models_container):
        normalize_signal = 0
        # custom standardization
        if use_normalization != 'none':
            if type(use_normalization) == 'list':
                if use_normalization[idx] == 'minmax':
                    scaler = MinMaxScaler()
                    X_train_normalize = scaler.fit_transform(X_train)
                    X_test_normalize = scaler.transform(X_test)
                    normalize_signal = 1
                if use_normalization[idx] == 'standard':
                    scaler = StandardScaler()
                    X_train_normalize = scaler.fit_transform(X_train)
                    X_test_normalize = scaler.transform(X_test)
                    normalize_signal = 1
        if normalize_signal == 0:
            model.fit(X_train, y_train)
            models_fitted.append(model)
            y_pred = model.predict(X_test)
        elif normalize_signal == 1:
            model.fit(X_train_normalize, y_train)
            models_fitted.append(model)
            y_pred = model.predict(X_test_normalize)

        # binary classification metrics
        if len(set(label)) == 2:
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
    print('-' * 60)
    print(score_df)
    print('-' * 60)

    return models_fitted, score_df
