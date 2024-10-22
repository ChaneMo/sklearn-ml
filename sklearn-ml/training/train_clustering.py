import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def train_clustering(data, use_normalization, models_container):
    # standardization
    if use_normalization == 'minmax':
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
    if use_normalization == 'standard':
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

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
                    data_normalize = scaler.fit_transform(data)
                    normalize_signal = 1
                if use_normalization[idx] == 'standard':
                    scaler = StandardScaler()
                    data_normalize = scaler.fit_transform(data)
                    normalize_signal = 1
        if normalize_signal == 0:
            y_pred = model.fit_predict(data)
            models_fitted.append(model)
        elif normalize_signal == 1:
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
    print('-' * 60)
    print(score_df)
    print('-' * 60)

    return models_fitted, score_df
