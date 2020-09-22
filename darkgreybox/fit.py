import numpy as np
import pandas as pd
import copy
from timeit import default_timer as timer
from joblib import Parallel, delayed


def train_models(models, X_train, y_train, error_metric,
                 splits=None, method='nelder', n_jobs=-1, verbose=10):

    if n_jobs != 1:
        with Parallel(n_jobs=n_jobs, verbose=verbose) as p:
            df = pd.concat(p(delayed(train_model)(model, X_train.loc[idx], y_train.loc[idx], error_metric, method)
                             for _, idx in splits or [(None, X_train.index)] for model in models), ignore_index=True)

    else:
        df = pd.concat([train_model(model, X_train.loc[idx], y_train.loc[idx], error_metric, method)
                         for _, idx in splits or [(None, X_train.index)] for model in models], ignore_index=True)

    return df


def train_model(base_model, X_train, y_train, error_metric, method='nelder'):

    start = timer()
    model = copy.deepcopy(base_model)
    
    try:
        model = model.fit(X=X_train,
                          y=y_train.values,
                          method=method,
                          ic_params=get_ic_params(model, X_train))
    except ValueError:
        end = timer()
        return pd.DataFrame({'start_date': [X_train.index[0]],
                             'end_date': [X_train.index[-1]],
                             'model': [np.NaN],
                             'model_result': [np.NaN],
                             'time': [end - start],
                             'method': [method],
                             'error': [np.NaN]})

    model_result = model.predict(X_train)
    end = timer()

    return pd.DataFrame({'start_date': [X_train.index[0]],
                         'end_date': [X_train.index[-1]],
                         'model': [model],
                         'model_result': [model_result],
                         'time': [end - start],
                         'method': [method],
                         'error': [error_metric(y_train.values, model_result.Z)]})


def get_ic_params(model, X_train):

    # TODO: make this more robust
    ic_params = {}
    for key in model.params.keys():
        if '0' in key:
            ic_params[key] = X_train.iloc[0][key]

    return ic_params
