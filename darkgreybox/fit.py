import numpy as np
import pandas as pd
import copy
from timeit import default_timer as timer


def train_model(base_model, X_train, y_train, method, error_metric):

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