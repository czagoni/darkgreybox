import numpy as np
import pandas as pd
import copy
from timeit import default_timer as timer
from joblib import Parallel, delayed

from darkgreybox import logger
from darkgreybox.base_model import DarkGreyModel


def darkgreyfit(models, X_train, y_train, X_test, y_test, ic_params_map, error_metric,
                prefit_splits=None, prefit_filter=None, reduce_train_results=True,
                method='nelder', obj_func=None, n_jobs=-1, verbose=10):
    """
    Given a list of `models` applies a prefit according to `prefit_splits`, then fits them
    to the training data and evaluates them on the test data.

    Params:
        models: list of `model.DarkGreyModel` objects
            list of models to be trained
        X_train: `pandas.DataFrame`
            A pandas DataFrame of the training input data X
        y_train: `pandas.Series`
            A pandas Series of the training input data y
        X_test: `pandas.DataFrame`
            A pandas DataFrame of the test input data X
        y_test: `pandas.Series`
            A pandas Series of the test input data y
        ic_params_map: dict
            A dictionary of mapping functions that return the
            initial condition parameters for the test set
        error_metric: function
            An error metric function that confirms to the `sklearn.metrics` interface
        prefit_splits: list
            A list of training data indices specifying sub-sections of `X_train` and `y_train`
            for the prefitting of models
        prefit_filter: function
            A function acting as a filter based on the 'error' values of the trained models
        reduce_train_results: bool
            If set to True, the training dataframe will be reduced / cleaned
            by removing nan and duplicate records
        method : str
            Name of the fitting method to use. Valid values are described in:
            `lmfit.minimize`
        obj_func: function
            The objective function to minimise during the fitting
        n_jobs: int
            The number of parallel jobs to be run as described by `joblib.Parallel`
        verbose: int
            The degree of verbosity as described by `joblib.Parallel`

    Returns:
        `pandas.DataFrame` with a record for each model's potentially viable results

    """

    prefit_df = train_models(models=models,
                             X_train=X_train,
                             y_train=y_train,
                             splits=prefit_splits,
                             error_metric=error_metric,
                             method=method,
                             obj_func=obj_func,
                             n_jobs=n_jobs,
                             verbose=verbose)

    if prefit_filter is not None:
        prefit_df = apply_prefit_filter(prefit_df, prefit_filter)

    train_df = train_models(models=prefit_df['model'].tolist(),
                            X_train=X_train,
                            y_train=y_train,
                            splits=None,
                            error_metric=error_metric,
                            method=method,
                            obj_func=obj_func,
                            n_jobs=n_jobs,
                            verbose=verbose)

    if reduce_train_results:
        train_df = reduce_results_df(train_df)

    test_df = predict_models(models=train_df['model'].tolist(),
                             X_test=X_test,
                             y_test=y_test,
                             ic_params_map=ic_params_map,
                             error_metric=error_metric,
                             train_results=train_df['model_result'].tolist(),
                             n_jobs=n_jobs,
                             verbose=verbose)

    return pd.concat([train_df, test_df], keys=['train', 'test'], axis=1)


def train_models(models, X_train, y_train, error_metric,
                 splits=None, method='nelder', obj_func=None, n_jobs=-1, verbose=10):
    """
    Trains the `models` for the given `X_train` and `y_train` training data
    for `splits` using `method`.

    Params:
        models: list of `model.DarkGreyModel` objects
            list of models to be trained
        X_train: `pandas.DataFrame`
            A pandas DataFrame of the training input data X
        y_train: `pandas.Series`
            A pandas Series of the training input data y
        error_metric: function
            An error metric function that confirms to the `sklearn.metrics` interface
        splits: list
            A list of training data indices specifying sub-sections of `X_train` and `y_train`
            for the models to be trained on
        method : str
            Name of the fitting method to use. Valid values are described in:
            `lmfit.minimize`
        obj_func: function
            The objective function to minimise during the fitting
        n_jobs: int
            The number of parallel jobs to be run as described by `joblib.Parallel`
        verbose: int
            The degree of verbosity as described by `joblib.Parallel`

    Returns:
        `pandas.DataFrame` with a record for each model's result for each split

    Example:
    ~~~~

    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold

    from darkgreybox.model import TiTe
    from darkgreybox.fit import train_models


    prefit_df = train_models(models=[TiTe(train_params, rec_duration=1)],
                             X_train=X_train,
                             y_train=y_train,
                             splits=KFold(n_splits=int(len(X_train) / 24), shuffle=False).split(X_train),
                             error_metric=mean_squared_error,
                             method='nelder',
                             n_jobs=-1,
                             verbose=10)
    ~~~~
    """

    logger.info('Training models...')

    if n_jobs != 1:
        with Parallel(n_jobs=n_jobs, verbose=verbose) as p:
            df = pd.concat(p(delayed(train_model)
                             (model, X_train.iloc[idx], y_train.iloc[idx], error_metric, method, obj_func)
                             for _, idx in splits or [(None, range(len(X_train)))] for model in models),
                           ignore_index=True)

    else:
        df = pd.concat([train_model(model, X_train.iloc[idx], y_train.iloc[idx], error_metric, method, obj_func)
                        for _, idx in splits or [(None, range(len(X_train)))] for model in models],
                       ignore_index=True)

    return df


def train_model(base_model, X_train, y_train, error_metric, method='nelder', obj_func=None):
    """
    Trains a copy of `basemodel` for the given `X_train` and `y_train` training data
    using `method`.

    Params:
        base_model: `model.DarkGreyModel`
            model to be trained (a copy will be made)
        X_train: `pandas.DataFrame`
            A pandas DataFrame of the training input data X
        y_train: `pandas.Series`
            A pandas Series of the training input data y
        error_metric: function
            An error metric function that confirms to the `sklearn.metrics` interface
        method : str
            Name of the fitting method to use. Valid values are described in:
            `lmfit.minimize`
        obj_func: function
            The objective function to minimise during the fitting

    Returns:
        `pandas.DataFrame` with a single record for the fit model's result
    """

    start = timer()
    model = copy.deepcopy(base_model)

    try:
        model = model.fit(X=X_train.to_dict(orient='list'),
                          y=y_train.values,
                          method=method,
                          ic_params=get_ic_params(model, X_train),
                          obj_func=obj_func)
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


def predict_models(models, X_test, y_test, ic_params_map, error_metric, train_results,
                   n_jobs=-1, verbose=10):
    """
    Generates the predictions for the `models` for the given `X_test` and `y_test` test data

    Params:
        models: list of `model.DarkGreyModel` objects
            list of models to be used for the predictions
        X_test: `pandas.DataFrame`
            A pandas DataFrame of the test input data X
        y_test: `pandas.Series`
            A pandas Series of the test input data y
        ic_params_map: dict
            A dictionary of mapping functions that return the initial condition parameters
        error_metric: function
            An error metric function that confirms to the `sklearn.metrics` interface
        train_results: list of `model.DarkGreyModelResult`
            The model results of the previously trained models
        n_jobs: int
            The number of parallel jobs to be run as described by `joblib.Parallel`
        verbose: int
            The degree of verbosity as described by `joblib.Parallel`

    Returns:
        `pandas.DataFrame` with a record for each model's predictions

    Example:
    ~~~~

    from sklearn.metrics import mean_squared_error

    from darkgreybox.fit import test_models


    prefit_df = train_models(models=[trained_model_1, trained_model_2],
                             X_test=X_test,
                             y_test=y_test,
                             ic_params_map={}
                             error_metric=mean_squared_error,
                             train_results=[trained_model_result_1, trained_model_result_2],
                             n_jobs=-1,
                             verbose=10)
    ~~~~
    """

    num_models = len(models)
    logger.info(f'Generating predictions for {num_models} models...')

    if n_jobs != 1:
        with Parallel(n_jobs=n_jobs, verbose=verbose) as p:
            df = pd.concat(p(delayed(predict_model)(model, X_test, y_test, ic_params_map, error_metric, train_result)
                             for model, train_result in zip(models, train_results)), ignore_index=True)

    else:
        df = pd.concat([predict_model(model, X_test, y_test, ic_params_map, error_metric, train_result)
                        for model, train_result in zip(models, train_results)], ignore_index=True)

    return df


def predict_model(model, X_test, y_test, ic_params_map, error_metric, train_result):
    """
    Calculates redictions  of `model` for the given `X_test` and `y_test` test data.

    Params:
        model: `model.DarkGreyModel`
            model used for the prediction
        X_test: `pandas.DataFrame`
            A pandas DataFrame of the test input data X
        y_test: `pandas.Series`
            A pandas Series of the test input data y
        ic_params_map: dict
            A dictionary of mapping functions that return the initial condition parameters
        error_metric: function
            An error metric function that confirms to the `sklearn.metrics` interface
        train_result: `model.DarkGreyModelResult`
            The model result of a previously trained model

    Returns:
        `pandas.DataFrame` with a single record for the fit model's predictions
    """

    start = timer()

    if isinstance(model, DarkGreyModel):

        ic_params = map_ic_params(ic_params_map, model, X_test, y_test, train_result)

        model_result = model.predict(X=X_test.to_dict(orient='list'),
                                     ic_params=ic_params)

        end = timer()

        return pd.DataFrame({
            'start_date': [X_test.index[0]],
            'end_date': [X_test.index[-1]],
            'model': [model],
            'model_result': [model_result],
            'time': [end - start],
            'error': [error_metric(y_test.values, model_result.Z)]
        })

    else:
        end = timer()
        return pd.DataFrame({
            'start_date': [X_test.index[0]],
            'end_date': [X_test.index[-1]],
            'model': [np.NaN],
            'model_result': [np.NaN],
            'time': [end - start],
            'error': [np.NaN]
        })


def reduce_results_df(df, decimals=6):
    """
    Reduces `df` dataframe by removing nan and duplicate records

    Params:
        df: `pandas.DataFrame`
            The dataframe to be reduced / cleaned
        decimal: int
            The number of decimal points for the float comparison when removing duplicates

    Returns :
        the reduced / cleaned `pandas.DataFrame`
    """

    return (df.replace([-np.inf, np.inf], np.nan)
              .dropna()
              .round({'error': decimals})
              .sort_values('time')
              .drop_duplicates(subset=['error'], keep='first')
              .sort_values('error')
              .reset_index(drop=True))


def apply_prefit_filter(prefit_df, prefit_filter):
    """
    Applies the prefit filter to the prefit dataframe
    """
    return prefit_df[prefit_filter(prefit_df['error'])].reset_index(drop=True)


def map_ic_params(ic_params_map, model, X_test, y_test, train_result):
    """
    Maps the test initial condition parameters according to `ic_params_map`

    Parameters:
        ic_params_map: dict
            A dictionary of mapping functions that return the initial condition parameters
        model: `model.DarkGreyModel`
            model used for the prediction
        X_test: `pandas.DataFrame`
            A pandas DataFrame of the test input data X
        y_test: `pandas.Series`
            A pandas Series of the test input data y
        train_results: `model.DarkGreyModelResult`
            model result object for training data

    Returns the initial conditions parameters dict

    ~~~~
    Assuming y_test holds the internal temperatures `Ti`

    ic_params_map = {
        'Ti0': lambda X_test, y_test, train_result: y_test.iloc[0],
        'Th0': lambda X_test, y_test, train_result: y_test.iloc[0],
        'Te0': lambda X_test, y_test, train_result: train_result.Te[-1],
    }

    will map the first internal temperature in the test set to both `Ti0` and `Th0`
    and the last `Te` value from the training results to `Te0`
    ~~~~
    """

    ic_params = {}

    for key in ic_params_map:
        if key in model.params:
            ic_params[key] = ic_params_map[key](X_test, y_test, train_result)

    return ic_params


def get_ic_params(model, X_train):
    """
    Returns the initial condition parameters of a model from the training data

    Params:
        model: `model.DarkGreyModel`
            model to get initial condition parameters from
        X_train: `pandas.DataFrame`
            A pandas DataFrame of the training input data X

    Returns:
        A dictionary containing the initial conditions and their corresponding values
        as defined by the training data

    """

    # TODO: this is horrible - make this clearer and more robust
    ic_params = {}
    for key in model.params:
        if '0' in key:
            ic_params[key] = X_train.iloc[0][key]

    return ic_params
