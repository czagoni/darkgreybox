from timeit import default_timer as timer
from typing import Any, Callable, Dict, List, Union, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from darkgreybox import logger
from darkgreybox.base_model import DarkGreyModel, DarkGreyModelResult


def predict_models(
    models: List[Union[DarkGreyModel, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ic_params_map: Dict,
    error_metric: Callable,
    train_results: List[DarkGreyModelResult],
    n_jobs: int = -1,
    verbose: int = 10
) -> pd.DataFrame:
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
        error_metric: Callable
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


    prefit_df = train_models(
        models=[trained_model_1, trained_model_2],
        X_test=X_test,
        y_test=y_test,
        ic_params_map={}
        error_metric=mean_squared_error,
        train_results=[trained_model_result_1, trained_model_result_2],
        n_jobs=-1,
        verbose=10
    )
    ~~~~
    """

    num_models = len(models)
    logger.info(f'Generating predictions for {num_models} models...')

    if n_jobs != 1:
        with Parallel(n_jobs=n_jobs, verbose=verbose) as p:
            df = cast(pd. DataFrame, pd.concat(
                cast(pd. DataFrame, p(delayed(predict_model)(
                    model, X_test, y_test, ic_params_map, error_metric, train_result)
                    for model, train_result in zip(models, train_results)
                )),
                ignore_index=True
            ))

    else:
        df = pd.concat([predict_model(model, X_test, y_test, ic_params_map, error_metric, train_result)
                        for model, train_result in zip(models, train_results)], ignore_index=True)

    return df


def predict_model(
    model: Union[DarkGreyModel, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ic_params_map: Dict,
    error_metric: Callable,
    train_result: DarkGreyModelResult,
) -> pd.DataFrame:
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
        error_metric: Callable
            An error metric function that confirms to the `sklearn.metrics` interface
        train_result: `model.DarkGreyModelResult`
            The model result of a previously trained model

    Returns:
        `pandas.DataFrame` with a single record for the fit model's predictions
    """

    start = timer()

    start_date = X_test.index[0]
    end_date = X_test.index[-1]

    X = X_test.to_dict(orient='list')
    y = y_test.values

    if isinstance(model, DarkGreyModel):

        ic_params = map_ic_params(ic_params_map, model, X_test, y_test, train_result)

        model_result = model.predict(X, ic_params)

        end = timer()

        return pd.DataFrame({
            'start_date': [start_date],
            'end_date': [end_date],
            'model': [model],
            'model_result': [model_result],
            'time': [end - start],
            'error': [error_metric(y, model_result.Z)]
        })

    else:
        end = timer()
        return pd.DataFrame({
            'start_date': [start_date],
            'end_date': [end_date],
            'model': [np.NaN],
            'model_result': [np.NaN],
            'time': [end - start],
            'error': [np.NaN]
        })


def map_ic_params(
    ic_params_map: Dict,
    model: DarkGreyModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_result: DarkGreyModelResult
) -> Dict:
    """
    Maps the test initial condition parameters according to `ic_params_map`

    Parameters:
        ic_params_map: Dict
            A dictionary of mapping functions that return the initial condition parameters
        model: `model.DarkGreyModel`
            model used for the prediction
        X_test: `pd.DataFrame`
            A pandas DataFrame of the test input data X
        y_test: `pd.Series`
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
