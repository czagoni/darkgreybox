
from typing import Any, Callable, Dict, List, Optional
import pandas as pd

from darkgreybox.predict import predict_models
from darkgreybox.prefit import prefit_models
from darkgreybox.train import train_models


def darkgreyfit(
    models: List[Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ic_params_map: Dict,
    error_metric: Callable,
    prefit_splits: Optional[List] = None,
    prefit_filter: Optional[Callable] = None,
    reduce_train_results: bool = False,
    method: str = 'nelder',
    obj_func: Optional[Callable] = None,
    n_jobs: int = -1,
    verbose: int = 10
) -> pd.DataFrame:
    """
    Given a list of `models` applies a prefit according to `prefit_splits`, then fits them
    to the training data and evaluates them on the test data.

    Params:
        models: list of `model.DarkGreyModel` objects
            list of models to be trained
        X_train: `pd.DataFrame`
            A pandas DataFrame of the training input data X
        y_train: `pd.Series`
            A pandas Series of the training input data y
        X_test: `pd.DataFrame`
            A pandas DataFrame of the test input data X
        y_test: `pd.Series`
            A pandas Series of the test input data y
        ic_params_map: Dict
            A dictionary of mapping functions that return the
            initial condition parameters for the test set
        error_metric: Callable
            An error metric function that confirms to the `sklearn.metrics` interface
        prefit_splits: Optional[List]
            A list of training data indices specifying sub-sections of `X_train` and `y_train`
            for the prefitting of models
        prefit_filter: Optional[Callable]
            A function acting as a filter based on the 'error' values of the trained models
        reduce_train_results: bool
            If set to True, the training dataframe will be reduced / cleaned
            by removing nan and duplicate records
        method : str
            Name of the fitting method to use. Valid values are described in:
            `lmfit.minimize`
        obj_func: Optional[Callable]
            The objective function to minimise during the fitting
        n_jobs: int
            The number of parallel jobs to be run as described by `joblib.Parallel`
        verbose: int
            The degree of verbosity as described by `joblib.Parallel`

    Returns:
        `pd.DataFrame` with a record for each model's potentially viable results

    """

    models_to_train = prefit_models(
        models,
        X_train,
        y_train,
        error_metric,
        prefit_splits,
        prefit_filter,
        method,
        obj_func,
        n_jobs,
        verbose
    )

    train_df = train_models(
        models=models_to_train,
        X_train=X_train,
        y_train=y_train,
        splits=None,
        error_metric=error_metric,
        method=method,
        obj_func=obj_func,
        reduce_train_results=reduce_train_results,
        n_jobs=n_jobs,
        verbose=verbose
    )

    test_df = predict_models(
        models=train_df['model'].tolist(),
        X_test=X_test,
        y_test=y_test,
        ic_params_map=ic_params_map,
        error_metric=error_metric,
        train_results=train_df['model_result'].tolist(),
        n_jobs=n_jobs,
        verbose=verbose
    )

    return pd.concat([train_df, test_df], keys=['train', 'test'], axis=1)
