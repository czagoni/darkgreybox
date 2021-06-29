from darkgreybox.predict import predict_models

import numpy as np
import pandas as pd

from darkgreybox.prefit import prefit_models
from darkgreybox.train import train_models


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
        n_jobs=n_jobs,
        verbose=verbose
    )

    if reduce_train_results:
        train_df = reduce_results_df(train_df)

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
