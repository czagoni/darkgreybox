from typing import Any, Callable, List, Optional

import pandas as pd

from darkgreybox.fit import apply_prefit_filter, train_models


def prefit_models(
    models: List[Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    error_metric: Callable,
    prefit_splits: Optional[List] = None,
    prefit_filter: Optional[Callable] = None,
    method: str = 'nelder',
    obj_func: Optional[Callable] = None,
    n_jobs: int = -1,
    verbose: int = 10
) -> List[Any]:

    if prefit_splits is None:
        return models

    prefit_df = train_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        splits=prefit_splits,
        error_metric=error_metric,
        method=method,
        obj_func=obj_func,
        n_jobs=n_jobs,
        verbose=verbose
    )

    filtered_df = apply_prefit_filter(prefit_df, prefit_filter)

    if 'model' not in filtered_df or len(filtered_df.dropna(subset=['model'])) == 0:
        raise ValueError('No valid models found during prefit')

    return filtered_df['model'].tolist()
