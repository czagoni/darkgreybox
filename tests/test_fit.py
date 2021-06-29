import datetime as dt
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from sklearn.metrics import mean_squared_error  # TODO: remove this dep

from darkgreybox.fit import (
    darkgreyfit,
    reduce_results_df,
)
from darkgreybox.models import Ti


def rmse(*args, **kwargs):
    return mean_squared_error(*args, **kwargs) ** 0.5


class TwoDarkGreyFitTest(unittest.TestCase):

    def test__darkgreyfit__returns_correct_dataframe__when_prefit_splits_specified(self):

        train_start = dt.datetime(2021, 1, 1, 1, 0)
        train_end = dt.datetime(2021, 1, 1, 6, 0)
        test_start = dt.datetime(2021, 1, 1, 7, 0)
        test_end = dt.datetime(2021, 1, 1, 9, 0)

        rec_duration = 1

        params = {
            'Ti0': {'value': 10, 'vary': False},
            'Ria': {'value': 1},
            'Ci': {'value': 1},
        }

        y_train = pd.Series([10, 10, 20, 20, 20, 30])

        X_train = pd.DataFrame(
            index=pd.date_range(train_start, train_end, freq=f'{rec_duration}H'),
            data={
                'Ta': [10, 10, 10, 20, 20, 20],
                'Ph': [0, 10, 0, 0, 10, 0],
                'Ti0': [10, 10, 20, 20, 20, 30]
            })

        X_test = pd.DataFrame(
            index=pd.date_range(test_start, test_end, freq=f'{rec_duration}H'),
            data={
                'Ta': [30, 30, 30],
                'Ph': [0, 10, 0],
            })

        y_test = pd.Series([30, 30, 40])

        models = [Ti(params, rec_duration)]

        ic_params_map = {
            'Ti0': lambda X_test, y_test, train_result: y_test.iloc[0],
        }

        error_metric = rmse

        prefit_splits = [([], [0, 1, 2]), ([], [3, 4, 5])]

        expected_columns = pd.MultiIndex.from_tuples([
            ('train', 'start_date'),
            ('train', 'end_date'),
            ('train', 'model'),
            ('train', 'model_result'),
            ('train', 'time'),
            ('train', 'method'),
            ('train', 'error'),
            ('test', 'start_date'),
            ('test', 'end_date'),
            ('test', 'model'),
            ('test', 'model_result'),
            ('test', 'time'),
            ('test', 'error')
        ])

        actual_df = darkgreyfit(
            models=models,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            ic_params_map=ic_params_map,
            error_metric=error_metric,
            prefit_splits=prefit_splits,
        )

        # assert on the returned dataframe being sane

        self.assertTrue(expected_columns.equals(actual_df.columns))

        self.assertListEqual([train_start], actual_df[('train', 'start_date')].tolist())
        self.assertListEqual([train_end], actual_df[('train', 'end_date')].tolist())
        self.assertListEqual([test_start], actual_df[('test', 'start_date')].tolist())
        self.assertListEqual([test_end], actual_df[('test', 'end_date')].tolist())

        self.assertIsInstance(actual_df[('train', 'model')].iloc[0], Ti)
        self.assertIsInstance(actual_df[('test', 'model')].iloc[0], Ti)

        assert_allclose(y_train.values, actual_df[('train', 'model_result')].iloc[0].Z, atol=0.01)
        assert_allclose(y_test.values, actual_df[('test', 'model_result')].iloc[0].Z, atol=0.01)


class FitTest(unittest.TestCase):

    def setUp(self):

        self.X_train = pd.DataFrame({
            'A0': [10, 20],
            'B': [30, 40],
            'C0': [50, 60],
            'D': [70, 80]
        })

        self.y_train = pd.Series([100, 110])
        self.Z_train = np.array([120, 130])

        self.X_test = pd.DataFrame({
            'A0': [10, 20],
            'B': [30, 40],
            'C0': [50, 60],
            'D': [70, 80]
        })

        self.y_test = pd.Series([100, 110])
        self.Z_test = np.array([120, 130])

    def test__reduce_results_df(self):

        df = pd.DataFrame(data={
            'value': [0, 0, 0, 10, 20, 30, 40, 50],
            'error': [np.nan, -np.inf, np.inf, 2.0000011, 2.0000012, 2.000002, 1, 3],
            'time': [0, 0, 0, 2, 1, 3, 4, 5]
        })

        expected_df = pd.DataFrame(data={
            'value': [40, 20, 30, 50],
            'error': [1, 2.0000012, 2.000002, 3],
            'time': [4, 1, 3, 5]
        })

        actual_df = reduce_results_df(df)

        assert_frame_equal(expected_df, actual_df)
