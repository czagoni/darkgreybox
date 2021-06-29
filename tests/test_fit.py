import datetime as dt
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from sklearn.metrics import mean_squared_error

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

        columns = pd.MultiIndex.from_tuples([
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

        self.assertTrue(columns.equals(actual_df.columns))

        self.assertListEqual([train_start], actual_df[('train', 'start_date')].tolist())
        self.assertListEqual([train_end], actual_df[('train', 'end_date')].tolist())
        self.assertListEqual([test_start], actual_df[('test', 'start_date')].tolist())
        self.assertListEqual([test_end], actual_df[('test', 'end_date')].tolist())

        self.assertIsInstance(actual_df[('train', 'model')].iloc[0], Ti)
        self.assertIsInstance(actual_df[('test', 'model')].iloc[0], Ti)

        assert_allclose(y_train.values, actual_df[('train', 'model_result')].iloc[0].Z, atol=0.01)
        assert_allclose(y_test.values, actual_df[('test', 'model_result')].iloc[0].Z, atol=0.01)


# @patch('darkgreybox.prefit.apply_prefit_filter')
# @patch('darkgreybox.fit.reduce_results_df')
# @patch('darkgreybox.fit.predict_models')
# @patch('darkgreybox.train.train_models')
# class DarkGreyFitTest(unittest.TestCase):

#     def setUp(self):

#         self.X_train = pd.DataFrame({
#             'A0': [10, 20],
#             'B': [30, 40],
#             'C0': [50, 60],
#             'D': [70, 80]
#         })

#         self.y_train = pd.Series([100, 110])
#         self.Z_train = np.array([120, 130])

#         self.X_test = pd.DataFrame({
#             'A0': [10, 20],
#             'B': [30, 40],
#             'C0': [50, 60],
#             'D': [70, 80]
#         })

#         self.y_test = pd.Series([100, 110])
#         self.Z_test = np.array([120, 130])

#         self.models = [MagicMock()]
#         self.train_model_results = [MagicMock()]
#         self.test_model_results = [MagicMock()]
#         self.train_time = [1.0]
#         self.test_time = [0.0]
#         self.train_error = [0.9]
#         self.test_error = [0.8]
#         self.ic_params_map = {'param': 'value'}
#         self.error_metric = MagicMock()
#         self.prefit_splits = MagicMock()
#         self.prefit_filter = lambda x: x < 1,

#         self.train_df = pd.DataFrame({
#             'model': self.models,
#             'model_result': self.train_model_results,
#             'start': self.X_train.index[0],
#             'end': self.X_train.index[-1],
#             'time': self.train_time,
#             'error': self.train_error
#         })

#         self.test_df = pd.DataFrame({
#             'model': self.models,
#             'model_result': self.test_model_results,
#             'start': self.X_test.index[0],
#             'end': self.X_test.index[-1],
#             'time': self.test_time,
#             'error': self.test_error
#         })

#         self.columns = pd.MultiIndex.from_tuples([
#             ('train', 'model'),
#             ('train', 'model_result'),
#             ('train', 'start'),
#             ('train', 'end'),
#             ('train', 'time'),
#             ('train', 'error'),
#             ('test', 'model'),
#             ('test', 'model_result'),
#             ('test', 'start'),
#             ('test', 'end'),
#             ('test', 'time'),
#             ('test', 'error')
#         ])

#     def test__darkgreyfit__correct_data_flow(
#             self,
#             mock_train_models,
#             mock_predict_models,
#             mock_reduce_results_df,
#             mock_apply_prefit_filter):

#         mock_obj_func = MagicMock()

#         mock_train_models.return_value = self.train_df
#         mock_predict_models.return_value = self.test_df

#         expected_df = pd.DataFrame(
#             columns=self.columns,
#             data=[[
#                 self.models[0], self.train_model_results[0], self.X_train.index[0],
#                 self.X_train.index[-1], self.train_time[0], self.train_error[0],
#                 self.models[0], self.test_model_results[0], self.X_test.index[0],
#                 self.X_test.index[-1], self.test_time[0], self.test_error[0]
#             ]]
#         )

#         actual_df = darkgreyfit(
#             models=self.models,
#             X_train=self.X_train,
#             y_train=self.y_train,
#             X_test=self.X_test,
#             y_test=self.y_test,
#             ic_params_map=self.ic_params_map,
#             error_metric=self.error_metric,
#             prefit_splits=self.prefit_splits,
#             prefit_filter=None,
#             reduce_train_results=False,
#             method='nelder',
#             obj_func=mock_obj_func,
#             n_jobs=-1,
#             verbose=10
#         )

#         mock_train_calls = mock_train_models.call_args_list

#         mock_train_calls[0] == ({
#             'models': self.models,
#             'X_train': self.X_train,
#             'y_train': self.y_train,
#             'splits': self.prefit_splits,
#             'error_metric': self.error_metric,
#             'method': 'nelder',
#             'obj_func': mock_obj_func,
#             'n_jobs': -1,
#             'verbose': 10
#         })

#         mock_train_calls[1] == ({
#             'models': self.models,
#             'X_train': self.X_train,
#             'y_train': self.y_train,
#             'splits': None,
#             'error_metric': self.error_metric,
#             'method': 'nelder',
#             'obj_func': mock_obj_func,
#             'n_jobs': -1,
#             'verbose': 10
#         })

#         mock_reduce_results_df.assert_not_called()

#         mock_predict_models.assert_called_with(
#             models=self.models,
#             X_test=self.X_test,
#             y_test=self.y_test,
#             ic_params_map=self.ic_params_map,
#             error_metric=self.error_metric,
#             train_results=self.train_model_results,
#             n_jobs=-1,
#             verbose=10
#         )

#         assert_frame_equal(expected_df, actual_df)

#     def test__darkgreyfit__with_reduce_train_results(
#             self,
#             mock_train_models,
#             mock_predict_models,
#             mock_reduce_results_df,
#             mock_apply_prefit_filter):

#         mock_obj_func = MagicMock()
#         mock_train_models.return_value = self.train_df
#         mock_predict_models.return_value = self.test_df
#         mock_reduce_results_df.return_value = self.train_df

#         darkgreyfit(
#             models=self.models,
#             X_train=self.X_train,
#             y_train=self.y_train,
#             X_test=self.X_test,
#             y_test=self.y_test,
#             ic_params_map=self.ic_params_map,
#             error_metric=self.error_metric,
#             prefit_splits=self.prefit_splits,
#             prefit_filter=None,
#             reduce_train_results=True,
#             method='nelder',
#             obj_func=mock_obj_func,
#             n_jobs=-1,
#             verbose=10
#         )

#         mock_reduce_results_df.assert_called_with(self.train_df)

#     def test__darkgreyfit__with_prefit_filter(
#             self,
#             mock_train_models,
#             mock_predict_models,
#             mock_reduce_results_df,
#             mock_apply_prefit_filter):

#         mock_obj_func = MagicMock()
#         mock_train_models.return_value = self.train_df
#         mock_predict_models.return_value = self.test_df
#         mock_apply_prefit_filter.return_value = self.train_df

#         darkgreyfit(
#             models=self.models,
#             X_train=self.X_train,
#             y_train=self.y_train,
#             X_test=self.X_test,
#             y_test=self.y_test,
#             ic_params_map=self.ic_params_map,
#             error_metric=self.error_metric,
#             prefit_splits=self.prefit_splits,
#             prefit_filter=self.prefit_filter,
#             reduce_train_results=False,
#             method='nelder',
#             obj_func=mock_obj_func,
#             n_jobs=-1,
#             verbose=10
#         )

#         mock_apply_prefit_filter.assert_called_with(self.train_df, self.prefit_filter)


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
