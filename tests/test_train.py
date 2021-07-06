import datetime as dt
import unittest
from typing import Callable, Optional, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from darkgreybox.base_model import DarkGreyModel
from darkgreybox.models import Ti
from darkgreybox.train import (
    get_ic_params,
    reduce_results_df,
    train_model,
    train_models
)


train_start = dt.datetime(2021, 1, 1, 1, 0)
train_end = dt.datetime(2021, 1, 1, 6, 0)

y_train = pd.Series([10, 10, 20, 20, 20, 30])

X_train = pd.DataFrame(
    index=pd.date_range(train_start, train_end, freq='1H'),
    data={
        'Ta': [10, 10, 10, 20, 20, 20],
        'Ph': [0, 10, 0, 0, 10, 0],
        'Ti0': [10, 10, 20, 20, 20, 30]
    })

method = 'nelder'

params = {
    'Ti0': {'value': 10, 'vary': False},
    'Ria': {'value': 1},
    'Ci': {'value': 1},
}


def error_metric(y: np.ndarray, Z: np.ndarray) -> float:
    return np.sum(y - Z)


class TrainTest(unittest.TestCase):

    @patch('darkgreybox.train.train_model')
    def test__train_models__not_parallel_without_splits__returns_correct_dataframe(self, mock_train_model):

        mock_train_model.side_effect = mock_train_model_side_effect

        models = [MagicMock(), MagicMock()]
        error_metric = MagicMock()

        expected_df = pd.DataFrame({
            'start_date': [train_start] * 2,
            'end_date': [train_end] * 2,
            'model': models,
            'model_result': ['model_result'] * 2,
            'time': [0.0] * 2,
            'method': ['nelder'] * 2,
            'error': [0.0] * 2
        })

        actual_df = train_models(
            models=models,
            X_train=X_train,
            y_train=y_train,
            error_metric=error_metric,
            splits=None,
            method='nelder',
            n_jobs=1,
            verbose=10
        )

        assert_frame_equal(expected_df, actual_df)

    @patch('darkgreybox.train.train_model')
    def test__train_models__not_parallel_with_reduce_results_true__returns_correct_dataframe(self, mock_train_model):

        mock_train_model.side_effect = mock_train_model_side_effect

        models = [MagicMock(), MagicMock()]
        error_metric = MagicMock()

        expected_df = pd.DataFrame({
            'start_date': [train_start],
            'end_date': [train_end],
            'model': [models[0]],
            'model_result': ['model_result'],
            'time': [0.0],
            'method': ['nelder'],
            'error': [0.0]
        })

        actual_df = train_models(
            models=models,
            X_train=X_train,
            y_train=y_train,
            error_metric=error_metric,
            splits=None,
            method='nelder',
            reduce_train_results=True,
            n_jobs=1,
            verbose=10
        )

        assert_frame_equal(expected_df, actual_df)

    @patch('darkgreybox.train.train_model')
    def test__train_models__not_parallel_with_splits__returns_correct_dataframe(self, mock_train_model):

        mock_train_model.side_effect = mock_train_model_side_effect

        models = [MagicMock()]
        error_metric = MagicMock()

        splits = [(None, [0]), (None, [1])]

        expected_df = pd.DataFrame({
            'start_date': [X_train.index[0], X_train.index[1]],
            'end_date': [X_train.index[0], X_train.index[1]],
            'model': models * 2,
            'model_result': ['model_result'] * 2,
            'time': [0.0] * 2,
            'method': ['nelder'] * 2,
            'error': [0.0] * 2
        })

        actual_df = train_models(
            models=models,
            X_train=X_train,
            y_train=y_train,
            error_metric=error_metric,
            splits=splits,
            method='nelder',
            n_jobs=1,
            verbose=10
        )

        assert_frame_equal(expected_df, actual_df)

    @patch('darkgreybox.train.train_model')
    def test__train_models__parallel(self, mock_train_model):
        # TODO
        pass

    def test__train_model__creates_a_copy_of_base_model(self):

        base_model = Ti(params, rec_duration=1)

        actual_df = train_model(base_model, X_train, y_train, error_metric, method)
        trained_model = actual_df['model'].iloc[0]

        self.assertNotEqual(base_model, trained_model)

    @patch('darkgreybox.train.timer')
    def test__train_model__returns_correct_dataframe__for_successfully_trained_model(self, mock_timer):

        timer_start = 1.0
        timer_stop = 2.0
        mock_timer.side_effect = [timer_start, timer_stop]

        base_model = Ti(params, rec_duration=1)

        expected_columns = pd.Index([
            'start_date',
            'end_date',
            'model',
            'model_result',
            'time',
            'method',
            'error',
        ])

        actual_df = train_model(base_model, X_train, y_train, error_metric, method)

        self.assertTrue(expected_columns.equals(actual_df.columns))

        self.assertEqual(train_start, actual_df['start_date'].iloc[0])
        self.assertEqual(train_end, actual_df['end_date'].iloc[0])

        self.assertIsInstance(actual_df['model'].iloc[0], Ti)

        assert_allclose(y_train.values, actual_df['model_result'].iloc[0].Z, atol=0.01)

        self.assertEqual(1.0, actual_df['time'].iloc[0])
        self.assertEqual(method, actual_df['method'].iloc[0])
        self.assertAlmostEqual(-0.01, cast(float, actual_df['error'].iloc[0]), places=4)

    @patch('darkgreybox.train.copy')
    @patch('darkgreybox.train.timer')
    def test__train_model__returns_correct_dataframe__for_model_that_raises_value_error_during_fitting(
        self,
        mock_timer,
        mock_copy
    ):

        timer_start = 1.0
        timer_stop = 2.0
        mock_timer.side_effect = [timer_start, timer_stop]

        base_model = MagicMock()
        mock_model = MagicMock()
        mock_model.fit.side_effect = ValueError('Test')
        mock_copy.deepcopy.return_value = mock_model

        expected_df = pd.DataFrame({
            'start_date': [train_start],
            'end_date': [train_end],
            'model': [np.NaN],
            'model_result': [np.NaN],
            'time': [timer_stop - timer_start],
            'method': [method],
            'error': [np.NaN]
        })

        actual_df = train_model(base_model, X_train, y_train, error_metric)

        assert_frame_equal(expected_df, actual_df)

    @patch('darkgreybox.train.copy')
    def test__train_model__passes_through_keyword_arguments(self, mock_copy):

        base_model = MagicMock()
        mock_model = MagicMock()
        mock_copy.deepcopy.return_value = mock_model

        mock_method = 'some-method'
        mock_obj_func = MagicMock()

        train_model(base_model, X_train, y_train, error_metric=MagicMock(), method=mock_method, obj_func=mock_obj_func)

        self.assertEqual(mock_model.fit.call_args.kwargs['method'], mock_method)
        self.assertEqual(mock_model.fit.call_args.kwargs['obj_func'], mock_obj_func)

    def test__get_ic_params__returns_correct_ic_params(self):

        X_train = pd.DataFrame({
            'A0': [10, 20],
            'B': [30, 40],
            'C0': [50, 60],
            'D': [70, 80]
        })

        model = MagicMock()

        for (model_params, expected_ic_params) in [
            ({}, {}),
            ({'B': 'value', }, {}),
            ({'B': 'value', 'D': 'value'}, {}),
            ({'A0': 'value'}, {'A0': 10}),
            ({'A0': 'value', 'B': 'value'}, {'A0': 10}),
            ({'A0': 'value', 'B': 'value', 'C0': 2, }, {'A0': 10, 'C0': 50}),
        ]:
            with self.subTest(model_params=model_params, expected_ic_params=expected_ic_params):
                model.params = model_params
                actual_ic_params = get_ic_params(model, X_train)
                self.assertEqual(expected_ic_params, actual_ic_params)

    def test__get_ic_params__raises_keyerror_for_param_with_non_existent_field(self):

        X_train = pd.DataFrame({
            'A0': [10, 20],
            'B': [30, 40],
            'C0': [50, 60],
            'D': [70, 80]
        })

        model = MagicMock()
        model.params = {'NonExistentField0': 'value'}

        with self.assertRaisesRegex(
            KeyError,
            'Initial condition key NonExistentField0 does not have corresponding X_train field'
        ):
            get_ic_params(model, X_train)

    def test__reduce_results_df(self):

        for (desc, input_df, expected_df) in [
            (
                'does nothing when no duplicates',
                pd.DataFrame(data={'value': [0.0, 1.0], 'error': [0.0, 1.0], 'time': [0.0, 1.0]}),
                pd.DataFrame(data={'value': [0.0, 1.0], 'error': [0.0, 1.0], 'time': [0.0, 1.0]}),
            ),
            (
                'removes exact duplicates',
                pd.DataFrame(data={'value': [0.0, 0.0], 'error': [0.0, 0.0], 'time': [0.0, 0.0]}),
                pd.DataFrame(data={'value': [0.0], 'error': [0.0], 'time': [0.0]}),
            ),
            (
                'drops records with nan and inf/-inf records',
                pd.DataFrame(data={
                    'value': [0.0, np.nan, np.inf, -np.inf],
                    'error': [0.0, 1.0, 1.0, 1.0],
                    'time': [0.0, 1.0, 1.0, 1.0]
                }),
                pd.DataFrame(data={'value': [0.0], 'error': [0.0], 'time': [0.0]}),
            ),
            (
                'rounds errors to specified decimal point',
                pd.DataFrame(data={
                    'value': [0.0, 0.0, np.nan, np.inf, -np.inf],
                    'error': [0.0000011, 0.0000001, 1.0, 1.0, 1.0],
                    'time': [0.0, 1.0, 1.0, 1.0, 1.0]
                }),
                pd.DataFrame(data={'value': [0.0, 0.0], 'error': [0.0, 0.000001], 'time': [1.0, 0.0]}),
            ),
            (
                'removes duplicates and keeps first record in ascending order based on time field',
                pd.DataFrame(data={
                    'value': [0.0, 0.0, np.nan, np.inf, -np.inf],
                    'error': [0.0000011, 0.0000009, 1.0, 1.0, 1.0],
                    'time': [1.0, 0.0, 1.0, 1.0, 1.0]
                }),
                pd.DataFrame(data={'value': [0.0], 'error': [0.000001], 'time': [0.0]}),
            ),
            (
                'sort records based on errors after duplicates removed',
                pd.DataFrame(data={
                    'value': [0.0, 0.0, 1.0, np.nan, np.inf, -np.inf],
                    'error': [0.0000011, 0.0000009, 0.0, 0.0, 1.0, 1.0],
                    'time': [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
                }),
                pd.DataFrame(data={'value': [1.0, 0.0], 'error': [0.0, 0.000001], 'time': [1.0, 0.0]}),
            ),
        ]:
            with self.subTest(desc, input_df=input_df, expected_df=expected_df):
                actual_df = reduce_results_df(input_df)
                assert_frame_equal(expected_df, actual_df)


def mock_train_model_side_effect(
    base_model: DarkGreyModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    error_metric: Callable,
    method: str = 'nelder',
    obj_func: Optional[Callable] = None,
) -> pd.DataFrame:

    return pd.DataFrame({
        'start_date': [X_train.index[0]],
        'end_date': [X_train.index[-1]],
        'model': [base_model],
        'model_result': ['model_result'],
        'time': [0.0],
        'method': [method],
        'error': [0.0]
    })
