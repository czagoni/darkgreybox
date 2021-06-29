from typing import cast
import unittest
from unittest.mock import MagicMock, patch

import datetime as dt
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from pandas.testing import assert_frame_equal

from darkgreybox.models import Ti
from darkgreybox.train import (
    get_ic_params,
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


def error_metric(y, Z):
    return np.sum(y - Z)


class Train2Test(unittest.TestCase):

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


class TrainTest(unittest.TestCase):

    def setUp(self):

        self.X_train = pd.DataFrame({
            'A0': [10, 20],
            'B': [30, 40],
            'C0': [50, 60],
            'D': [70, 80]
        })

        self.y_train = pd.Series([100, 110])
        self.Z_train = np.array([120, 130])

    @patch('darkgreybox.train.train_model')
    def test__train_models__not_parallel_splits_none(self, mock_train_model):

        mock_train_model.side_effect = self.mock_train_model_side_effect

        models = [MagicMock(), MagicMock()]
        error_metric = MagicMock()

        expected_df = pd.DataFrame({
            'start_date': [self.X_train.index[0]] * 2,
            'end_date': [self.X_train.index[-1]] * 2,
            'model': models,
            'model_result': ['model_result'] * 2,
            'time': [0.0] * 2,
            'method': ['nelder'] * 2,
            'error': [0.0] * 2
        })

        actual_df = train_models(
            models=models,
            X_train=self.X_train,
            y_train=self.y_train,
            error_metric=error_metric,
            splits=None,
            method='nelder',
            n_jobs=1,
            verbose=10
        )

        assert_frame_equal(expected_df, actual_df)

    @patch('darkgreybox.train.train_model')
    def test__train_models__not_parallel(self, mock_train_model):

        mock_train_model.side_effect = self.mock_train_model_side_effect

        models = [MagicMock()]
        error_metric = MagicMock()

        splits = [(None, [0]), (None, [1])]

        expected_df = pd.DataFrame({
            'start_date': [self.X_train.index[0], self.X_train.index[1]],
            'end_date': [self.X_train.index[0], self.X_train.index[1]],
            'model': models * 2,
            'model_result': ['model_result'] * 2,
            'time': [0.0] * 2,
            'method': ['nelder'] * 2,
            'error': [0.0] * 2
        })

        actual_df = train_models(
            models=models,
            X_train=self.X_train,
            y_train=self.y_train,
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

    def test__get_ic_params(self):

        model = MagicMock()
        model.params = {'A0': 0, 'B': 1, 'C0': 2, 'D': 3}

        expected = {'A0': 10, 'C0': 50}
        actual = get_ic_params(model, self.X_train)

        self.assertEqual(expected, actual)

    def mock_train_model_side_effect(self, base_model, X_train, y_train, error_metric, method, obj_func):
        return pd.DataFrame({
            'start_date': [X_train.index[0]],
            'end_date': [X_train.index[-1]],
            'model': [base_model],
            'model_result': ['model_result'],
            'time': [0.0],
            'method': [method],
            'error': [0.0]
        })
