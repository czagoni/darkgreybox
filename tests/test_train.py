import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import mean_squared_error

from darkgreybox.train import (
    get_ic_params,
    train_model,
    train_models
)


def rmse(*args, **kwargs):
    return mean_squared_error(*args, **kwargs) ** 0.5


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

    @patch('darkgreybox.train.copy.deepcopy')
    @patch('darkgreybox.train.timer')
    @patch('darkgreybox.train.get_ic_params')
    def test__train_model(self, mock_get_ic_params, mock_timer, mock_deepcopy):

        model = MagicMock()
        mock_deepcopy.return_value = model
        model_result = MagicMock()
        model.predict.return_value = model_result
        model_result.Z.return_value = self.Z_train
        error_metric = MagicMock()
        error_metric.return_value = 0.95
        mock_get_ic_params.return_value = {'A0': 1}
        mock_timer.return_value = 1.0
        mock_obj_func = MagicMock()

        expected_df = pd.DataFrame({
            'start_date': [self.X_train.index[0]],
            'end_date': [self.X_train.index[-1]],
            'model': [model],
            'model_result': [model_result],
            'time': [0.0],
            'method': ["splendid"],
            'error': [0.95]
        })

        actual_df = train_model(
            base_model=model,
            X_train=self.X_train,
            y_train=self.y_train,
            error_metric=error_metric,
            method="splendid",
            obj_func=mock_obj_func
        )

        model.fit.assert_called_with(
            X=self.X_train.to_dict(orient='list'),
            y=self.y_train.values,
            method="splendid",
            ic_params={'A0': 1},
            obj_func=mock_obj_func
        )

        model.predict.assert_called_with(self.X_train.to_dict(orient='list'))
        # error_metric.assert_called_with(self.y_train.values, self.Z_train)

        assert_frame_equal(expected_df, actual_df)

    @patch('darkgreybox.train.copy.deepcopy')
    @patch('darkgreybox.train.timer')
    @patch('darkgreybox.train.get_ic_params')
    def test__train_model__fit_exception(self, mock_get_ic_params, mock_timer, mock_deepcopy):

        model = MagicMock()
        mock_deepcopy.return_value = model
        model.fit.side_effect = MagicMock(side_effect=ValueError('Test'))
        mock_get_ic_params.return_value = {'A0': 1}
        error_metric = MagicMock()
        mock_timer.return_value = 1.0
        mock_obj_func = MagicMock()

        expected_df = pd.DataFrame({
            'start_date': [self.X_train.index[0]],
            'end_date': [self.X_train.index[-1]],
            'model': [np.NaN],
            'model_result': [np.NaN],
            'time': [0.0],
            'method': ["splendid"],
            'error': [np.NaN]
        })

        actual_df = train_model(
            base_model=model,
            X_train=self.X_train,
            y_train=self.y_train,
            error_metric=error_metric,
            method="splendid",
            obj_func=mock_obj_func
        )

        model.fit.assert_called_with(
            X=self.X_train.to_dict(orient='list'),
            y=self.y_train.values,
            method="splendid",
            ic_params={'A0': 1},
            obj_func=mock_obj_func
        )

        assert_frame_equal(expected_df, actual_df)

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
