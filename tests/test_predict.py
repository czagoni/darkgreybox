import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from darkgreybox.base_model import DarkGreyModel
from darkgreybox.predict import (
    map_ic_params,
    predict_model,
    predict_models,
)


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

    @patch('darkgreybox.predict.predict_model')
    def test__predict_models__not_parallel(self, mock_predict_model):

        mock_predict_model.side_effect = self.mock_predict_model_side_effect

        models = [MagicMock()] * 2
        train_results = [MagicMock()] * 2
        error_metric = MagicMock()
        ic_params_map = {}

        expected_df = pd.DataFrame({
            'start_date': [self.X_test.index[0], self.X_test.index[0]],
            'end_date': [self.X_test.index[-1], self.X_test.index[-1]],
            'model': models,
            'model_result': ['model_result'] * 2,
            'time': [0.0] * 2,
            'error': [0.0] * 2
        })

        actual_df = predict_models(
            models=models,
            X_test=self.X_test,
            y_test=self.y_test,
            ic_params_map=ic_params_map,
            error_metric=error_metric,
            train_results=train_results,
            n_jobs=1,
            verbose=10
        )

        mock_predict_model.assert_called_with(
            models[0],
            self.X_test,
            self.y_test,
            ic_params_map,
            error_metric,
            train_results[0]
        )

        assert_frame_equal(expected_df, actual_df, check_dtype=False)

    @patch('darkgreybox.predict.predict_model')
    def test__predict_models__parallel(self, mock_predict_model):
        # TODO
        pass

    @patch('darkgreybox.predict.map_ic_params')
    @patch('darkgreybox.predict.timer')
    @patch('darkgreybox.base_model.DarkGreyModel.predict')
    def test__predict_model(self, mock_predict, mock_timer, mock_map_ic_params):

        model = DarkGreyModel({}, 1)
        model_result = MagicMock()
        train_result = MagicMock()
        mock_predict.return_value = model_result
        mock_timer.return_value = 1.0
        mock_map_ic_params.return_value = {'A0': 1}
        error_metric = MagicMock()
        error_metric.return_value = 0.95

        expected_df = pd.DataFrame({
            'start_date': [self.X_test.index[0]],
            'end_date': [self.X_test.index[-1]],
            'model': [model],
            'model_result': [model_result],
            'time': [0.0],
            'error': [0.95]
        })

        actual_df = predict_model(
            model=model,
            X_test=self.X_test,
            y_test=self.y_test,
            ic_params_map={'B': 1},
            error_metric=error_metric,
            train_result=train_result
        )

        mock_map_ic_params.assert_called_with({'B': 1}, model, self.X_test, self.y_test, train_result)

        mock_predict.assert_called_with(
            X=self.X_test.to_dict(orient='list'),
            ic_params={'A0': 1}
        )

        assert_frame_equal(expected_df, actual_df)

    @patch('darkgreybox.predict.timer')
    def test__predict_model__not_model_instance(self, mock_timer):

        model = 1.0
        mock_timer.return_value = 1.0
        error_metric = MagicMock()

        expected_df = pd.DataFrame({
            'start_date': [self.X_test.index[0]],
            'end_date': [self.X_test.index[-1]],
            'model': [np.nan],
            'model_result': [np.nan],
            'time': [0.0],
            'error': [np.nan]
        })

        actual_df = predict_model(
            model=model,
            X_test=self.X_test,
            y_test=self.y_test,
            ic_params_map={},
            error_metric=error_metric,
            train_result=None
        )

        assert_frame_equal(expected_df, actual_df)

    def test__map_ic_params(self):

        train_result = [57]

        ic_params_map = {
            'A0': lambda X_test, y_test, train_result: y_test.iloc[0],
            'B0': lambda X_test, y_test, train_result: y_test.iloc[0],
            'C0': lambda X_test, y_test, train_result: X_test['C0'].iloc[0],
            'D': lambda X_test, y_test, train_result: train_result[0],
        }

        model = MagicMock()
        model.params = {'A0': 0, 'B': 1, 'C0': 2, 'D': 3}

        expected = {'A0': 100, 'C0': 50, 'D': 57}
        actual = map_ic_params(ic_params_map, model, self.X_test, self.y_test, train_result)

        self.assertEqual(expected, actual)

    def mock_predict_model_side_effect(self, model, X_test, y_test, ic_params_map, error_metric, train_result):
        return pd.DataFrame({
            'start_date': [X_test.index[0]],
            'end_date': [X_test.index[-1]],
            'model': [model],
            'model_result': ['model_result'],
            'time': [0.0],
            'error': [0.0]
        })
