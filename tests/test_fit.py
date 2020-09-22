import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

import unittest 
from unittest.mock import MagicMock, patch

from darkgreybox.fit import (get_ic_params,
                             train_model)


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

    @patch('darkgreybox.fit.copy.deepcopy')
    @patch('darkgreybox.fit.timer')
    @patch('darkgreybox.fit.get_ic_params')
    def test_train_model(self, mock_get_ic_params, mock_timer, mock_deepcopy):

        model = MagicMock()
        mock_deepcopy.return_value = model
        model_fit = MagicMock()
        model.fit.return_value = model_fit
        model_result = MagicMock()
        model_fit.predict.return_value = model_result
        model_result.Z.return_value = self.Z_train
        error_metric = MagicMock()
        error_metric.return_value = 0.95
        mock_get_ic_params.return_value = {'A0': 1}
        mock_timer.return_value = 1.0

        expected_df = pd.DataFrame({
            'start_date': [self.X_train.index[0]],
            'end_date': [self.X_train.index[-1]],
            'model': [model_fit],
            'model_result': [model_result],
            'time': [0.0],
            'method': ["splendid"],
            'error': [0.95]
        })

        actual_df = train_model(base_model=model, X_train=self.X_train,
                                y_train=self.y_train, method="splendid", error_metric=error_metric)

        model.fit.assert_called_with(
            X=self.X_train, y=self.y_train.values, method="splendid", ic_params={'A0': 1})
        model_fit.predict.assert_called_with(self.X_train)
        # error_metric.assert_called_with(self.y_train.values, self.Z_train)

        assert_frame_equal(expected_df, actual_df)

    @patch('darkgreybox.fit.copy.deepcopy')
    @patch('darkgreybox.fit.timer')
    @patch('darkgreybox.fit.get_ic_params')
    def test_train_model_fit_exception(self, mock_get_ic_params, mock_timer, mock_deepcopy):

        model = MagicMock()
        mock_deepcopy.return_value = model
        model.fit.side_effect = MagicMock(side_effect=ValueError('Test'))
        mock_get_ic_params.return_value = {'A0': 1}
        error_metric = MagicMock()
        mock_timer.return_value = 1.0

        expected_df = pd.DataFrame({
            'start_date': [self.X_train.index[0]],
            'end_date': [self.X_train.index[-1]],
            'model': [np.NaN],
            'model_result': [np.NaN],
            'time': [0.0],
            'method': ["splendid"],
            'error': [np.NaN]
        })

        actual_df = train_model(base_model=model, X_train=self.X_train,
                                y_train=self.y_train, method="splendid", error_metric=error_metric)

        model.fit.assert_called_with(
            X=self.X_train, y=self.y_train.values, method="splendid", ic_params={'A0': 1})

        assert_frame_equal(expected_df, actual_df)

    def test_get_ic_params(self):

        model = MagicMock()
        model.params = {'A0': 0, 'B': 1, 'C0': 2, 'D': 3}



        expected = {'A0': 10, 'C0': 50}
        actual = get_ic_params(model, self.X_train)

        self.assertEqual(expected, actual)

