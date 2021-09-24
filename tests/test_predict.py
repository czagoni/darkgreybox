import datetime as dt
import unittest
from typing import Any, Callable, Dict, Union, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose

from darkgreybox.base_model import DarkGreyModel, DarkGreyModelResult
from darkgreybox.models import Ti
from darkgreybox.predict import (
    map_ic_params,
    predict_model,
    predict_models,
)

test_start = dt.datetime(2021, 1, 1, 7, 0)
test_end = dt.datetime(2021, 1, 1, 8, 0)

X_test = pd.DataFrame(
    index=pd.date_range(test_start, test_end, freq='1H'),
    data={
        'Ta': [10, 10],
        'Ph': [10, 0],
        'Ti0': [10, 20]
    })

y_test = pd.Series([10, 20])

params = {
    'Ti0': {'value': 10, 'vary': False},
    'Ria': {'value': 1},
    'Ci': {'value': 1},
}


def error_metric(y: np.ndarray, Z: np.ndarray) -> float:
    return np.sum(y - Z)


class PredictTest(unittest.TestCase):

    @patch('darkgreybox.predict.predict_model')
    def test__predict_models__not_parallel(self, mock_predict_model):

        mock_predict_model.side_effect = mock_predict_model_side_effect

        models = [MagicMock()] * 2
        train_results = [DarkGreyModelResult(None)] * 2
        ic_params_map = {}

        expected_df = pd.DataFrame({
            'start_date': [test_start, test_start],
            'end_date': [test_end, test_end],
            'model': models,
            'model_result': ['model_result'] * 2,
            'time': [0.0] * 2,
            'error': [0.0] * 2
        })

        actual_df = predict_models(
            models=models,
            X_test=X_test,
            y_test=y_test,
            ic_params_map=ic_params_map,
            error_metric=error_metric,
            train_results=train_results,
            n_jobs=1,
            verbose=10
        )

        mock_predict_model.assert_called_with(
            models[0],
            X_test,
            y_test,
            ic_params_map,
            error_metric,
            train_results[0]
        )

        assert_frame_equal(expected_df, actual_df, check_dtype=False)

    @patch('darkgreybox.predict.predict_model')
    def test__predict_models__parallel(self, mock_predict_model):
        # TODO
        pass

    @patch('darkgreybox.predict.timer')
    def test__predict_model(self, mock_timer):

        train_result = MagicMock()
        train_result.Ti = [0.0, 10.0]

        timer_start = 1.0
        timer_stop = 2.0
        mock_timer.side_effect = [timer_start, timer_stop]

        model = Ti(params, rec_duration=1)

        ic_params_map = {
            'Ti0': lambda X_test, y_test, train_result: train_result.Ti[-1]
        }

        expected_columns = pd.Index([
            'start_date',
            'end_date',
            'model',
            'model_result',
            'time',
            'error',
        ])

        actual_df = predict_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            ic_params_map=ic_params_map,
            error_metric=error_metric,
            train_result=train_result
        )

        self.assertTrue(expected_columns.equals(actual_df.columns))

        self.assertEqual(test_start, actual_df['start_date'].iloc[0])
        self.assertEqual(test_end, actual_df['end_date'].iloc[0])

        self.assertIsInstance(actual_df['model'].iloc[0], Ti)

        assert_allclose(y_test.values, cast(DarkGreyModelResult, actual_df['model_result'].iloc[0]).Z, atol=0.01)

        self.assertEqual(1.0, actual_df['time'].iloc[0])
        self.assertAlmostEqual(0.0, cast(float, actual_df['error'].iloc[0]), places=4)

    @ patch('darkgreybox.predict.timer')
    def test__predict_model__returns_correct_dataframe__for_not_a_model_instance(self, mock_timer):

        model = 'not-a-model'
        train_result = DarkGreyModelResult(None)

        timer_start = 1.0
        timer_stop = 2.0
        mock_timer.side_effect = [timer_start, timer_stop]

        expected_df = pd.DataFrame({
            'start_date': [test_start],
            'end_date': [test_end],
            'model': [np.nan],
            'model_result': [np.nan],
            'time': [timer_stop - timer_start],
            'error': [np.nan]
        })

        actual_df = predict_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            ic_params_map={},
            error_metric=error_metric,
            train_result=train_result,
        )

        assert_frame_equal(expected_df, actual_df)

    def test__map_ic_params__returns_correct_ic_params(self):

        X_test = pd.DataFrame({
            'A0': [10, 20],
            'B': [30, 40],
        })

        y_test = pd.Series([1, 2])

        train_result = MagicMock()
        train_result.Te = [100, 200]

        model = MagicMock()
        model.params = {
            'A0': 'value', 'B0': 'value', 'C0': 'value'
        }

        for (ic_params_map, expected_ic_params) in [
            ({}, {}),
            ({'A0': lambda X_test, y_test, train_result: X_test['A0'].iloc[0]}, {'A0': 10}),
            ({'B0': lambda X_test, y_test, train_result: y_test.iloc[0]}, {'B0': 1}),
            ({'C0': lambda X_test, y_test, train_result: train_result.Te[0]}, {'C0': 100}),
            ({
                'A0': lambda X_test, y_test, train_result: X_test['A0'].iloc[0],
                'B0': lambda X_test, y_test, train_result: y_test.iloc[0],
                'C0': lambda X_test, y_test, train_result: train_result.Te[0]
            },
                {
                'A0': 10,
                'B0': 1,
                'C0': 100
            }),
        ]:
            with self.subTest(ic_params_map=ic_params_map, expected_ic_params=expected_ic_params):
                actual_ic_params = map_ic_params(ic_params_map, model, X_test, y_test, train_result)
                self.assertEqual(expected_ic_params, actual_ic_params)


def mock_predict_model_side_effect(
    model: Union[DarkGreyModel, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ic_params_map: Dict,
    error_metric: Callable,
    train_result: DarkGreyModelResult,
) -> pd.DataFrame:
    return pd.DataFrame({
        'start_date': [test_start],
        'end_date': [test_end],
        'model': [model],
        'model_result': ['model_result'],
        'time': [0.0],
        'error': [0.0]
    })
