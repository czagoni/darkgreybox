import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import unittest
from unittest.mock import MagicMock, patch

from darkgreybox.base_model import DarkGreyModel
from darkgreybox.fit import (get_ic_params,
                             map_ic_params,
                             train_model,
                             train_models,
                             predict_model,
                             predict_models,
                             darkgreyfit,
                             reduce_results_df,
                             apply_prefit_filter)


@patch('darkgreybox.fit.apply_prefit_filter')
@patch('darkgreybox.fit.reduce_results_df')
@patch('darkgreybox.fit.predict_models')
@patch('darkgreybox.fit.train_models')
class DarkGreyFitTest(unittest.TestCase):

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

        self.models = [MagicMock()]
        self.train_model_results = [MagicMock()]
        self.test_model_results = [MagicMock()]
        self.train_time = [1.0]
        self.test_time = [0.0]
        self.train_error = [0.9]
        self.test_error = [0.8]
        self.ic_params_map = {'param': 'value'}
        self.error_metric = MagicMock()
        self.prefit_splits = MagicMock()
        self.prefit_filter = lambda x: x < 1,

        self.train_df = pd.DataFrame({
            'model': self.models,
            'model_result': self.train_model_results,
            'start': self.X_train.index[0],
            'end': self.X_train.index[-1],
            'time': self.train_time,
            'error': self.train_error
        })

        self.test_df = pd.DataFrame({
            'model': self.models,
            'model_result': self.test_model_results,
            'start': self.X_test.index[0],
            'end': self.X_test.index[-1],
            'time': self.test_time,
            'error': self.test_error
        })

        self.columns = pd.MultiIndex.from_tuples([('train', 'model'),
                                                  ('train', 'model_result'),
                                                  ('train', 'start'),
                                                  ('train', 'end'),
                                                  ('train', 'time'),
                                                  ('train', 'error'),
                                                  ('test', 'model'),
                                                  ('test', 'model_result'),
                                                  ('test', 'start'),
                                                  ('test', 'end'),
                                                  ('test', 'time'),
                                                  ('test', 'error')])

    def test__darkgreyfit__correct_data_flow(
            self,
            mock_train_models,
            mock_predict_models,
            mock_reduce_results_df,
            mock_apply_prefit_filter):

        mock_obj_func = MagicMock()

        mock_train_models.return_value = self.train_df
        mock_predict_models.return_value = self.test_df

        expected_df = pd.DataFrame(columns=self.columns,
                                   data=[[self.models[0], self.train_model_results[0], self.X_train.index[0],
                                          self.X_train.index[-1], self.train_time[0], self.train_error[0],
                                          self.models[0], self.test_model_results[0], self.X_test.index[0],
                                          self.X_test.index[-1], self.test_time[0], self.test_error[0]]])

        actual_df = darkgreyfit(models=self.models,
                                X_train=self.X_train,
                                y_train=self.y_train,
                                X_test=self.X_test,
                                y_test=self.y_test,
                                ic_params_map=self.ic_params_map,
                                error_metric=self.error_metric,
                                prefit_splits=self.prefit_splits,
                                prefit_filter=None,
                                reduce_train_results=False,
                                method='nelder',
                                obj_func=mock_obj_func,
                                n_jobs=-1,
                                verbose=10)

        mock_train_calls = mock_train_models.call_args_list

        mock_train_calls[0] == ({'models': self.models,
                                 'X_train': self.X_train,
                                 'y_train': self.y_train,
                                 'splits': self.prefit_splits,
                                 'error_metric': self.error_metric,
                                 'method': 'nelder',
                                 'obj_func': mock_obj_func,
                                 'n_jobs': -1,
                                 'verbose': 10})

        mock_train_calls[1] == ({'models': self.models,
                                 'X_train': self.X_train,
                                 'y_train': self.y_train,
                                 'splits': None,
                                 'error_metric': self.error_metric,
                                 'method': 'nelder',
                                 'obj_func': mock_obj_func,
                                 'n_jobs': -1,
                                 'verbose': 10})

        mock_reduce_results_df.assert_not_called()

        mock_predict_models.assert_called_with(models=self.models,
                                               X_test=self.X_test,
                                               y_test=self.y_test,
                                               ic_params_map=self.ic_params_map,
                                               error_metric=self.error_metric,
                                               train_results=self.train_model_results,
                                               n_jobs=-1,
                                               verbose=10)

        assert_frame_equal(expected_df, actual_df)

    def test__darkgreyfit__with_reduce_train_results(
            self,
            mock_train_models,
            mock_predict_models,
            mock_reduce_results_df,
            mock_apply_prefit_filter):

        mock_obj_func = MagicMock()
        mock_train_models.return_value = self.train_df
        mock_predict_models.return_value = self.test_df
        mock_reduce_results_df.return_value = self.train_df

        darkgreyfit(models=self.models,
                    X_train=self.X_train,
                    y_train=self.y_train,
                    X_test=self.X_test,
                    y_test=self.y_test,
                    ic_params_map=self.ic_params_map,
                    error_metric=self.error_metric,
                    prefit_splits=self.prefit_splits,
                    prefit_filter=None,
                    reduce_train_results=True,
                    method='nelder',
                    obj_func=mock_obj_func,
                    n_jobs=-1,
                    verbose=10)

        mock_reduce_results_df.assert_called_with(self.train_df)

    def test__darkgreyfit__with_prefit_filter(
            self,
            mock_train_models,
            mock_predict_models,
            mock_reduce_results_df,
            mock_apply_prefit_filter):

        mock_obj_func = MagicMock()
        mock_train_models.return_value = self.train_df
        mock_predict_models.return_value = self.test_df
        mock_apply_prefit_filter.return_value = self.train_df

        darkgreyfit(models=self.models,
                    X_train=self.X_train,
                    y_train=self.y_train,
                    X_test=self.X_test,
                    y_test=self.y_test,
                    ic_params_map=self.ic_params_map,
                    error_metric=self.error_metric,
                    prefit_splits=self.prefit_splits,
                    prefit_filter=self.prefit_filter,
                    reduce_train_results=False,
                    method='nelder',
                    obj_func=mock_obj_func,
                    n_jobs=-1,
                    verbose=10)

        mock_apply_prefit_filter.assert_called_with(self.train_df, self.prefit_filter)


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

    @patch('darkgreybox.fit.train_model')
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

        actual_df = train_models(models=models,
                                 X_train=self.X_train,
                                 y_train=self.y_train,
                                 error_metric=error_metric,
                                 splits=None,
                                 method='nelder',
                                 n_jobs=1,
                                 verbose=10)

        assert_frame_equal(expected_df, actual_df)

    @patch('darkgreybox.fit.train_model')
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

        actual_df = train_models(models=models,
                                 X_train=self.X_train,
                                 y_train=self.y_train,
                                 error_metric=error_metric,
                                 splits=splits,
                                 method='nelder',
                                 n_jobs=1,
                                 verbose=10)

        assert_frame_equal(expected_df, actual_df)

    @patch('darkgreybox.fit.train_model')
    def test__train_models__parallel(self, mock_train_model):
        # TODO
        pass

    @patch('darkgreybox.fit.copy.deepcopy')
    @patch('darkgreybox.fit.timer')
    @patch('darkgreybox.fit.get_ic_params')
    def test__train_model(self, mock_get_ic_params, mock_timer, mock_deepcopy):

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
        mock_obj_func = MagicMock()

        expected_df = pd.DataFrame({
            'start_date': [self.X_train.index[0]],
            'end_date': [self.X_train.index[-1]],
            'model': [model_fit],
            'model_result': [model_result],
            'time': [0.0],
            'method': ["splendid"],
            'error': [0.95]
        })

        actual_df = train_model(base_model=model,
                                X_train=self.X_train,
                                y_train=self.y_train,
                                error_metric=error_metric,
                                method="splendid",
                                obj_func=mock_obj_func)

        model.fit.assert_called_with(X=self.X_train.to_dict(orient='list'),
                                     y=self.y_train.values,
                                     method="splendid",
                                     ic_params={'A0': 1},
                                     obj_func=mock_obj_func)

        model_fit.predict.assert_called_with(self.X_train)
        # error_metric.assert_called_with(self.y_train.values, self.Z_train)

        assert_frame_equal(expected_df, actual_df)

    @patch('darkgreybox.fit.copy.deepcopy')
    @patch('darkgreybox.fit.timer')
    @patch('darkgreybox.fit.get_ic_params')
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

        actual_df = train_model(base_model=model,
                                X_train=self.X_train,
                                y_train=self.y_train,
                                error_metric=error_metric,
                                method="splendid",
                                obj_func=mock_obj_func)

        model.fit.assert_called_with(X=self.X_train.to_dict(orient='list'),
                                     y=self.y_train.values,
                                     method="splendid",
                                     ic_params={'A0': 1},
                                     obj_func=mock_obj_func)

        assert_frame_equal(expected_df, actual_df)

    @patch('darkgreybox.fit.predict_model')
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

        actual_df = predict_models(models=models,
                                   X_test=self.X_test,
                                   y_test=self.y_test,
                                   ic_params_map=ic_params_map,
                                   error_metric=error_metric,
                                   train_results=train_results,
                                   n_jobs=1,
                                   verbose=10)

        mock_predict_model.assert_called_with(models[0], self.X_test, self.y_test,
                                              ic_params_map, error_metric, train_results[0])

        assert_frame_equal(expected_df, actual_df, check_dtype=False)

    @patch('darkgreybox.fit.predict_model')
    def test__predict_models__parallel(self, mock_predict_model):
        # TODO
        pass

    @patch('darkgreybox.fit.map_ic_params')
    @patch('darkgreybox.fit.timer')
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

        actual_df = predict_model(model=model,
                                  X_test=self.X_test,
                                  y_test=self.y_test,
                                  ic_params_map={'B': 1},
                                  error_metric=error_metric,
                                  train_result=train_result)

        mock_map_ic_params.assert_called_with({'B': 1}, model, self.X_test, self.y_test, train_result)

        mock_predict.assert_called_with(X=self.X_test.to_dict(orient='list'),
                                        ic_params={'A0': 1})

        assert_frame_equal(expected_df, actual_df)

    @patch('darkgreybox.fit.timer')
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

        actual_df = predict_model(model=model,
                                  X_test=self.X_test,
                                  y_test=self.y_test,
                                  ic_params_map={},
                                  error_metric=error_metric,
                                  train_result=None)

        assert_frame_equal(expected_df, actual_df)

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

    def test__apply_prefit_filter(self):

        df = pd.DataFrame(data={
            'value': [0, 0, 0, 10, 20, 30, 40, 50],
            'error': [np.nan, -np.inf, np.inf, 2.0000011, 2.0000012, 2.000002, 1, 3],
            'time': [0, 0, 0, 2, 1, 3, 4, 5]
        })

        expected_df = pd.DataFrame(data={
            'value': [10, 40],
            'error': [2.0000011, 1],
            'time': [2, 4]
        })

        actual_df = apply_prefit_filter(df, prefit_filter=lambda x: abs(x) < 2.0000012)

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

    def mock_predict_model_side_effect(self, model, X_test, y_test, ic_params_map, error_metric, train_result):
        return pd.DataFrame({
            'start_date': [X_test.index[0]],
            'end_date': [X_test.index[-1]],
            'model': [model],
            'model_result': ['model_result'],
            'time': [0.0],
            'error': [0.0]
        })
