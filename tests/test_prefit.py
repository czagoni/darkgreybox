import unittest
from unittest.mock import MagicMock, patch

import datetime as dt
import numpy as np
import pandas as pd

from darkgreybox.models import Ti
from darkgreybox.prefit import prefit_models


train_start = dt.datetime(2021, 1, 1, 1, 0)
train_end = dt.datetime(2021, 1, 1, 6, 0)

rec_duration = 1

y_train = pd.Series([10, 10, 20, 20, 20, 30])

X_train = pd.DataFrame(
    index=pd.date_range(train_start, train_end, freq=f'{rec_duration}H'),
    data={
        'Ta': [10, 10, 10, 20, 20, 20],
        'Ph': [0, 10, 0, 0, 10, 0],
        'Ti0': [10, 10, 20, 20, 20, 30]
    })


class PrefitModelsTest(unittest.TestCase):

    def test__prefit_models__returns_original_models__when_prefit_splits_is_None(self):

        prefit_splits = None
        prefit_filter = None

        error_metric = MagicMock()

        model_1 = MagicMock()
        model_2 = MagicMock()

        models = [model_1, model_2]

        actual_models = prefit_models(
            models,
            X_train,
            y_train,
            error_metric,
            prefit_splits,
            prefit_filter,
            n_jobs=1
        )

        self.assertListEqual(models, actual_models)

    def test__prefit_models__returns_prefit_models__when_prefit_splits_is_not_None(self):

        prefit_splits = [([], [0, 1, 2]), ([], [3, 4, 5])]
        prefit_filter = None

        error_metric = MagicMock()

        params = {
            'Ti0': {'value': 10, 'vary': False},
            'Ria': {'value': 1},
            'Ci': {'value': 1},
        }

        model_1 = Ti(params, rec_duration=1)
        model_2 = Ti(params, rec_duration=2)

        models = [model_1, model_2]

        actual_models = prefit_models(
            models,
            X_train,
            y_train,
            error_metric,
            prefit_splits,
            prefit_filter,
            n_jobs=1
        )

        for actual_model in actual_models:
            self.assertIsInstance(actual_model, Ti)

        self.assertEqual(model_1.rec_duration, actual_models[0].rec_duration)
        self.assertEqual(model_2.rec_duration, actual_models[1].rec_duration)
        self.assertEqual(model_1.rec_duration, actual_models[2].rec_duration)
        self.assertEqual(model_2.rec_duration, actual_models[3].rec_duration)

    def test__prefit_models__returns_filtered_prefit_models__according_to_prefit_filter(self):

        prefit_splits = [([], [0, 1, 2]), ([], [3, 4, 5])]

        error_metric = MagicMock(side_effect=[-2, 0, 1, 2] * 4)

        params = {
            'Ti0': {'value': 10, 'vary': False},
            'Ria': {'value': 1},
            'Ci': {'value': 1},
        }

        model_1 = Ti(params, rec_duration=1)
        model_2 = Ti(params, rec_duration=2)

        models = [model_1, model_2]

        for (prefit_filter_value, num_actual_models) in [
            (1, 1),
            (1.1, 2),
            (2.0, 2),
            (2.1, 4),
        ]:
            with self.subTest(prefit_filter_value=prefit_filter_value, num_actual_models=num_actual_models):

                def prefit_filter(x): return abs(x) < prefit_filter_value

                actual_models = prefit_models(
                    models,
                    X_train,
                    y_train,
                    error_metric,
                    prefit_splits,
                    prefit_filter,
                    n_jobs=1
                )

                for actual_model in actual_models:
                    self.assertIsInstance(actual_model, Ti)

                self.assertEqual(num_actual_models, len(actual_models))

    @patch('darkgreybox.prefit.train_models')
    def test__prefit_models__raises_exception__when_there_are_no_valid_models_to_return(self, mock_train_models):

        prefit_splits = [([], [0, 1, 2]), ([], [3, 4, 5])]
        prefit_filter = None

        error_metric = MagicMock()

        model_1 = MagicMock()
        model_2 = MagicMock()

        models = [model_1, model_2]

        for (mock_train_models_return_value) in [
            pd.DataFrame({'models': [np.nan] * 4}),
            pd.DataFrame(),
        ]:
            with self.subTest(mock_train_models_return_value=mock_train_models_return_value):

                mock_train_models.return_value = mock_train_models_return_value

                with self.assertRaisesRegex(ValueError, 'No valid models found during prefit'):
                    prefit_models(
                        models,
                        X_train,
                        y_train,
                        error_metric,
                        prefit_splits,
                        prefit_filter,
                        n_jobs=1
                    )
