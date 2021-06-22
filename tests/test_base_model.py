import unittest

import numpy as np
from lmfit import Parameters

from darkgreybox.base_model import DarkGreyModel, DarkGreyModelResult


class DGMTest(DarkGreyModel):

    def model(self, params, X):

        num_rec = len(X['C'])

        A = np.zeros(num_rec)
        A[0] = params['A0']
        B = params['B'].value
        C = X['C']

        for i in range(1, num_rec):
            A[i] = C[i] * B

        return DarkGreyModelResult(A)


class DarkGreyModelTest(unittest.TestCase):

    def test__init__parameters__from_dict(self):

        params = {'A0': {'value': 10},
                  'B': {'value': 0.01}}

        m = DGMTest(params=params, rec_duration=1)

        self.assertIsInstance(m.params, Parameters)
        self.assertEqual(m.params['A0'].value, 10)
        self.assertEqual(m.params['B'].value, 0.01)

    def test__init__parameters__from_paramters_object(self):

        params = Parameters()
        params.add('A0', value=10)
        params.add('B', value=0.01)

        m = DGMTest(params=params, rec_duration=1)

        self.assertIsInstance(m.params, Parameters)
        self.assertEqual(m.params['A0'].value, 10)
        self.assertEqual(m.params['B'].value, 0.01)

    def test__fit(self):

        y = np.array([1, 2])
        params = {'A0': {'value': 10},
                  'B': {'value': 0.01}}
        X = {'C': np.array([10, 20])}

        m = DGMTest(params=params, rec_duration=1).fit(X=X, y=y, method='nelder')

        self.assertAlmostEqual(1, m.result.params['A0'].value, places=3)
        self.assertAlmostEqual(0.1, m.result.params['B'].value, places=3)

    def test__fit__min_max(self):

        y = np.array([1, 2])
        params = {'A0': {'value': 10, 'min': 2},
                  'B': {'value': 0.01, 'max': 0.05}}
        X = {'C': np.array([10, 20])}

        m = DGMTest(params=params, rec_duration=1).fit(X=X, y=y, method='nelder')

        self.assertAlmostEqual(2, m.result.params['A0'].value, places=3)
        self.assertAlmostEqual(0.05, m.result.params['B'].value, places=3)

    def test__fit__vary(self):

        y = np.array([1, 2])
        params = {'A0': {'value': 10, 'vary': False},
                  'B': {'value': 0.01, 'vary': True}}
        X = {'C': np.array([10, 20])}

        m = DGMTest(params=params, rec_duration=1).fit(X=X, y=y, method='nelder')

        self.assertAlmostEqual(10, m.result.params['A0'].value, places=3)
        self.assertAlmostEqual(0.1, m.result.params['B'].value, places=3)

    def test__fit__ic_params(self):

        y = np.array([1, 2])
        params = {'A0': {'value': 10, 'vary': False},
                  'B': {'value': 0.01, 'vary': True}}
        X = {'C': np.array([10, 20])}
        ic_params = {'A0': 15}

        m = DGMTest(params=params, rec_duration=1).fit(X=X, y=y, method='nelder', ic_params=ic_params)

        self.assertAlmostEqual(15, m.result.params['A0'].value, places=3)
        self.assertAlmostEqual(0.1, m.result.params['B'].value, places=3)

    def test__fit__ic_params_missing_key(self):

        y = np.array([1, 2])
        params = {'A0': {'value': 10, 'vary': False},
                  'B': {'value': 0.01, 'vary': True}}
        X = {'C': np.array([10, 20])}
        ic_params = {'A0': 15, 'A10': 150}

        m = DGMTest(params=params, rec_duration=1).fit(X=X, y=y, method='nelder', ic_params=ic_params)

        self.assertAlmostEqual(15, m.result.params['A0'].value, places=3)
        self.assertAlmostEqual(0.1, m.result.params['B'].value, places=3)

    def test__fit__custom_obj_func(self):

        def obj_func(params, *args, **kwargs):
            return ((kwargs['model'](params=params, X=kwargs['X']).Z - kwargs['y']) ** 2).ravel()

        y = np.array([1, 2])
        params = {'A0': {'value': 10, 'vary': False},
                  'B': {'value': 0.01, 'vary': True}}
        X = {'C': np.array([10, 20])}

        m = DGMTest(params=params, rec_duration=1).fit(X=X, y=y, method='nelder', obj_func=obj_func)

        self.assertAlmostEqual(10, m.result.params['A0'].value, places=3)
        self.assertAlmostEqual(0.1, m.result.params['B'].value, places=3)

    def test__predict(self):

        y = np.array([1, 2])
        params = {'A0': {'value': 10},
                  'B': {'value': 0.01}}
        X = {'C': np.array([10, 20])}

        actual_result = DGMTest(params=params, rec_duration=1) \
            .fit(X=X, y=y, method='nelder') \
            .predict({'C': np.array([10, 20, 30])})

        self.assertAlmostEqual(1, actual_result.Z[0], places=3)
        self.assertAlmostEqual(2, actual_result.Z[1], places=3)
        self.assertAlmostEqual(3, actual_result.Z[2], places=3)

    def test__predict__ic_params(self):

        y = np.array([1, 2])
        params = {'A0': {'value': 10},
                  'B': {'value': 0.01}}
        X = {'C': np.array([10, 20])}
        ic_params = {'A0': 15}

        actual_result = DGMTest(params=params, rec_duration=1) \
            .fit(X=X, y=y, method='nelder') \
            .predict({'C': np.array([10, 20, 30])}, ic_params=ic_params)

        self.assertAlmostEqual(15, actual_result.Z[0], places=3)
        self.assertAlmostEqual(2, actual_result.Z[1], places=3)
        self.assertAlmostEqual(3, actual_result.Z[2], places=3)

    def test__lock(self):

        params = {'A0': {'value': 10},
                  'B': {'value': 0.01}}

        m = DGMTest(params=params, rec_duration=1).lock()

        for param in m.params.keys():
            self.assertFalse(m.params[param].vary)
