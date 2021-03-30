import numpy as np
from numpy.testing import assert_array_equal

import unittest

from darkgreybox.model import (DarkGreyModel,
                               Ti, TiTe, TiTh, TiTeTh, TiTeThRia,
                               DarkGreyModelResult)


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

    def test_fit(self):

        y = np.array([1, 2])
        params = {'A0': {'value': 10},
                  'B': {'value': 0.01}}
        X = {'C': np.array([10, 20])}

        m = DGMTest(params=params, rec_duration=1).fit(X=X, y=y, method='nelder')

        self.assertAlmostEqual(1, m.result.params['A0'].value, places=3)
        self.assertAlmostEqual(0.1, m.result.params['B'].value, places=3)

    def test_fit_min_max(self):

        y = np.array([1, 2])
        params = {'A0': {'value': 10, 'min': 2},
                  'B': {'value': 0.01, 'max': 0.05}}
        X = {'C': np.array([10, 20])}

        m = DGMTest(params=params, rec_duration=1).fit(X=X, y=y, method='nelder')

        self.assertAlmostEqual(2, m.result.params['A0'].value, places=3)
        self.assertAlmostEqual(0.05, m.result.params['B'].value, places=3)

    def test_fit_vary(self):

        y = np.array([1, 2])
        params = {'A0': {'value': 10, 'vary': False},
                  'B': {'value': 0.01, 'vary': True}}
        X = {'C': np.array([10, 20])}

        m = DGMTest(params=params, rec_duration=1).fit(X=X, y=y, method='nelder')

        self.assertAlmostEqual(10, m.result.params['A0'].value, places=3)
        self.assertAlmostEqual(0.1, m.result.params['B'].value, places=3)

    def test_fit_ic_params(self):

        y = np.array([1, 2])
        params = {'A0': {'value': 10, 'vary': False},
                  'B': {'value': 0.01, 'vary': True}}
        X = {'C': np.array([10, 20])}
        ic_params = {'A0': 15}

        m = DGMTest(params=params, rec_duration=1).fit(X=X, y=y, method='nelder', ic_params=ic_params)

        self.assertAlmostEqual(15, m.result.params['A0'].value, places=3)
        self.assertAlmostEqual(0.1, m.result.params['B'].value, places=3)

    def test_fit_ic_params_missing_key(self):

        y = np.array([1, 2])
        params = {'A0': {'value': 10, 'vary': False},
                  'B': {'value': 0.01, 'vary': True}}
        X = {'C': np.array([10, 20])}
        ic_params = {'A0': 15, 'A10': 150}

        m = DGMTest(params=params, rec_duration=1).fit(X=X, y=y, method='nelder', ic_params=ic_params)

        self.assertAlmostEqual(15, m.result.params['A0'].value, places=3)
        self.assertAlmostEqual(0.1, m.result.params['B'].value, places=3)

    def test_fit_custom_obj_func(self):

        def obj_func(params, *args, **kwargs):
            return ((kwargs['model'](params=params, X=kwargs['X']).Z - kwargs['y']) ** 2).ravel()

        y = np.array([1, 2])
        params = {'A0': {'value': 10, 'vary': False},
                  'B': {'value': 0.01, 'vary': True}}
        X = {'C': np.array([10, 20])}

        m = DGMTest(params=params, rec_duration=1).fit(X=X, y=y, method='nelder', obj_func=obj_func)

        self.assertAlmostEqual(10, m.result.params['A0'].value, places=3)
        self.assertAlmostEqual(0.1, m.result.params['B'].value, places=3)

    def test_predict(self):

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

    def test_predict_ic_params(self):

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

    def test_lock(self):

        params = {'A0': {'value': 10},
                  'B': {'value': 0.01}}

        m = DGMTest(params=params, rec_duration=1).lock()

        for param in m.params.keys():
            self.assertFalse(m.params[param].vary)


class TiTest(unittest.TestCase):

    def test_model(self):

        params = {
            'Ti0': {'value': 10},
            'Ria': {'value': 4},
            'Ci': {'value': 0.25},
        }

        X = {
            'Ta': np.array([10, 10, 10]),
            'Ph': np.array([10, 0, 0]),
        }

        m = Ti(params=params, rec_duration=1)
        actual_result = m.model(m.params, X)

        assert_array_equal(np.array([10, 50, 10]), actual_result.Ti)
        assert_array_equal(actual_result.Z, actual_result.Ti)

    def test_fit(self):

        y = np.array([10, 10, 20])

        params = {
            'Ti0': {'value': 10},
            'Ria': {'value': 1},
            'Ci': {'value': 1},
        }

        X = {
            'Ta': np.array([10, 10, 10]),
            'Ph': np.array([0, 10, 0]),
        }

        m = Ti(params=params, rec_duration=1) \
            .fit(X=X, y=y, method='nelder')

        for k, v in params.items():
            self.assertAlmostEqual(v['value'], m.result.params[k].value, places=3)

        assert_array_equal(y, m.model(m.result.params, X).Z)


class TiThTest(unittest.TestCase):

    def test_model(self):

        params = {
            'Ti0': {'value': 10},
            'Th0': {'value': 20},
            'Rih': {'value': 2},
            'Ria': {'value': 4},
            'Ci': {'value': 0.25},
            'Ch': {'value': 0.5}
        }

        X = {
            'Ta': np.array([10, 10, 10]),
            'Ph': np.array([10, 0, 0]),
        }

        m = TiTh(params=params, rec_duration=1)
        actual_result = m.model(m.params, X)

        assert_array_equal(np.array([10, 30, 10]), actual_result.Ti)
        assert_array_equal(np.array([20, 30, 30]), actual_result.Th)
        assert_array_equal(actual_result.Z, actual_result.Ti)

    def test_fit(self):

        y = np.array([10, 10, 20])

        params = {
            'Ti0': {'value': 10},
            'Th0': {'value': 10},
            'Rih': {'value': 1},
            'Ria': {'value': 1},
            'Ci': {'value': 1},
            'Ch': {'value': 1}
        }

        X = {
            'Ta': np.array([10, 10, 10]),
            'Ph': np.array([10, 0, 0]),
        }

        m = TiTh(params=params, rec_duration=1) \
            .fit(X=X, y=y, method='nelder')

        for k, v in params.items():
            self.assertAlmostEqual(v['value'], m.result.params[k].value, places=3)

        assert_array_equal(y, m.model(m.result.params, X).Z)


# class TiTeTest(unittest.TestCase):

#     def test_model(self):

#         params = {
#             'Ti0': {'value': 10},
#             'Te0': {'value': 20},
#             'Rie': {'value': 2},
#             'Rea': {'value': 4},
#             'Ci': {'value': 0.25},
#             'Ce': {'value': 0.5}
#         }

#         X = {
#             'Ta': np.array([10, 10, 10]),
#             'Ph': np.array([10, 0, 0]),
#         }

#         m = TiTe(params=params, rec_duration=1)
#         actual_result = m.model(m.params, X)

#         assert_array_equal(np.array([10, 30, 10]), actual_result.Ti)
#         assert_array_equal(np.array([20, 30, 30]), actual_result.Te)
#         assert_array_equal(actual_result.Z, actual_result.Ti)

#     def test_fit(self):

#         y = np.array([10, 10, 20])

#         params = {
#             'Ti0': {'value': 10},
#             'Te0': {'value': 10},
#             'Rie': {'value': 1},
#             'Rea': {'value': 1},
#             'Ci': {'value': 1},
#             'Ce': {'value': 1}
#         }

#         X = {
#             'Ta': np.array([10, 10, 10]),
#             'Ph': np.array([10, 0, 0]),
#         }

#         m = TiTe(params=params, rec_duration=1) \
#                 .fit(X=X, y=y, method='nelder')

#         for k, v in params.items():
#             self.assertAlmostEqual(v['value'], m.result.params[k].value, places=3)

#         assert_array_equal(y, m.model(m.result.params, X).Z)


class TiTeTest(unittest.TestCase):

    def test_model(self):

        params = {
            'Ti0': {'value': 20},
            'Te0': {'value': 10},
            'Rie': {'value': 1},
            'Rea': {'value': 4},
            'Ci': {'value': 2},
            'Ce': {'value': 1},
        }

        X = {
            'Ta': np.array([10, 10, 10]),
            'Ph': np.array([10, 0, 0]),
        }

        m = TiTe(params=params, rec_duration=1)
        actual_result = m.model(m.params, X)

        assert_array_equal(np.array([20, 20, 20]), actual_result.Ti)
        assert_array_equal(np.array([10, 20, 17.5]), actual_result.Te)

    def test_fit(self):

        y = np.array([20, 20, 20])

        params = {
            'Ti0': {'value': 20},
            'Te0': {'value': 10},
            'Rie': {'value': 1},
            'Rea': {'value': 4},
            'Ci': {'value': 2},
            'Ce': {'value': 1},
        }

        X = {
            'Ta': np.array([10, 10, 10]),
            'Ph': np.array([10, 0, 0]),
        }

        m = TiTe(params=params, rec_duration=1) \
            .fit(X=X, y=y, method='nelder')

        for k, v in params.items():
            self.assertAlmostEqual(v['value'], m.result.params[k].value, places=3)

        assert_array_equal(y, m.model(m.result.params, X).Z)


class TiTeThTest(unittest.TestCase):

    def test_model(self):

        params = {
            'Ti0': {'value': 20},
            'Te0': {'value': 10},
            'Th0': {'value': 10},
            'Rih': {'value': 2},
            'Rie': {'value': 1},
            'Rea': {'value': 4},
            'Ci': {'value': 2},
            'Ce': {'value': 1},
            'Ch': {'value': 4}
        }

        X = {
            'Ta': np.array([10, 10, 10]),
            'Ph': np.array([10, 0, 0]),
        }

        m = TiTeTh(params=params, rec_duration=1)
        actual_result = m.model(m.params, X)

        assert_array_equal(np.array([20, 12.5, 16.5625]), actual_result.Ti)
        assert_array_equal(np.array([10, 20, 10]), actual_result.Te)
        assert_array_equal(np.array([10, 13.75, 13.59375]), actual_result.Th)

    def test_fit(self):

        y = np.array([10, 10, 20])

        params = {
            'Ti0': {'value': 10},
            'Te0': {'value': 10},
            'Th0': {'value': 10},
            'Rih': {'value': 1},
            'Rie': {'value': 1},
            'Rea': {'value': 1},
            'Ci': {'value': 1},
            'Ce': {'value': 1},
            'Ch': {'value': 1}
        }

        X = {
            'Ta': np.array([10, 10, 10]),
            'Ph': np.array([10, 0, 0]),
        }

        m = TiTeTh(params=params, rec_duration=1) \
            .fit(X=X, y=y, method='nelder')

        for k, v in params.items():
            self.assertAlmostEqual(v['value'], m.result.params[k].value, places=3)

        assert_array_equal(y, m.model(m.result.params, X).Z)


class TiTeThRiaTest(unittest.TestCase):

    def test_model(self):

        params = {
            'Ti0': {'value': 20},
            'Te0': {'value': 10},
            'Th0': {'value': 10},
            'Rih': {'value': 2},
            'Rie': {'value': 1},
            'Rea': {'value': 4},
            'Ria': {'value': 8},
            'Ci': {'value': 2},
            'Ce': {'value': 1},
            'Ch': {'value': 4}
        }

        X = {
            'Ta': np.array([10, 10, 10]),
            'Ph': np.array([10, 0, 0]),
        }

        m = TiTeThRia(params=params, rec_duration=1)
        actual_result = m.model(m.params, X)

        assert_array_equal(np.array([20, 11.875, 16.2890625]), actual_result.Ti)
        assert_array_equal(np.array([10, 20, 9.375]), actual_result.Te)
        assert_array_equal(np.array([10, 13.75, 13.515625]), actual_result.Th)

    def test_fit(self):

        y = np.array([10, 10, 20])

        params = {
            'Ti0': {'value': 10},
            'Te0': {'value': 10},
            'Th0': {'value': 10},
            'Rih': {'value': 1},
            'Rie': {'value': 1},
            'Rea': {'value': 1},
            'Ria': {'value': 1},
            'Ci': {'value': 1},
            'Ce': {'value': 1},
            'Ch': {'value': 1}
        }

        X = {
            'Ta': np.array([10, 10, 10]),
            'Ph': np.array([10, 0, 0]),
        }

        m = TiTeThRia(params=params, rec_duration=1) \
            .fit(X=X, y=y, method='nelder')

        for k, v in params.items():
            self.assertAlmostEqual(v['value'], m.result.params[k].value, places=3)

        assert_array_equal(y, m.model(m.result.params, X).Z)
