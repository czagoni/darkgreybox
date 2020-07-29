import numpy as np
from numpy.testing import assert_array_equal

import unittest 

from darkgreybox.model import DarkGreyModel, TiTh, TiTeTh, TiTeThRia


class DGMTest(DarkGreyModel):

    def model(self, params, **X):

        Y = np.zeros(self.num_rec)
        Y[0] = params['Y0']
        R = params['R'].value
        Z = X['Z']

        for i in range(1, self.num_rec):
            Y[i] = Z[i] * R

        return (Y, )   


class DarkGreyModelTest(unittest.TestCase):

    def test_fit(self):

        y = np.array([1, 2])
        params = {'Y0': {'value': 10},
                  'R': {'value': 0.01}}
        X = {'Z': np.array([10, 20])}
        
        result = DGMTest(y=y, rec_duration=1, X=X, params=params, method='nelder').fit()

        self.assertAlmostEqual(1, result.params['Y0'].value, places=3)
        self.assertAlmostEqual(0.1, result.params['R'].value, places=3)
       
    def test_fit_min_max(self):

        y = np.array([1, 2])
        params = {'Y0': {'value': 10, 'min': 2},
                  'R': {'value': 0.01, 'max': 0.05}}
        X = {'Z': np.array([10, 20])}
        
        result = DGMTest(y=y, rec_duration=1, X=X, params=params, method='nelder').fit()

        self.assertAlmostEqual(2, result.params['Y0'].value, places=3)
        self.assertAlmostEqual(0.05, result.params['R'].value, places=3)

    def test_fit_vary(self):

        y = np.array([1, 2])
        params = {'Y0': {'value': 10, 'vary': False},
                  'R': {'value': 0.01, 'vary': True}}
        X = {'Z': np.array([10, 20])}
        
        result = DGMTest(y=y, rec_duration=1, X=X, params=params, method='nelder').fit()

        self.assertAlmostEqual(10, result.params['Y0'].value, places=3)
        self.assertAlmostEqual(0.1, result.params['R'].value, places=3)


class TiThTest(unittest.TestCase):

    def test_model(self):

        y = np.array([10, 10, 20])

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
        
        m = TiTh(y=y, rec_duration=1, X=X, params=params, method='nelder')
        Ti, Th = m.model(m.params, **m.X)

        assert_array_equal(np.array([10, 30, 10]), Ti)
        assert_array_equal(np.array([20, 30, 30]), Th)
        
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
        
        m = TiTh(y=y, rec_duration=1, X=X, params=params, method='nelder')
        result = m.fit()

        for k, v in params.items():
            self.assertAlmostEqual(v['value'], result.params[k].value, places=3)

        assert_array_equal(y, m.model(result.params, **X)[0])


class TiTeThTest(unittest.TestCase):

    def test_model(self):

        y = np.array([10, 10, 20])

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
        
        m = TiTeTh(y=y, rec_duration=1, X=X, params=params, method='nelder')
        Ti, Te, Th = m.model(m.params, **m.X)

        assert_array_equal(np.array([20, 12.5, 16.5625]), Ti)
        assert_array_equal(np.array([10, 20, 10]), Te)
        assert_array_equal(np.array([10, 13.75, 13.59375]), Th)

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
        
        m = TiTeTh(y=y, rec_duration=1, X=X, params=params, method='nelder')
        result = m.fit()

        for k, v in params.items():
            self.assertAlmostEqual(v['value'], result.params[k].value, places=3)

        assert_array_equal(y, m.model(result.params, **X)[0])


class TiTeThRiaTest(unittest.TestCase):

    def test_model(self):

        y = np.array([10, 10, 20])

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
        
        m = TiTeThRia(y=y, rec_duration=1, X=X, params=params, method='nelder')
        Ti, Te, Th = m.model(m.params, **m.X)

        assert_array_equal(np.array([20, 11.875, 16.2890625]), Ti)
        assert_array_equal(np.array([10, 20, 9.375]), Te)
        assert_array_equal(np.array([10, 13.75, 13.515625]), Th)

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
        
        m = TiTeThRia(y=y, rec_duration=1, X=X, params=params, method='nelder')
        result = m.fit()

        for k, v in params.items():
            self.assertAlmostEqual(v['value'], result.params[k].value, places=3)

        assert_array_equal(y, m.model(result.params, **X)[0])