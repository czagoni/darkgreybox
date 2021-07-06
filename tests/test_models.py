import unittest

import numpy as np
from numpy.testing import assert_array_equal

from darkgreybox.models import Ti, TiTe, TiTeTh, TiTeThRia, TiTh


class TiTest(unittest.TestCase):

    def test__model(self):

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

    def test__fit(self):

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

    def test__model(self):

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

    def test__fit(self):

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


class TiTeTest(unittest.TestCase):

    def test__model(self):

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

    def test__fit(self):

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

    def test__model(self):

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

    def test__fit(self):

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

    def test__model(self):

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

    def test__fit(self):

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
