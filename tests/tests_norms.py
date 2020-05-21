import unittest

import numpy as np

from .context import pyspeech
from pyspeech.features import norms


class TestsAvgReduction(unittest.TestCase):
    
    def test_avg_list(self):
        data = [i for i in range(20)]
        normdata = norms.avg_reduction(data)


    def test_avg_is_reducing(self):
        data = [i for i in range(20)]
        normdata = norms.avg_reduction(data)
        allnormed = all(abs(normdata + 9.5 - np.array(data)) <= 0.0001)
        self.assertTrue(allnormed)

    def test_avg_constant_data(self):
        data = [1] * 20
        normdata = norms.avg_reduction(data)
        allnormed = all(0 <= nd <= 0.001 for nd in normdata)
        self.assertTrue(allnormed)

    def test_avg_zeros(self):
        data = np.zeros(20)
        normdata = norms.avg_reduction(data)
        self.assertTrue(all(normdata == 0))

    def test_avg_negative_data(self):
        data = np.arange(20) * -1
        normdata = norms.avg_reduction(data)
        negative_avg = all(abs(normdata - 9.5 - np.array(data)) <= 0.0001)
        self.assertTrue(negative_avg)

    def test_avg_2d(self):
        data = np.ones((3, 4)) + np.arange(4)
        normdata = norms.avg_reduction(data)
        has_reduced_at_cols = np.all(abs(normdata) <= 0.0001)
        self.assertTrue(has_reduced_at_cols)

    def test_avg_col_matrix(self):
        data = np.arange(5).reshape(5, 1)
        normdata = norms.avg_reduction(data)
        has_reduced = np.all(abs(normdata + 2 - np.array(data)) <= 0.0001)
        self.assertTrue(has_reduced)

    def test_avg_empty_col(self):
        data = np.empty(5).reshape(5, 1)
        normdata = norms.avg_reduction(data)
        self.assertTrue(np.all(data == []))


class TestsMeanNormalise(unittest.TestCase):

    def test_list(self):
        data = [i for i in range(20)]
        normdata = norms.mean_normalise(data)

    def test_is_reducing(self):
        data = [i for i in range(20)]
        normdata = norms.mean_normalise(data)
        is_applied = abs(normdata * max(data) + 9.5 - np.array(data)) <= 0.0001
        self.assertTrue(all(is_applied))

    def tests_zero(self):
        data = np.zeros(50)
        normdata = norms.mean_normalise(data)
        self.assertTrue(np.all(normdata == 0))

    def tests_zero2d(self):
        data = np.zeros((10, 5))
        normdata = norms.mean_normalise(data)
        self.assertTrue(np.all(normdata == 0))

    def tests_close_zero_right(self):
        data = np.zeros((10, 5)) + 1e-5
        normdata = norms.mean_normalise(data)
        self.assertTrue(np.all(normdata == 0))

    def tests_close_zero_left(self):
        data = np.zeros((10, 5)) - 1e-5
        normdata = norms.mean_normalise(data)
        self.assertTrue(np.all(normdata == 0))


