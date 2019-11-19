import unittest
import numpy as np

from .context import pyspeech
import pyspeech.features.derivs as derivs


class TestsDeltas(unittest.TestCase):

    def test_delta_10frames_3dim(self):
        frames = np.ones((10, 3))
        deltas = derivs.delta(frames, smooth=2)
        for delta in deltas:
            self.assertListEqual([0, 0, 0], delta.tolist())

    def test_10frames_5dim_sequence(self):
        frames = np.arange(50).reshape((10, 5))
        deltas = derivs.delta(frames, smooth=2)

