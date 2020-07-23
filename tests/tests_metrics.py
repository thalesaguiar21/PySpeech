import unittest

from .context import pyspeech
import pyspeech.dsp.metrics as spmet


class TestsMetrics(unittest.TestCase):

    def test_m2hz(self):
        mel = spmet.hz2mel(8000)
        self.assertAlmostEqual(mel, 2840.02, 1)

    def test_hz2m(self):
        hz = spmet.mel2hz(700)
        self.assertAlmostEqual(hz, 602.70, 1)

    def test_twoway_hz2mel_mel2hz(self):
        mel = spmet.hz2mel(8000)
        hz = spmet.mel2hz(mel)
        self.assertAlmostEqual(8000, hz)

