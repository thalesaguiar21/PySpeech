import unittest

from .context import pyspeech
import pyspeech.dsp.metrics as spmet


class TestsMetrics(unittest.TestCase):

    def test_m2hz(self):
        mel = spmet.hz_to_mel(8000)
        self.assertAlmostEqual(mel, 2840.023046708319)

    def test_hz2m(self):
        hz = spmet.mel_to_hz(700)
        self.assertAlmostEqual(hz, 602.709434226635865)

    def test_twoway_hz2mel_mel2hz(self):
        mel = spmet.hz_to_mel(8000)
        hz = spmet.mel_to_hz(mel)
        self.assertAlmostEqual(8000, hz)

