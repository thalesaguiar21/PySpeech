import math
import unittest

from .context import pyspeech
import pyspeech.dsp.processing as sp


class TestsMinNFFT(unittest.TestCase):

    def test_inside(self):
        nfft = sp.find_best_nfft(16000, 0.025)
        self.assertEqual(512, nfft)

    def test_negative_freq(self):
        with self.assertRaises(ValueError):
            sp.find_best_nfft(-30, 0.025)

    def test_negative_winlen(self):
        with self.assertRaises(ValueError):
            sp.find_best_nfft(1600, -1)

    def test_nfft_power_of_2(self):
        nfft = sp.find_best_nfft(13093, 0.025)
        self.assertEqual(is_power_of_2(nfft), True)




def is_power_of_2(x):
    return math.ceil(math.log2(x)) == math.floor(math.log2(x))

