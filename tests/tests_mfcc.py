import unittest
import numpy as np

from .context import pyspeech
import pyspeech.dsp.processing as sp
import pyspeech.features.mfcc as mfcc
from pyspeech.configs import confs

class TestsMFCC(unittest.TestCase):

    def setUp(self):
        self.mfcc = mfcc.MFCC(13, 22)
        self.mfilter = mfcc.MelFilter(40, 4000, 300)
        self.emph = 0.97
        self.signnal = sp.Signal([], 0)

    def test_signal_200l_20hz(self):
        self.make_signal(32000, 8000)
        mfccs = self.extract()

    def test_has_energy(self):
        self.make_signal(32000, 8000)
        mfccs = self.extract()
        self.assertEqual(mfccs.shape[1], 14)

    def test_no_energy_append(self):
        self.make_signal(32000, 8000)
        confs['append_energy'] = False
        mfccs = self.extract()
        self.assertEqual(mfccs.shape[1], self.mfcc.ncep)

    def make_signal(self, slen, samplerate):
        amps = np.arange(slen)
        self.signal = sp.Signal(amps, samplerate)

    def extract(self):
        return mfcc.extract(self.signal, self.mfcc, self.mfilter, self.emph)

