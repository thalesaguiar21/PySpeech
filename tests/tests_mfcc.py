import unittest
import numpy as np

from .context import pyspeech
import pyspeech.dsp.processing as sp
import pyspeech.features.mfcc as mfcc
from pyspeech.configs import confs


class TestsMFCC(unittest.TestCase):

    def setUp(self):
        self.nfilt = 40
        self.ncep = 13
        self.highfreq = 4000
        self.lowfreq = 300
        self.emph = 0.97
        self.signnal = sp.Signal([], 0)
        self.ceplift = 22

    def test_signal_200l_20hz(self):
        self.make_signal(32000, 8000)
        self.extract()

    def make_signal(self, slen, samplerate):
        amps = np.arange(slen)
        self.signal = sp.Signal(amps, samplerate)

    def extract(self):
        mfcc.extract(self.signal, self.nfilt, self.ncep, self.emph,
                     self.ceplift, self.lowfreq, self.highfreq)

