import matplotlib.pyplot as plt
import os
import unittest
import math

import numpy as np
from scipy.io import wavfile

from .context import pyspeech
from pyspeech.dsp import transform
from pyspeech.configs import confs
from pyspeech.dsp.processing import Signal
from pyspeech.dsp import frame


SIGNALPATH = os.path.abspath('tests/voice/OSR_us_000_0011_8k.wav')


class TestsTrasnform(unittest.TestCase):

    def setUp(self):
        confs['frame_size'] = 25
        confs['frame_stride'] = 10

    def test_st_energy(self):
        signal = read_signal()
        energies = list(transform.short_time_energy(signal))
        flen = frame.flength(signal)
        stride = frame.stride(signal)
        nframes = 1 + math.ceil((signal.size-flen) / stride)
        self.assertEqual(len(energies), nframes)

    def test_negative_energy(self):
        signal = read_signal()
        energies = list(transform.short_time_energy(signal))
        allpositive = all(egy > 0 for egy in energies)
        self.assertTrue(allpositive)

    def test_logenergy(self):
        signal = read_signal()
        lenergies = transform.log_energy(signal)

    def test_zrate(self):
        signal = read_signal()
        rates = transform.zcr(signal)
        allpositive = all(rate >= 0 for rate in rates)
        self.assertTrue(allpositive)


def read_signal():
    fs, amps = wavfile.read(SIGNALPATH)
    return Signal(amps, fs)

