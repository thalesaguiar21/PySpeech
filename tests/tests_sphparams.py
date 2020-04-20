import matplotlib.pyplot as plt
import os
import unittest
import math

import numpy as np
from scipy.io import wavfile

from .context import pyspeech
from pyspeech.conf import framing
from pyspeech.dsp import sphparams
from pyspeech.dsp.processing import Signal
from pyspeech.dsp import frame


SIGNALPATH = os.path.abspath('tests/voice/OSR_us_000_0011_8k.wav')


class TestsSphparams(unittest.TestCase):

    def setUp(self):
        framing['size'] = 25
        framing['stride'] = 10

    def test_st_energy(self):
        signal = read_signal()
        frames = get_frames(signal)
        energies = sphparams.st_energy(frames)
        flength = frame.size(signal.fs)
        stride = frame.stride(signal.fs)
        nframes = 1 + math.ceil((signal.size-flength) / stride)
        self.assertEqual(len(energies), nframes)

    def test_negative_energy(self):
        signal = get_frames()
        energies = list(sphparams.st_energy(signal))
        allpositive = all(egy > 0 for egy in energies)
        self.assertTrue(allpositive)

    def test_logenergy(self):
        signal = get_frames()
        lenergies = sphparams.log_energy(signal)

    def test_zrate(self):
        signal = read_signal()
        frames = get_frames(signal)
        rates = sphparams.zcr(frames, signal.fs)
        allpositive = all(rate >= 0 for rate in rates)
        self.assertTrue(allpositive)


def read_signal():
    fs, amps = wavfile.read(SIGNALPATH)
    return Signal(amps, fs)


def get_frames(signal=None):
    signal = read_signal() if signal is None else signal
    return frame.apply(signal)

