import os
import math
import unittest

from scipy.io import wavfile
import numpy as np

from .context import pyspeech
from pyspeech.data import oversample


SIGNALPATH = os.path.abspath('tests/voice/OSR_us_000_0011_8k.wav')


class TestsOversampling(unittest.TestCase):

    def test_by_duration_shape(self):
        fs, signal = wavfile.read(SIGNALPATH)
        newdata = oversample.by_duration([signal], fs, 500)
        self.assertEqual(newdata.shape, (66, 4000))

    def test_by_duration_negative(self):
        fs, signal = wavfile.read(SIGNALPATH)
        with self.assertRaises(ValueError):
            oversample.by_duration([signal], fs, -300)

    def test_by_duration_bigger_than_signal(self):
        fs, signal = wavfile.read(SIGNALPATH)
        with self.assertRaises(Warning):
            oversample.by_duration([signal], fs, 40000)
    
    def test_by_shortest(self):
        durations = [12, 10, 3.5, 40]
        fs, signals = _build_signals(durations)
        newdata = oversample.by_shortest(signals, fs)
        self.assertEqual(newdata.shape, (20, durations[2]*fs))

    def test_by_scalar_shortest(self):
        durations = [12, 10, 3.5, 40]
        fs, signals = _build_signals(durations)
        newdata = oversample.by_scalar_shortest(signals, fs, 2)
        self.assertEqual(newdata.shape, (38, durations[2]/2 * fs))


def _build_signals(durations):
    fs = 8000
    amp = 300
    signal_sizes = [math.ceil(fs*sec) for sec in durations]
    signals = [amp * np.random.rand(size) for size in signal_sizes]
    return fs, signals

