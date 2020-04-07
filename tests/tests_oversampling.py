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
        newdata, __ = oversample.by_duration([signal], fs, [1], 500)
        self.assertEqual(newdata.shape, (66, 4000))

    def test_by_duration_nlabels(self):
        fs, signal = wavfile.read(SIGNALPATH)
        newdata, labels = oversample.by_duration([signal], fs, [1], 500)
        self.assertEqual(newdata.shape[0], labels.size)

    def test_by_duration_labels_single(self):
        fs, signal = wavfile.read(SIGNALPATH)
        newdata, labels = oversample.by_duration([signal], fs, [1], 500)
        expected_labels = [1] * newdata.shape[0]
        isequal = np.all(expected_labels == labels)
        self.assertTrue(isequal)

    def test_by_duration_labels_mult(self):
        durations = [12, 10, 3.5, 40]
        fs, signals, ids = _build_signals(durations)
        __, labels = oversample.by_duration(signals, fs, ids, 500)
        expected_ids = [ids[0]]*24 + [ids[1]]*20 + [ids[2]]*7 + [ids[3]]*80
        isequal = np.all(expected_ids == labels)
        self.assertTrue(isequal)

    def test_by_duration_negative(self):
        fs, signal = wavfile.read(SIGNALPATH)
        with self.assertRaises(Warning):
            oversample.by_duration([signal], fs, [1], -300)

    def test_by_duration_bigger_than_signal(self):
        fs, signal = wavfile.read(SIGNALPATH)
        with self.assertRaises(Warning):
            oversample.by_duration([signal], fs, [1], 40000)
    
    def test_by_shortest(self):
        durations = [12, 10, 3.5, 40]
        fs, signals, ids = _build_signals(durations)
        newdata, __ = oversample.by_shortest(signals, fs, ids)
        self.assertEqual(newdata.shape, (20, durations[2]*fs))

    def test_by_scalar_shortest(self):
        durations = [12, 10, 3.5, 40]
        fs, signals, ids = _build_signals(durations)
        newdata, __ = oversample.by_scalar_shortest(signals, fs, ids, 2)
        self.assertEqual(newdata.shape, (38, durations[2]/2 * fs))


def _build_signals(durations):
    fs = 8000
    amp = 300
    ids = [0, 1, 0, 1]
    signal_sizes = [math.ceil(fs*sec) for sec in durations]
    signals = [amp * np.random.rand(size) for size in signal_sizes]
    return fs, signals, ids

