import os
import unittest

import numpy as np
from scipy.io import wavfile

from tests import signal01
from .context import pyspeech
from pyspeech import conf
import pyspeech.dsp.silence as silence
import pyspeech.dsp.processing as sp
from pyspeech.dsp import frame


class TestsSilRemove(unittest.TestCase):

    def setUp(self):
        _configure()

    def test_leq_signal_size(self):
        voiced = silence.remove(signal01)
        flen = frame.size(signal01.fs, 25)
        wnd_len = flen / conf.fir['order']
        max_len_signal = signal01.size + flen + wnd_len - 2
        self.assertLessEqual(voiced.size,  max_len_signal)
        self.assertGreater(voiced.size, 0)

    def test_voiced_smaller(self):
        amps = np.zeros(32000)
        amps[2000:4000] += 1
        signal = sp.Signal(amps, 16000)
        voiced = silence.remove(signal)
        self.assertLessEqual(voiced.size, 8000)

    def test_zero_silence_only_input(self):
        signal = sp.Signal(np.zeros(32000), 16000)
        voiced = silence.remove(signal)
        self.assertGreater(voiced.size, 0)
        pading = 80
        self.assertLessEqual(voiced.size - pading, signal.size)

    def test_silence_empty_signal(self):
        signal = sp.Signal([], 16000)
        voiced = silence.remove(signal)
        self.assertEqual(voiced.size, 0)

    def test_silence_no_sil_high(self):
        signal = sp.Signal(np.zeros(32000) + 1e5, 8000)
        voiced = silence.remove(signal)
        pading = 40
        self.assertLessEqual(voiced.size - pading, signal.size)

    def test_silence_low_only(self):
        signal = sp.Signal(np.zeros(8000) + 1e-10, 2000)
        voiced = silence.remove(signal)
        self.assertGreater(voiced.size, 0)


def _configure(size=25, stride=10):
    conf.framing['size'] = size
    conf.framing['stride'] = stride

