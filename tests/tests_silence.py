import os
import unittest

import numpy as np
from scipy.io import wavfile

from .context import pyspeech
from pyspeech.conf import framing 
import pyspeech.dsp.silence as psil
import pyspeech.dsp.processing as sp


SIGNALPATH = os.path.abspath('tests/voice/OSR_us_000_0011_8k.wav')


class TestsSilRemove(unittest.TestCase):

    def test_remove_max_amp_20_from_40(self):
        framing['size'] = 200  # ms
        framing['stride'] = 100  # ms
        signal = sp.Signal(np.arange(-20, 20), 20)
        psil.remove(signal)


def read_signal():
    fs, amps = wavfile.read(SIGNALPATH)
    return sp.Signal(amps, fs)
