import unittest

import numpy as np

from .context import pyspeech
from pyspeech.configs import confs
import pyspeech.dsp.silence as psil
import pyspeech.dsp.processing as sp


class TestsSilRemove(unittest.TestCase):

    def test_remove_max_amp_20_from_40(self):
        confs['frame_size'] = 200  # ms
        signal = sp.Signal(np.arange(-20, 20), 20)

        voiced_signal = psil.remove_silence(signal, 6)
        expected = np.append(signal.amps[:8], signal.amps[32:])
        self.assertNumpyArrayEqual(expected, voiced_signal.amps)

    def assertNumpyArrayEqual(self, left, right):
        left_list = left.tolist()
        right_list = right.tolist()
        self.assertListEqual(left_list, right_list)

