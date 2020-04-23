import os

import unittest
from scipy.io import wavfile
import numpy as np

from .context import pyspeech
from pyspeech.dsp import frame as fr
from pyspeech.dsp.processing import Signal
from pyspeech import conf


class TestsFrame(unittest.TestCase):

    def test_split_19_frames_dim_4(self):
        conf.framing['size'] = 200  # ms
        conf.framing['stride'] = 100  # ms
        signal = Signal(np.arange(-20, 20), 20)
        frames = fr.apply(signal)
        self.assertEqual(19, frames.shape[0])

    def test_padding_0(self):
        conf.framing['size'] = 200  # ms
        conf.framing['stride'] = 100  # ms
        signal = Signal(np.arange(-20, 20), 20)
        frames = fr.apply(signal)
        self.assertEqual(76, frames.size)

    def test_padding_2_zeros(self):
        conf.framing['size'] = 300  # ms
        conf.framing['stride'] = 150  # ms
        signal = Signal(np.arange(-20, 20), 20)
        frames = fr.apply(signal)
        self.assertEqual(78, frames.size)
        self.assertEqual(0, frames[-1, -1])
        self.assertEqual(0, frames[-1, -2])

