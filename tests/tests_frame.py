import os

import unittest
from scipy.io import wavfile

from .context import pyspeech
from pyspeech.dsp import frame as fr
from pyspeech.configs import confs


SIGNALPATH = os.path.abspath('tests/voice/OSR_us_000_0011_8k.wav')


class TestsFrame(unittest.TestCase):

    def test_split_19_frames_dim_4(self):
        confs['frame_size'] = 200  # ms
        confs['frame_stride'] = 100  # ms
        signal = sp.Signal(np.arange(-20, 20), 20)
        frames = fr.striding(signal)
        self.assertEqual(19, frames.shape[0])

    def test_padding_0(self):
        confs['frame_size'] = 200  # ms
        confs['frame_stride'] = 100  # ms
        signal = sp.Signal(np.arange(-20, 20), 20)
        frames = fr.striding(signal)
        self.assertEqual(76, frames.size)

    def test_padding_2_zeros(self):
        confs['frame_size'] = 300  # ms
        confs['frame_stride'] = 150  # ms
        signal = sp.Signal(np.arange(-20, 20), 20)
        frames = fr.striding(signal)
        self.assertEqual(78, frames.size)
        self.assertEqual(0, frames[-1, -1])
        self.assertEqual(0, frames[-1, -2])

