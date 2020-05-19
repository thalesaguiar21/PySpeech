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
        frames = _make_frames(200, 100)
        self.assertEqual((19, 4), frames.shape)

    def test_padding_0(self):
        frames = _make_frames(200, 100)
        self.assertEqual(76, frames.size)

    def test_padding_2_zeros(self):
        frames = _make_frames(300, 150)
        self.assertEqual(78, frames.size)
        self.assertEqual(0, frames[-1, -1])
        self.assertEqual(0, frames[-1, -2])


    def test_overlap(self):
        frames = _make_frames(200, 100)
        for n in range(1, frames.shape[0]):
            has_overlap = frames[n-1, 2:] == frames[n, :2]
            self.assertTrue(all(has_overlap),
                            f'No overlap between frames {n-1} and {n}')


def _make_frames(size, stride):
    conf.framing['size'] = size
    conf.framing['stride'] = stride
    signal = Signal(np.arange(-20, 20), 20)
    return fr.apply(signal)

