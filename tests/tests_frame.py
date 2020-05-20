import os

import unittest
from scipy.io import wavfile
import numpy as np

from .context import pyspeech
from tests import signal01, fs1
from pyspeech.dsp import frame as fr
from pyspeech.dsp.processing import Signal
from pyspeech import conf


class TestsFrame(unittest.TestCase):

    def test_split_19_frames_dim_4(self):
        frames = _make_frames(200, 100)
        self.assertEqual((19, 4), frames.shape)

    def test_no_padding(self):
        frames = _make_frames(200, 100)
        self.assertEqual(76, frames.size)

    def test_padding_2_zeros(self):
        frames = _make_frames(300, 150)
        self.assertEqual(78, frames.size)
        self.assertEqual(0, frames[-1, -1])
        self.assertEqual(0, frames[-1, -2])

    def test_overlap(self):
        frames = _make_frames(25, 10, signal01)
        L = int((conf.framing['size']/1000) * fs1)
        R = int((conf.framing['stride']/1000) * fs1)
        for n in range(1, frames.shape[0]):
            has_overlap = frames[n-1, R:] == frames[n, :L - R]
            self.assertTrue(all(has_overlap),
                            f'Wrong overlap between frames {n-1} and {n}')

    def test_empty_signal(self):
        frames = _make_frames(200, 100, Signal([], 10))
        self.assertEqual(frames.size, 0)

    def test_smaller_than_flen(self):
        frames = _make_frames(2001, 100)
        self.assertEqual(frames.shape, (2, 41))


class TestsRestore(unittest.TestCase):

    def test_size(self):
        frames = _make_frames(25, 10, signal01)
        restored = fr.restore(frames, fs1)
        padlen = _padlen(*frames.shape, 10, signal01)
        self.assertEqual(signal01.size + padlen, restored.size)

    def test_compare_to_original(self):
        original = Signal(np.arange(-20, 20), 20)
        frames = _make_frames(200, 150, original)
        restored = fr.restore(frames, 20)
        for i in range(original.size):
            msg = f'Difference at {i}: {original.amps[i]} != {restored[i]}'
            self.assertEqual(original.amps[i], restored[i], msg)

    def test_append_only_zeros(self):
        original = Signal(np.arange(-20, 20), 20)
        frames = _make_frames(200, 150, original)
        restored = fr.restore(frames, 20)
        padding = _padlen(*frames.shape, 150, original)
        for idx, pad in enumerate(restored[:-padding]):
            msg = f'Non-zero padding at {frames.shape[0] + idx}'
            self.assertEqual(0, pad, msg)

    def test_empty_frames(self):
        restored = fr.restore(np.array([]), 16000)
        self.assertEqual(restored.size, 0)

    def test_empty_frames_2d(self):
        restored = fr.restore(np.array([[]]), 8000)
        self.assertEqual(restored.size, 0)


def _make_frames(size, stride, signal=None):
    conf.framing['size'] = size
    conf.framing['stride'] = stride
    signal = Signal(np.arange(-20, 20), 20) if signal is None else signal
    return fr.apply(signal)


def _padlen(nframes, flen, stride, signal):
    stride_smp = fr.stride(signal.fs, stride)
    return flen + ((nframes - 1)*stride_smp - signal.size)

