import matplotlib.pyplot as plt
import os
import unittest
import math

import numpy as np
from scipy.io import wavfile

from tests import signal01
from .context import pyspeech
from pyspeech.conf import framing
from pyspeech.dsp import shorttime 
from pyspeech.dsp.processing import Signal
from pyspeech.dsp import frame


class TestsShorttime(unittest.TestCase):

    def setUp(self):
        framing['size'] = 25
        framing['stride'] = 10

    def test_energy(self):
        frames = get_frames(signal01)
        energies = shorttime.energy(frames)
        flength = frame.size(signal01.fs)
        stride = frame.stride(signal01.fs)
        nframes = 1 + math.ceil((signal01.size-flength) / stride)
        self.assertEqual(len(energies), nframes)

    def test_negative_energy(self):
        signal = get_frames()
        energies = list(shorttime.energy(signal))
        allpositive = all(egy > 0 for egy in energies)
        self.assertTrue(allpositive)

    def test_logenergy(self):
        signal = get_frames()
        lenergies = shorttime.log_energy(signal)

    def test_logenergy_zero(self):
        _configure_frame()
        amps = [0] * 10
        frames = frame.apply(Signal(amps, 5))
        lenergies = shorttime.log_energy(frames)
        all50 = all(legy == -50 for legy in lenergies)
        self.assertTrue(all50)

    def test_zrate_allpositive(self):
        frames = get_frames(signal01)
        rates = shorttime.zcr(frames)
        allpositive = all(rate >= 0 for rate in rates)
        self.assertTrue(allpositive)

    def test_zcr_ncross(self):
        _configure_frame()
        amps = np.array([3, 3, 3, -3, -4, -5, 10, 15, 12, -1])
        fs = 5
        frames = frame.apply(Signal(amps, fs))
        reals = [0, 1/3, 1/3, 0, 1/3, 1/3, 0, 1/3]
        zerocross = shorttime.zcr(frames)
        equal = all(rate == real for rate, real in zip(zerocross, reals))
        self.assertTrue(equal)

    def test_zcr_nocrossing_neg(self):
        allzero = self.zcr_nocrossing(-1)
        self.assertTrue(allzero)

    def test_zcr_nocrossing_pos(self):
        notnull = self.zcr_nocrossing(2)
        positive = self.zcr_nocrossing(0)
        self.assertEqual(notnull, positive)

    def zcr_nocrossing(self, amp):
        _configure_frame()
        signal = Signal([amp]*10, 5)
        frames = frame.apply(signal)
        zerocross = shorttime.zcr(frames)
        allzero = all(zcr == 0 for zcr in zerocross)
        return allzero


class TestsAutocorr(unittest.TestCase):

    def setUp(self):
        framing['size'] = 25
        framing['stride'] = 10

    def test_simple_signal_norm(self):
        frames = _make_simple_frames()
        reals = [0.158, 0, 0.026, 0.158, -0.094, 0.079, 0.116, 0.092]
        corrs = shorttime.autocorr_norm(frames)
        equals = [abs(cor-real) < 1e-3 for cor, real in zip(corrs, reals)]
        self.assertTrue(equals)

    def test_simple_signal(self):
        frames = _make_simple_frames()
        reals = [1.44, 0., 0.24, 2.56, -2.4, 8, 26.4, 13.44]
        corrs, __ = shorttime.autocorr(frames)
        equals = [abs(cor-real) < 1e-3 for cor, real in zip(corrs, reals)]
        self.assertTrue(equals)

    def test_zero_amps(self):
        amps = [0] * 10
        _configure_frame()
        frames = frame.apply(Signal(amps, 5))
        corrs, __ = shorttime.autocorr(frames)
        self.assertTrue(all(cor == 0 for cor in corrs))

    def test_zero_amps_norm(self):
        amps = [0] * 10
        _configure_frame()
        frames = frame.apply(Signal(amps, 5))
        corrs = shorttime.autocorr_norm(frames)
        self.assertTrue(all(cor == 0 for cor in corrs))


    def test_lag_negative(self):
        frames = _make_simple_frames()
        data = [-1, 20, 3, 0]
        for lag in data:
            try:
                corrs = shorttime.autocorr_norm(frames, lag)
                self.fail(f"{lag} did not raised ValueError")
            except ValueError:
                pass


def _make_simple_frames():
    _configure_frame()
    signal = _make_simple_signal()
    return frame.apply(signal)


def _make_simple_signal():
    amps = np.array([3, 3, 3, -3, -4, -5, 10, 15, 12, -1])
    fs = 5
    return Signal(amps, fs)


def _configure_frame(size=500, stride=100):
    framing['size'] = 500
    framing['stride'] = 100


def get_frames(signal=None):
    signal = signal01 if signal is None else signal
    return frame.apply(signal)

