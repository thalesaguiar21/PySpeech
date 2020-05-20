import math

import numpy as np
from scipy.signal import firwin, butter, medfilt
import matplotlib.pyplot as plt

from . import processing as proc
from . import frame
from . import shorttime
from .. import conf


def remove(signal):
    """ Removes silence based on a simple dB (short-time energy) threshold

    Returns:
        A new signal composed of voiced segments from the input
    """
    voiced = np.array([])
    if signal.size > 0:
        filtered = _filter_signal(signal)
        or_frames = frame.apply(filtered)
        voiced_indexes, __ = _detect_silence(or_frames)
        voiced_frames = or_frames[voiced_indexes]
        restored_signal = frame.restore(voiced_frames, signal.fs)
        voiced = proc.Signal(restored_signal, signal.fs)
    return voiced


def _filter_signal(signal):
    highpass = _make_filter(signal.fs)
    famps = np.convolve(signal.amps, highpass)
    return proc.Signal(famps, signal.fs)


def _make_filter(freq):
    length = (conf.framing['size'] / 1000) * freq
    N = math.ceil(length / conf.fir['len'])
    if not N % 2:
        N += 1

    n = np.arange(N)
    sinc = np.sinc(2*conf.fir['fc'] * (n - (N - 1)/2))
    wndsinc = sinc * np.blackman(N)
    normsinc = wndsinc / np.sum(wndsinc)

    rev_sinc = -normsinc
    rev_sinc[int((N - 1)/2)] += 1
    return rev_sinc


def _detect_silence(frames):
    egys = _get_norm_egys(frames)
    threshold = _compute_threshold(egys)
    non_sil_indexes = []
    sil_indexes = []
    for i, egy in enumerate(egys):
        if egy > threshold:
            non_sil_indexes.append(i)
        else:
            sil_indexes.append(i)
    return non_sil_indexes, sil_indexes


def _get_norm_egys(frames):
    egys = shorttime.log_energy(frames)
    return egys - np.max(egys)


def _compute_threshold(egys):
    med_egys = medfilt(egys, 5)
    until = math.ceil(100 / conf.framing['size'])
    egyavg, egysig = np.mean(med_egys[:until]), np.std(med_egys[:until])
    return egyavg + 3*egysig

