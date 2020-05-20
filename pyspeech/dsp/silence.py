import math

import numpy as np
from scipy.signal import filtfilt, butter, medfilt
import matplotlib.pyplot as plt

from . import processing as proc
from . import frame
from . import shorttime
from .. import conf

# Upper silence threshold
HSdB = -40


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
    b, a = _make_filter(signal.fs)
    famps = filtfilt(b, a, signal.amps)
    return proc.Signal(famps, signal.fs)


def _make_filter(freq):
    nyq = 0.5 * freq
    # normalise the cutoff frequence with respect to nyquist freq
    normal_cutoff = conf.fir['fc'] / nyq
    b, a = butter(conf.fir['order'], normal_cutoff, btype='high', analog=False)
    return b, a


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
    return min(HSdB, egyavg + 3*egysig)

