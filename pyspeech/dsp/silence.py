import math

import numpy as np
from scipy.signal import filtfilt, butter, medfilt
import matplotlib.pyplot as plt

from . import processing as proc
from . import frame
from . import shorttime
from .. import conf

# Upper silence threshold
_ITU = -35 # dB
_IF = 35


def detect(signal):
    filtered_signal = _filter_signal(signal)
    frames = frame.apply(filtered_signal)
    voiced_indexes = _detect_silence(frames, signal.fs)
    return voiced_indexes


def remove(signal):
    """ Removes silence based on a simple dB (short-time energy) threshold

    Returns:
        A new signal composed of voiced segments from the input
    """
    voiced = np.array([])
    if signal.size > 0:
        filtered = _filter_signal(signal)
        or_frames = frame.apply(signal)
        filtered_frames = frame.apply(filtered)
        voiced_indexes = _detect_silence(filtered_frames, signal.fs)
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


def _detect_silence(frames, fs):
    zero_peak_egys = _get_norm_egys(frames)
    zcrs = frame.stride(fs) * shorttime.zcr(frames, fs)
    egy_threshold = _compute_threshold(_ITU, zero_peak_egys)
    zcr_threshold = _compute_threshold(_IF, zcrs)
    voiced_frames = []
    for i in range(frames.shape[0]):
       is_speech = zero_peak_egys[i] > egy_threshold
       is_voiced = zcrs[i] < zcr_threshold
       if is_speech and is_voiced:
           voiced_frames.append(i)
    return voiced_frames


def _get_norm_egys(frames):
    egys = shorttime.log_energy(frames)
    return egys - np.max(egys)


def _compute_threshold(fixed, signal_rep):
    until = math.ceil(100 / conf.framing['stride'])
    med_values = medfilt(signal_rep, 5)
    nonspeech = med_values[:until]
    avg, sig = np.mean(nonspeech), np.std(nonspeech)
    return max(fixed, avg + 3*sig)

