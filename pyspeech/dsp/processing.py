""" This module contains several signal processing functions, such as
emphasis, framing, and power spectrums. Besides a few data classes to
encapsulate arguments.
"""
import math
import numpy as np


from pyspeech.configs import confs


class Signal:
    """ A simple class to represent a digital signal

    Atributes:
        amps (ndarray): The amplitudes
        size (ndarray): The signal length
        fs (int): The sampling rate on Hz
    """
    def __init__(self, amps, freq):
        self.amps = amps
        self.size = len(amps)
        self.fs = freq


def frame_len(freq):
    return int(round(confs['frame_size']/1000. * freq))


def frame_step(freq):
    return int(round(confs['frame_stride']/1000. *freq))


def split_with_stride(signal):
    """ Splits signals into frames

    Args:
        signals (list:Signal): The signals to be split
        frame (Frame): The frame size and stride

    Returns:
        frames (ndarray): The frammed signal

    Example:
        >>> sig = Signal(np.arange(24000), 80)
        >>> frame = Frame(25, 10)
        >>> split_with_stride(signal, frame)
        [[1, 2],
         [2, 3],
         ...
         [22998, 22999]]
    """
    flen = frame_len(signal.fs)
    fstride= frame_step(signal.fs)
    nframes = 1 + math.ceil((signal.size - flen) / fstride)
    padding = (nframes-1)*fstride + flen - signal.size
    padded_amps = np.append(signal.amps, np.zeros((padding)))
    base_idx = np.tile(np.arange(0, flen), (nframes, 1))
    idx_step = np.tile(np.arange(0, nframes * fstride, fstride), (flen, 1)).T
    indices = (base_idx + idx_step).astype(dtype=np.int32)
    frames = padded_amps[indices]
    return frames


def split(signal):
    flen = frame_len(signal.fs)
    nframes = math.ceil(signal.size / flen)
    padlen = nframes*flen - signal.size

    pad_amps = np.append(signal.amps, np.zeros(padlen))
    frames = np.reshape(pad_amps, (nframes, flen))
    return frames


def emphasize(signal, gain):
    emph_amps = np.append(signal.amps[0], signal.amps[1:] - gain*signal.amps[:-1])
    return Signal(emph_amps, signal.fs)


def normalise(signal):
    max_amp = np.absolute(signal.amps).max()
    if max_amp == 0.0:
        normalised_amps = signal.amps
    else:
        normalised_amps = signal.amps / max_amp
    return Signal(normalised_amps, signal.fs)


def find_best_nfft(freq, flen=None):
    flen = frame_len(confs['frame_size'], freq) if flen is None else flen
    if freq > 0 and flen > 0:
        nfft = 1
        while nfft < freq * flen:
            nfft *= 2
        return nfft
    else:
        raise ValueError('Frame length and frequency must be greater than zero')

