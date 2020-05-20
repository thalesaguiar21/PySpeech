import os

import math
import numpy as np

from ..conf import framing


def apply(signal, size_=None, stride_=None):
    """ Splits signals into frames

    Args:
        size_ (int, optional): the frame length in ms, if None the 'frame_size'
            configuration will be used.
        stride_ (int, optional): the frame striding in ms, if None the
            'frame_stride' configuration will be used.

    Returns:
        The framed signal (ndarray)

    Example:
        >>> sig = Signal(np.arange(24000), 80)
        >>> frame = Frame(25, 10)
        >>> split_with_stride(signal, frame)
        [[1, 2],
         [2, 3],
         ...
         [22998, 22999]]
    """
    if signal.size == 0:
        return np.array([])
    flength = size(signal.fs, size_)
    fstride = stride(signal.fs, stride_)
    nframes = 1 + math.ceil(abs(signal.size-flength) / fstride)
    padlen = flength + ((nframes - 1)*fstride - signal.size)
    paded_amps = np.append(signal.amps, np.zeros(padlen))
    indexes = np.tile(np.arange(flength), (nframes, 1))
    striding = np.arange(0, nframes*fstride, fstride).reshape(nframes, 1)
    mask = indexes + striding
    return paded_amps[mask]


def restore(frames, fs):
    restored_signal = np.array([])
    if frames.size > 0:
        L, R = size(fs), stride(fs)
        tail = frames[1:, L - R:].reshape(-1)
        restored_signal = np.append(frames[0], tail)
    return restored_signal


def size(freq, flen=None):
    if flen is None:
        flen = framing['size']
    return _ms_to_samples(freq, flen)


def stride(freq, size=None):
    if size is None:
        size = framing['stride']
    return _ms_to_samples(freq, size)


def _ms_to_samples(freq, ms):
    return math.ceil(ms/1000 * freq)

