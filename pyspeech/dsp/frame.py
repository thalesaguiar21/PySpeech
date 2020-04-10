import os

import math
import numpy as np

from ..configs import confs


def striding(signal):
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
    flen = flength(signal)
    fstride = stride(signal)
    nframes = 1 + math.ceil((signal.size-flen) / fstride)
    padlen = flen + (nframes*fstride) - signal.size
    paded_amps = np.append(signal.amps, np.zeros(padlen))
    indexes = np.tile(np.arange(flen), (nframes, 1))
    striding = np.arange(0, nframes*fstride, fstride).reshape(nframes, 1)
    mask = indexes + striding
    return paded_amps[mask]


def flength(signal):
    return _ms_to_samples(signal, confs['frame_size'])


def stride(signal):
    return _ms_to_samples(signal, confs['frame_stride'])


def _ms_to_samples(signal, ms):
    nsamples = round(ms/1000 * signal.fs)
    return int(nsamples)

