import os

import math
import numpy as np

from ..configs import confs


def striding(signal, flen=None, fstride=None):
    """ Splits signals into frames

    Args:
        flen (int, optional): the frame length in ms, if None the 'frame_size'
            configuration will be used.
        fstride (int, optional): the frame striding in ms, if None the
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
    flen = flength(signal.fs, flen)
    fstride = stride(signal.fs, fstride)
    nframes = 1 + math.ceil((signal.size-flen) / fstride)
    padlen = flen + (nframes*fstride) - signal.size
    paded_amps = np.append(signal.amps, np.zeros(padlen))
    indexes = np.tile(np.arange(flen), (nframes, 1))
    striding = np.arange(0, nframes*fstride, fstride).reshape(nframes, 1)
    mask = indexes + striding
    return paded_amps[mask]


def flength(freq, flen=None):
    if flen is None:
        flen = confs['frame_size']
    return ms_to_samples(freq, flen)


def stride(freq, fstride=None):
    if fstride is None:
        fstride = confs['frame_stride']
    return ms_to_samples(freq, fstride)


def ms_to_samples(freq, ms):
    nsamples = round(ms/1000 * freq)
    return int(nsamples)

