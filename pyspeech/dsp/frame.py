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
    flength = size(signal.fs, size_)
    fstride = stride(signal.fs, stride_)
    nframes = 1 + math.ceil((signal.size-flength) / fstride)
    padlen = flength + (nframes*fstride) - signal.size
    paded_amps = np.append(signal.amps, np.zeros(padlen))
    indexes = np.tile(np.arange(flength), (nframes, 1))
    striding = np.arange(0, nframes*fstride, fstride).reshape(nframes, 1)
    mask = indexes + striding
    return paded_amps[mask]


def size(freq, flen=None):
    if flen is None:
        flen = framing['size']
    return _ms_to_samples(freq, flen)


def stride(freq, size=None):
    if size is None:
        size = framing['stride']
    return _ms_to_samples(freq, size)


def _ms_to_samples(freq, ms):
    nsamples = round(ms/1000 * freq)
    return int(nsamples)

