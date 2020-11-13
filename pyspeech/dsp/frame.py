""" Functions to split and restore signals into frames fo given milliseconds """
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


def restore(frames, freq):
    """ Concatenates the frames into a single signal

    Args:
        frames: the frammed signal
        freq: the signal sampling rate
    """
    restored_signal = np.array([])
    if frames.size > 0:
        left, right = size(freq), stride(freq)
        tail = frames[1:, left - right:].reshape(-1)
        restored_signal = np.append(frames[0], tail)
    return restored_signal


def size(freq, flen=None):
    """ The size in samples of the frame

    Args:
        freq: the sampling frequency
        flen: the length of the frame, optional, defaults to None
    """
    if flen is None:
        flen = framing['size']
    return _ms_to_samples(freq, flen)


def stride(freq, slen=None):
    """ The stride in samples of the frame """
    if slen is None:
        slen = framing['stride']
    return _ms_to_samples(freq, slen)


def overlap():
    """ The overlap in samples of the current frames """
    return framing['size'] - framing['stride']


def _ms_to_samples(freq, ms):
    return math.ceil(ms/1000 * freq)


def get_duration(frames, freq):
    nframes, fr_len = frames.shape
    signal_size = (nframes - 1)*stride(freq) + size(freq)
    duration = signal_size / freq
    return duration

