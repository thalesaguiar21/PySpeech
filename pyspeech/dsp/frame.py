import os

import math

from ..configs import confs


# Frames length and stride in miliseconds
_FLEN = confs['frame_size']
_FSTRIDE = confs['frame_stride']


def striding(signal):
    flen = _flenght(signal)
    fstride = _stride(signal)
    nframes = 1 + math.ceil((signal.size-flen) / fstride)
    pdlen = flen + (nframes*fstride) - signal.size
    paded_amps = np.append(signal.amps, np.zeros(padlen))
    indexes = np.tile(np.arange(flen), nframes)
    striding = np.arange(0, nframes*fstride, fstride)
    mask = indexes + striding
    return signal.amps[mask]


def _flength(signal):
    flen = _ms_to_frames(signal, _FLEN)


def _stride(signal):
    return _ms_to_frames(signal, _FSTRIDE)


def _ms_to_samples(signal, ms):
    nsamples = ms/1000 * signal.fs
    return math.floor(nsamples)

