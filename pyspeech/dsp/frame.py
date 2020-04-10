import os

import math
import numpy as np

from ..configs import confs


def striding(signal):
    flen = _flength(signal)
    fstride = _stride(signal)
    nframes = 1 + math.ceil((signal.size-flen) / fstride)
    padlen = flen + (nframes*fstride) - signal.size
    paded_amps = np.append(signal.amps, np.zeros(padlen))
    indexes = np.tile(np.arange(flen), (nframes, 1))
    striding = np.arange(0, nframes*fstride, fstride).reshape(nframes, 1)
    mask = indexes + striding
    return paded_amps[mask]


def _flength(signal):
    return _ms_to_samples(signal, confs['frame_size'])


def _stride(signal):
    return _ms_to_samples(signal, confs['frame_stride'])


def _ms_to_samples(signal, ms):
    nsamples = round(ms/1000 * signal.fs)
    return int(nsamples)

