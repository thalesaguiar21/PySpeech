import numpy as np

from . import frame


def log_energy(frames):
    energy = list(st_energy(frames))
    return 10 * np.log10(energy)


def st_energy(frames):
    fixedframes = _fix_frames(frames)
    wnd_frames = fixedframes * np.hamming(frames.shape[1])
    sqr_wnds = wnd_frames ** 2
    return np.sum(sqr_wnds, axis=1)


def zcr(frames, fs):
    fixedframes = _fix_frames(frames)
    nframes = fixedframes.shape[0]
    zeros = np.zeros((nframes, 1))
    sgns = np.apply_along_axis(sgn, 0, fixedframes)
    difs = sgns - np.hstack((sgns[:, 1:], zeros))
    absdifs = np.abs(difs)
    norm = frame.stride(fs) / 2*fixedframes[0].size
    zcrs = norm * np.sum(absdifs, axis=1)
    return zcrs


def _fix_frames(frames):
    if len(frames.shape) == 1:
        return np.array([frames])
    elif len(frames.shape) == 2:
        return frames
    else:
        raise ValueError("Can use only 1 and 2 dimensional arrays!")


def sgn(arr):
    sgns = []
    for x in arr:
        if x>= 0:
            sgns.append(1)
        else:
            sgns.append(-1)
    return sgns

