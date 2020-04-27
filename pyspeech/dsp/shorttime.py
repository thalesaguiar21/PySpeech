import numpy as np

from . import frame


def log_energy(frames):
    energy = st_energy(frames)
    return 10 * np.log10(1e-5 + energy)


def st_energy(frames):
    fixedframes = _fix_frames(frames)
    __, flen = fixedframes.shape
    wnd_frames = fixedframes * np.hamming(flen)
    sqr_wnds = wnd_frames ** 2
    return np.sum(sqr_wnds, axis=1) / flen


def zcr(frames, fs):
    fixedframes = _fix_frames(frames)
    nframes, flen = fixedframes.shape
    sgns = np.apply_along_axis(sgn, 0, fixedframes)
    lastcol = np.reshape(sgns[:, -1], (nframes, 1))
    difs = np.hstack((sgns[:, 1:], lastcol)) - sgns
    absdifs = np.abs(difs)
    norm = frame.stride(fs) / (2*flen)
    zcrs = norm * np.sum(absdifs, axis=1)
    return zcrs


def sgn(arr):
    sgns = []
    for x in arr:
        if x>= 0:
            sgns.append(1)
        else:
            sgns.append(-1)
    return sgns


def autocorr_norm(frames, lag=1):
    corr, wnd_frames = autocorr(frames, lag)
    sqr_frames = wnd_frames ** 2
    sum1 = np.sum(sqr_frames[:, :-lag], axis=1)
    sum2 = np.sum(sqr_frames[:, lag:], axis=1)
    denom = np.sqrt(sum1 * sum2) + 1e-15
    return corr / denom


def autocorr(frames, lag=1):
    nframes, flen = frames.shape
    if lag > flen - 1 or lag < 1:
        raise ValueError('Lag has to be at most the frame samples - 1')
    wnd_frames = frames * np.hamming(flen)
    corr = np.sum(wnd_frames[:, :-lag] * wnd_frames[:, lag:], axis=1)
    return corr, wnd_frames


def _fix_frames(frames):
    if len(frames.shape) == 1:
        return np.array([frames])
    elif len(frames.shape) == 2:
        return frames
    else:
        raise ValueError("Can use only 1 and 2 dimensional arrays!")



