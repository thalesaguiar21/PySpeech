import numpy as np

from . import frame


def log_energy(frames):
    energies = energy(frames)
    return 10 * np.log10(1e-5 + energies)


def energy(frames):
    fixedframes = _fix_frames(frames)
    __, flen = fixedframes.shape
    wnd_frames = fixedframes * np.hamming(flen)
    sqr_wnds = wnd_frames ** 2
    return np.sum(sqr_wnds, axis=1)


def zcr(frames):
    """ Computes the short-time zcr of the given frames

    Args:
        frames: 2d numpy array

    Returns:
        short-time zero-crossin rate of each frame
    """
    fixedframes = _fix_frames(frames)
    nframes, flen = fixedframes.shape
    sgns = _sign(fixedframes)
    abs_diff = np.abs(sgns[:, 1:] - sgns[:, :-1])
    zcrs = (1 / (2*flen)) * np.sum(abs_diff, axis=1)
    return zcrs


def _sign(X):
    signs = np.zeros(X.shape)
    signs[np.where(X >= 0)] = 1
    signs[np.where(X < 0)] = -1
    return signs


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



