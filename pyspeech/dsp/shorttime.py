""" Provides functions to compute short-time representations of signals """
import numpy as np

from . import frame


def log_energy(frames):
    """ Computes the normalised short-time energy of the given frames """
    energies = energy(frames)
    bounded_egys = np.fmax(energies, np.finfo(np.float64).eps)
    return 10 * np.log10(bounded_egys)


def energy(frames):
    """ Computes the short-time energy """
    fixedframes = _fix_frames(frames)
    __, flen = fixedframes.shape
    wnd_frames = fixedframes * np.hamming(flen)
    sqr_wnds = wnd_frames ** 2
    return np.sum(sqr_wnds, axis=1)


def zcr(frames, freq=16000):
    """ Computes the short-time zcr of the given frames

    Args:
        frames: 2d numpy array
        fs (int): signal sampling rate, defaults to 16000

    Returns:
        short-time zero-crossin rate of each frame
    """
    fixedframes = _fix_frames(frames)
    fstride = frame.stride(freq)
    sgns = _sign(fixedframes)
    sample_before = np.roll(sgns[:, fstride - 1, None], 1)
    sample_before[0] = sgns[0, 0]
    shifted_sgns = np.hstack((sample_before, sgns[:, :-1]))
    abs_diff = np.abs(sgns - shifted_sgns)
    zcrs = (1 / (2*frames.shape[1])) * np.sum(abs_diff, axis=1)
    return zcrs


def _sign(mat):
    signs = np.zeros(mat.shape)
    signs[np.where(mat >= 0)] = 1
    signs[np.where(mat < 0)] = -1
    return signs


def autocorr_norm(frames, lag=1):
    """ Computes the normalised autocorrelation """
    corr, wnd_frames = autocorr(frames, lag)
    sqr_frames = wnd_frames ** 2
    sum1 = np.sum(sqr_frames[:, :-lag], axis=1)
    sum2 = np.sum(sqr_frames[:, lag:], axis=1)
    denom = np.sqrt(sum1 * sum2) + 1e-15
    return corr / denom


def autocorr(frames, lag=1):
    """ Computes the autocorrelation coefficients between frames """
    __, flen = frames.shape
    if lag > flen - 1 or lag < 1:
        raise ValueError('Lag has to be at most the frame samples - 1')
    wnd_frames = frames * np.hamming(flen)
    corr = np.sum(wnd_frames[:, :-lag] * wnd_frames[:, lag:], axis=1)
    return corr, wnd_frames


def _fix_frames(frames):
    if frames.ndim > 2:
        raise ValueError("Can use only 1 and 2 dimensional arrays!")
    if len(frames.shape) == 1:
        return np.array([frames])
    return frames
