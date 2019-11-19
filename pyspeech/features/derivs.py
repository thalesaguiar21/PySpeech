import numpy as np
import scipy.fftpack as scifft


def delta(feats, smooth=2):
    """ Extract deltas from the frames

    Args:
        frames (ndarray): The feature matrix
        smooth (int): The number frames to jump
    """
    denom = 2/6 * smooth * (smooth+1) * (2*smooth + 1)
    nframes, fdim = feats.shape
    pad_feats = np.pad(feats, ((smooth, smooth), (0, 0)), mode='edge')
    deltas = np.zeros(feats.shape)
    for t in range(nframes):
        coefs = np.arange(-smooth, smooth + 1)
        deltas[t] = coefs @ pad_feats[t:t + 2*smooth + 1]
    return deltas / denom


def log_energy(frames):
    energies = np.sum(frames, axis=1)
    energies_bounded = np.fmax(energies, np.finfo(np.float64).eps)
    return np.log(energies_bounded)

