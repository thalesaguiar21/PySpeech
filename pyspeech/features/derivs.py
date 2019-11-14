import numpy as np
import scipy.fftpack as scifft

import pyspeech.dsp.processing as spproc
import pyspeech.dsp.filters as spfilt

def delta(frames, smooth=2):
    """ Extract deltas from the frames

    Args:
        frames (ndarray): The feature matrix
        smooth (int): The number frames to jump
    """
    frames_t = frames.T
    denom = 2/6 * (smooth+1) * (2*smooth + 1)
    deltas = np.zeros((frames_t.shape[0], frames_t.shape[1] - smooth))
    for frm, delta in zip(frames_t, deltas):
        for t in range(frames_t.shape[0] - smooth):
            for n in range(1, smooth + 1):
               delta[t] += n*(frm[t+n] - frm[t-n]) / denom
    return deltas


def log_energy(signal, emph, frame, lfreq, hfreq):
    wnd_signal = sproc.split(signal, frame)
    pow_spec = sproc.power_spectrum(wnd_signal, nfft)
    energies = np.sum(pow_spec, axis=1)
    bounded_energies = np.fmax(energy, np.finfo(np.float64).eps)
    return bounded_energies

