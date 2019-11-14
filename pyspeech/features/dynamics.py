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

def make_log_energy(windowed_frames):
    epsilon = 10e-5
    frame_energies = []
    frame_size = windowed_frames.shape[1]
    for frame in windowed_frames:
        arg = 1.0/frame_size * frame**2.0
        frame_energies.append(10 * np.log10(epsilon + arg.sum()))
    return np.array(frame_energies)

