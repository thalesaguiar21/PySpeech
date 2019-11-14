import numpy as np
import scipy.fftpack as scifft

import pyspeech.dsp.processing as spproc
import pyspeech.dsp.filters as spfilt


class Delta:

    def __init__(self, smooth):
       self.smooth = smooth
       self.denom = sum([2 * n**2 for n in range(1, smooth + 1)])

    def make_delta_and_ddelta_means(self, frames):
        delta1 = [self.make(frame) for frame in frames]
        d1means = np.array([self.make_means(frame) for frame in frames])
        d2means = np.array([self.make_means(dt1.T) for dt1 in delta1])
        return d1means, d2means

    def make_means(self, frames):
        deltas = np.array(self.make(frames))
        return np.mean(deltas, axis=1)

    def make(self, frames):
        frames_t = _fix_frame_dim(frames).T
        fr_deltas = [self._deltas(fr_i) for fr_i in frames_t]
        return np.array(fr_deltas)

    def _deltas(self, frame):
        max_length = frame.shape[0] - self.smooth
        deltas = [self._delta(frame, t) for t in range(max_length)]
        return deltas

    def _delta(self, frame, t):
        num = 0
        denom = 0
        for n in range(1, self.smooth + 1):
            num += n * (frame[t+n] - frame[t-n])
        delta = num / self.denom
        return delta


def _fix_frame_dim(frame):
    fixed_frames = np.array(frame)
    if not isinstance(frame[0], (list, np.ndarray)):
        fixed_frames = np.array([frame])
    return fixed_frames


def make_log_energy(windowed_frames):
    epsilon = 10e-5
    frame_energies = []
    frame_size = windowed_frames.shape[1]
    for frame in windowed_frames:
        arg = 1.0/frame_size * frame**2.0
        frame_energies.append(10 * np.log10(epsilon + arg.sum()))
    return np.array(frame_energies)

