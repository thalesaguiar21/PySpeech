import numpy as np
import pyspeech.dsp.processing as spproc
import pyspeech.dsp.filters as spfilt
import scipy.fftpack as scifft


def extract_mfcc(signals, frequencies, nfilt, processor):
    signals, frequencies = _fix_dimensions(signals, frequencies)
    mfccs = []
    for signal, frequency in zip(signals, frequencies):
        mfccs.append(_mfcc(signal, frequency, nfilt, processor))
    return _fix_mfcc_single_sample_output(mfccs)


def _fix_dimensions(signals, frequencies):
    augmented_signal = signals
    augmented_freqs = frequencies
    if not isinstance(signals[0], (list, np.ndarray)):
        augmented_signal = [signals]
    if not isinstance(frequencies, list):
        augmented_freqs = [frequencies]
    return augmented_signal, augmented_freqs


def _mfcc(signal, frequency, nfilt, processor):
    # Applies a Discrete Cosine Transforma (DCT) on Filter Banks
    power_spec = processor.preprocess(signal, frequency)
    filtered_frames = spfilt.mel_banks(
        power_spec, nfilt, frequency, processor.NFFT)
    dctframes = scifft.dct(filtered_frames, axis=1, norm='ortho')
    mfccs = np.array(dctframes)
    return mfccs


def _fix_mfcc_single_sample_output(mfccs):
    if len(mfccs) == 1:
        return mfccs[0]
    else:
        return mfccs


class Delta:

    def __init__(self, smooth):
       self.smooth = smooth 
       self.denom = sum([2 * n**2 for n in range(1, smooth + 1)])

    def make_means(self, frames):
        deltas = np.array(self.make(frames))
        return np.mean(deltas, axis=0)

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


def mean_normalise(feature):
    feature -= np.mean(feature, axis=0) + 1e-8

