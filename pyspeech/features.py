import numpy as np
import pyspeech.dsp.processing as spproc
import pyspeech.dsp.filters as spfilt
import scipy.fftpack as scifft


def mfcc(signals, frequencies, nfilt):
    fix_dimensions(signals, frequencies)
    mfccs = []
    for signal, frequency in zip(signals, frequencies):
        mfccs.append(_mfcc(signal, frequency, nfilt))
    return mfccs


def fix_dimensions(signals, frequencies):
    augmented_signal = []
    augmented_freqs = []
    if not isinstance(signals[0], list):
        augmented_signal.append(signals)
    if not isinstance(frequencies, list):
        augmented_freqs.append(frequencies)
    return augmented_freqs, augmented_signal


def _mfcc(signal, frequency, nfilt):
    # Applies a Discrete Correlation Transforma(DCT) on Filter Banks
    power_spec = spproc.preprocess(signal, frequency, 25, 10, 0.97)
    filtered_frames = spfilt.mel_banks(power_spec, nfilt, frequency, 512)
    dctframes = scifft.dct(filtered_frames, axis=1, norm='ortho')
    mfccs = np.array(dctframes)
    return mfccs


def deltas(feats):
    ''' Compute the first derivative of a given feature vector '''
    dim = len(feats.shape)
    if dim == 1:
        return sum(_deltas(feats))
    elif dim == 2:
        deltas = np.zeros((feats.shape[1], ))
        for fi in feats:
            deltas += _deltas(fi)
        return deltas
    else:
        raise ValueError('Dimension ({}) not supported!'.format(dim))


def _deltas(feats):
    ''' Compute the first derivative of a given array '''
    vec_size = feats.size
    forwarded_feats = feats[1:] - feats[:-1]
    denom = 2 * sum([i**2.0 for i in range(vec_size - 1)])
    coefs = np.array([i / denom for i in range(vec_size - 1)])
    return forwarded_feats * coefs


def log_energy(windowed_frames):
    epsilon = 10e-5
    frame_energies = []
    frame_size = windowed_frames.shape[1]
    for frame in windowed_frames:
        arg = 1.0/frame_size * frame**2.0
        frame_energies.append(10 * np.log10(epsilon + arg.sum()))
    return np.array(frame_energies)


def mean_normalization(feature):
    feature -= np.mean(feature, axis=0) + 1e-8
