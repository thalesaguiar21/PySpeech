import numpy as np
import pyspeech.dsp.processing as spproc
import pyspeech.dsp.filters as spfilt
import scipy.fftpack as scifft
import pyspeech.features.dynamics as spdyn


def make_means_and_deltas(signals, frequencies, nfilt, processor, cepstrums):
    ''' Extract MFCC cepstrums, delta, and double delta '''
    mfccs = extract(signals, frequencies, nfilt, processor, cepstrums)
    mfcc_means = np.array([np.mean(mfcc, axis=0) for mfcc in mfccs])
    delta = spdyn.Delta(smooth=2)
    delta1 = [delta.make(mfcc) for mfcc in mfccs]
    delta1means = np.array([delta.make_means(mfcc) for mfcc in mfccs])
    delta2means = np.array([delta.make_means(dt1.T) for dt1 in delta1])
    allfeats = np.hstack((mfcc_means, delta1means, delta2means))
    return allfeats


def extract(signals, frequencies, nfilt, processor, cepstrums=13):
    signals, frequencies = _fix_dimensions(signals, frequencies)
    mfccs = []
    for signal, frequency in zip(signals, frequencies):
        mfccs.append(_mfcc(signal, frequency, nfilt, processor, cepstrums))
    return mfccs


def _fix_dimensions(signals, frequencies):
    augmented_signal = signals
    augmented_freqs = frequencies
    if not isinstance(signals[0], (list, np.ndarray)):
        augmented_signal = [signals]
    if not isinstance(frequencies, list):
        augmented_freqs = [frequencies]
    return augmented_signal, augmented_freqs


def _mfcc(signal, frequency, nfilt, processor, cepstrums):
    # Applies a Discrete Cosine Tranform (DCT) on Filter Banks
    power_spec = processor.preprocess(signal, frequency)
    filtered_frames = spfilt.mel_banks(power_spec, nfilt,
                                       frequency, processor.NFFT)
    dctframes = scifft.dct(filtered_frames, type=2, axis=1, norm='ortho')
    mfccs = np.array(dctframes)[:, 1:cepstrums + 1]
    return mfccs

