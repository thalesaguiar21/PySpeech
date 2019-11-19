import numpy as np
import scipy.fftpack as scifft

import pyspeech.dsp.processing as sp
import pyspeech.dsp.spectrum as spec
import pyspeech.dsp.filters as spfilt
from pyspeech.configs import confs


def _extract(signal, nfilt, ncep, highfreq, lowfreq, emph):
    wnd_signal = make_frames_and_spectrum(signal, emph)
    power_spectrum = spec.power(wnd_signal)
    filter_banks = apply_filter_banks(power_spectrum, highfreq, lowfreq)



def make_frames_and_spectrum(signal, emph):
    emph_signal = sp.emphasize(signal, emph)
    frames = sp.split_with_stride(signal)
    ham_frames = frames * np.hamming(sp.frame_len(confs['frame_size']))
    return ham_frames


def apply_filter_banks(power_spec, highfreq, lowfreq):
    highfreq = highfreq if highfreq is not None else signal.samplerate/2


def make_means_and_deltas(signals, frequencies, nfilt, processor, cepstrums):
    ''' Extract MFCC cepstrums, delta, and double delta '''
    mfccs = extract(signals, frequencies, nfilt, processor, cepstrums)
    mfcc_means = np.array([np.mean(mfcc, axis=0) for mfcc in mfccs])
    delta = spdyn.Delta(smooth=2)
    d1, d2 = delta.make_delta_and_ddelta_means(mfccs)
    allfeats = np.hstack((mfcc_means, d1, d2))
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

