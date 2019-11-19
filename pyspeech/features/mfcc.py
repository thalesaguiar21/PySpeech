import numpy as np
import scipy.fftpack as scifft

import pyspeech.dsp.processing as sp
import pyspeech.dsp.spectrum as spec
import pyspeech.dsp.metrics as smet
from pyspeech.configs import confs


def _extract(signal, nfilt, ncep, highfreq, lowfreq, emph):
    wnd_signal = make_frames_and_spectrum(signal, emph)
    power_spectrum = spec.power(wnd_signal)
    filter_banks = make_filter_banks(power_spectrum, highfreq, lowfreq,
                                     nfilt, signal.samplerate)

    fbanks_energies = power_spectrum @ filter_banks.T


def make_frames_and_spectrum(signal, emph):
    emph_signal = sp.emphasize(signal, emph)
    frames = sp.split_with_stride(signal)
    ham_frames = frames * np.hamming(sp.frame_len(confs['frame_size']))
    return ham_frames


def make_filter_banks(power_spec, highfreq, lowfreq, nfilt, srate):
    highfreq = highfreq if highfreq is not None else srate/2
    highmel = smet.hz_to_mel(highfreq)
    lowmel = smet.hz_to_mel(lowfreq)
    mel_points = np.linspace(lowmel, highmel, nfilt + 2)
    bins = np.floor((confs['nfft'] + 1)*smet.mel_to_hz(mel_points) / srate)

    fbanks = np.zeros((nfilt, confs['nfft']//2 + 1 ))
    for j in range(nfilt):
        for i in range(int(bins[j]), int(bins[j + 1])):
            fbanks[j, i] = (i-bins[j]) / (bins[j + 1]-bins[j])
        for i in range(int(bins[j + 1]), int(bins[j + 2])):
            fbanks[j, i] = (bins[j + 2]-i) / (bins[j + 2]-bins[j + 1])
    return fbanks


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

