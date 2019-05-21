import numpy as np
import math
import pyspeech.scales as spscale


def mel_banks(pow_frames, qtd_filters, samplerate, nfft):
    low_freq_mel = 0
    high_freq_mel = spscale.hz_to_mel(samplerate)
    # Center points of triangular filters
    mel_points = np.linspace(low_freq_mel, high_freq_mel, qtd_filters + 2)
    hz_points = spscale.mel_signal_to_hz(mel_points)
    bins = np.floor((nfft + 1) * hz_points/samplerate)
    return _triangular_filter_banks(pow_frames, qtd_filters, bins, nfft)


def _triangular_filter_banks(pow_frames, qtd_filters, bins, nfft):
    fbank = np.zeros((qtd_filters, int(np.floor(nfft/2 + 1))))
    for m in range(1, qtd_filters + 1):
        f_m_minus = int(bins[m - 1])   # left
        f_m = int(bins[m])             # center
        f_m_plus = int(bins[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(
        filter_banks == 0, np.finfo(float).eps, filter_banks)
    return filter_banks


def hz_to_mel(rate):
    ''' Converts a frequency in Hz to Mel '''
    srate = rate / 2.0
    return 2595.0 * math.log10(1.0 + srate/700.0)


def mel_signal_to_hz(spectrum):
    ''' Converts a signal from Mel to Hz '''
    return 700 * (10**(spectrum/2595) - 1)


def hz_to_bark(rate):
    ''' Converts a hertz rate to bark scale '''
    return 13*math.atan(0.00076 * rate) + 3.5*math.atan((rate / 7500)**2)


def bark_signal_to_hz(spectrum):
    ''' Converts a bark scaled signal to hz '''
    normspectrum = spectrum / 600.0
    return 6 * np.log(normspectrum + np.sqrt((normspectrum + 1)**2))
