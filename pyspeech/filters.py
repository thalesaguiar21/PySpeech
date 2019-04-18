import numpy as np
import math


def mel_filer_banks(pow_frames, qtd_filters, samplerate, nfft):
    low_freq_mel = 0
    high_freq_mel = hertz_to_mel(samplerate)
    # Center points of triangular filters
    mel_points = np.linspace(low_freq_mel, high_freq_mel, qtd_filters + 2)
    hz_points = mel_points_to_hertz(mel_points)
    bin = np.floor((nfft + 1) * hz_points / samplerate)

    fbank = np.zeros((qtd_filters, int(np.floor(nfft / 2 + 1))))
    for m in range(1, qtd_filters + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(
        filter_banks == 0, np.finfo(float).eps, filter_banks)
    return filter_banks


def bark_filter_banks():
    pass


def hertz_to_mel(samplerate):
    # Converts a signal from Hz to Mel
    srate = samplerate / 2.0
    return 2595.0 * math.log10(1.0 + srate / 700.0)


def mel_points_to_hertz(points):
    # Converts a signal from Mel to Hz
    return 700 * (10 ** (points / 2595) - 1)
