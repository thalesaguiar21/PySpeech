import numpy as np
import pyspeech.scales as spscale


def mel_banks(pow_frames, qtd_filters, samplerate, nfft):
    low_freq_mel = 0
    high_freq_mel = spscale.hz_to_mel(samplerate)
    # Center points of triangular filters
    mel_points = np.linspace(low_freq_mel, high_freq_mel, qtd_filters + 2)
    hz_points = spscale.mel_signal_to_hz(mel_points)
    bins = np.floor((nfft + 1) * hz_points/samplerate)
    return _triangular_filter_banks(pow_frames, qtd_filters, bins, nfft)


def bark_banks(pow_frames, qtd_filters, samplerate, nfft):
    low_freq_bark = 0
    high_freq_bark = spscale.hz_to_bark(samplerate)
    # Center points of triangular filters
    bark_points = np.linspace(low_freq_bark, high_freq_bark, qtd_filters + 2)
    hz_points = spscale.bark_signal_to_hz(bark_points)
    bins = np.floor((nfft + 1) * hz_points / samplerate)
    return _bark_filter_banks(pow_frames, qtd_filters, bins, nfft)


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


def _bark_filter_banks(pow_frames, qtd_filters, bins, nfft):
    raise NotImplementedError()
