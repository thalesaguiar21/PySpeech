import numpy as np
import pyspeech.dsp.processing as spproc
import pyspeech.dsp.filters as spfilt
import scipy.fftpack as scifft
import math


def extract(signal, frequency, nfilt, processor, cepstrums=13):
    power_spec = processor.preprocess(signal, frequency)
    wraped_power_spec = wrap_on_bark(power_spec)


def wrap_on_bark(frames):
    norm_frames = frames / 1200*math.pi
    return 6 * np.log(norm_frames + np.sqrt(norm_frames**2 + 1))


def bark_filter(frames, nfilters, bins, nfft):
    fbank = np.zeros((nfilters, int(np.floor(nfft/2 + 1))))
    for m in range(1, nfilters + 1):
        psi_left = int(bins[m - 1])
        psi_mid = int(bins[m])
        psi_right = int(bins[m + 1])

        for omega in range(psi_left, psi_mid):
            fbank[m - 1, omega] = 10 ** (2.5*(omega + 0.5))
        for omega in range(psi_mid, psi_right):
            fbank[m - 1, omega] = 10 ** (-10*(omega + 0.5))
    filter_banks = np.dot(frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    return 20 * np.log10(filter_banks)




