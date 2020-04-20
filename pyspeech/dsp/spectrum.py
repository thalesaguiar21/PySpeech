import numpy as np

from pyspeech import conf
import pyspeech.dsp.processing as sp


def norm_log_power(signal):
    log_spec = log_power(signal)
    return log_spec - np.max(log_spec)


def log_power(signal, frame, nfft):
    pow_spec = power(signal)
    bounded_psec = np.fmax(pow_spec, np.finfo(np.float64).eps)
    log_spec = np.log10(bounded_psec)


def power(signal):
    mag_spec = magnitude(signal)
    return 1.0/conf.nfft * mag_spec**2


def magnitude(wnd_frames):
    spectrum = np.fft.rfft(wnd_frames, conf.nfft)
    return np.absolute(spectrum)

