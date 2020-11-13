""" Provides function to represent a signal in the frequency domain """
import numpy as np

from .. import conf
from ..dsp import processing as sp


def norm_log_power(signal):
    """ Computes the log power spectrum with 0 peak """
    log_spec = log_power(signal)
    return log_spec - np.max(log_spec)


def log_power(signal):
    """ Computes the logarithm power spectrum of a signal """
    pow_spec = power(signal)
    bounded_psec = np.fmax(pow_spec, np.finfo(np.float64).eps)
    log_spec = np.log10(bounded_psec)
    return log_spec


def power(frames):
    """ Computes the power spectrum of a framed signal """
    mag_spec = magnitude(frames)
    return (1.0/conf.nfft) * (mag_spec**2)


def magnitude(wnd_frames):
    """ Computes the magnitude spectrum of framed signal """
    spectrum = np.fft.rfft(wnd_frames, conf.nfft)
    return np.absolute(spectrum)
