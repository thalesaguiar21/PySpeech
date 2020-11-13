""" This module contains several signal processing functions, such as
emphasis, framing, and power spectrums. Besides a few data classes to
encapsulate arguments.
"""
import numpy as np


from . import frame
from ..conf import framing


class Signal:
    """ A simple class to represent a digital signal

    Atributes:
        amps (ndarray): The amplitudes
        size (ndarray): The signal length
        fs (int): The sampling rate on Hz
    """
    def __init__(self, amps, freq):
        self.amps = amps
        self.size = len(amps)
        self.fs = freq


def emphasize(signal, gain):
    """ Apply an emphasis into the signal corresponding to the gain """
    emph_amps = np.append(signal.amps[0], signal.amps[1:] - gain*signal.amps[:-1])
    return Signal(emph_amps, signal.fs)


def normalise(signal):
    """ Normalise the amplitudes to [-1, 1] """
    max_amp = np.absolute(signal.amps).max()
    if max_amp == 0.0:
        normalised_amps = signal.amps
    else:
        normalised_amps = signal.amps / max_amp
    return Signal(normalised_amps, signal.fs)


def find_best_nfft(freq, flen=None):
    """ Computes the minimun number of points to achive the Nyquist frequency

    Args:
        freq: the signal sampling frequency
        flen: the frame length, defaults to None
            if None, uses the 'size' in configuration file
    """
    flen = frame.size(framing['size'], freq) if flen is None else flen
    if freq > 0 and flen > 0:
        nfft = 1
        while nfft < freq * flen:
            nfft *= 2
        return nfft
    raise ValueError('Frame length and frequency must be greater than zero')
