import math
import numpy as np


def hz_to_mel(rate):
    ''' Converts a frequency in Hz to Mel '''
    srate = rate / 2.0
    return 2595.0 * math.log10(1.0 + srate / 700.0)


def mel_signal_to_hz(spectrum):
    ''' Converts a signal from Mel to Hz '''
    return 700 * (10 ** (spectrum / 2595) - 1)


def hz_to_bark(rate):
    ''' Converts a hertz rate to bark scale '''
    t1 = 13.0 * math.atan(0.00076 * rate)
    t2 = 3.5 * math.atan((rate / 7500) ** 2)
    return t1 + t2


def bark_signal_to_hz(spectrum):
    ''' Converts a bark scaled signal to hz '''
    normspectrum = spectrum / 600.0
    return 6 * np.log(normspectrum + np.sqrt((normspectrum + 1) ** 2))
