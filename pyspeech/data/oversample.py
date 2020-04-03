import math
import numpy as np


from ..dsp import processing as sproc


def by_duration(signals, fs, duration=500):
    """Splits all signals into specified duration

    Args:
        signals (list): the signal to be split
        fs (float): the sampling rate
        duration (int): the duration in ms, defaults to 500

    Returns:
        dataset (ndarray): splits of signals of the specified duration
    """
    samplesize = math.ceil(duration/1000 * fs) # Duration in number of samples
    splits = _split_all(signals, samplesize)
    return splits


def by_shortest(signals, fs):
    """Splits all signals by the duration of the shortest signal.
    It calls 'by_scalar_shortest'"""
    splits = by_scalar_shortest(signals, fs, 1)
    return splits


def by_scalar_shortest(signals, fs, alpha=3):
    """Splits all signals by a divisor of the shortest signal

    Args:
        signals (list): the signals to be split
        fs (float): the sampling rate
        alpha (int): the shortest signal divisor

    Returns:
        dataset (ndarray): splits of signals at the specified length
    """
    shortest = _find_shortest(signals)
    samplesize = math.ceil(shortest.size / alpha)
    splits = _split_all(signals, samplesize)
    return splits


def _find_shortest(signals):
    shortest = signals[0]
    for signal in signals:
        if signal.size < shortest.size:
            shortest = signal
    return shortest


def _split_all(signals, nsamples):
    splits = []
    for signal in signals:
        sigsplits = _split(signal, nsamples)
        splits.extend(sigsplits)
    return np.array(splits)


def _split(signal, nsamples):
    nsplits = math.ceil(signal.size / nsamples)
    padlen = nsplits*nsamples - signal.size
    paddedsignal = np.append(signal, np.zeros(padlen))
    return paddedsignal.reshape((nsplits, nsamples))

