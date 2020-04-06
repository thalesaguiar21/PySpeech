import math
import numpy as np


from ..dsp import processing as sproc


_MIN_DURATION = 100  # seconds


def by_duration(signals, fs, duration=500):
    """Splits all signals into specified duration

    Args:
        signals (list): the signal to be split
        fs (float): the sampling rate
        duration (int): the duration in ms, defaults to 500

    Returns:
        dataset (ndarray): splits of signals of the specified duration
    """
    max_duration = _find_max_duration(signals, fs)
    _warn_duration(duration, max_duration)
    duration = max(min(max_duration, duration), _MIN_DURATION)

    samplesize = math.ceil(duration/1000 * fs) # Duration in number of samples
    splits = _split_all(signals, samplesize)
    return splits


def _warn_duration(duration, max_duration):
    if max_duration < duration:
        raise Warning(f"Longest signal has {max_duration}ms, using this instead"
                      f" of {duration}ms")
    if duration < _MIN_DURATION:
        raise Warning(f"Min duration is {_MIN_DURATION}ms, using this instead"
                      f" of {duration}ms")



def _find_max_duration(signals, fs):
    max_duration = signals[0].size
    for signal in signals[1:]:
        if max_duration < signal.size:
            max_duration = signal.size
    return max_duration/fs * 1000


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

