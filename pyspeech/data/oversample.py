import math
import numpy as np


from ..dsp import processing as sproc


_MIN_DURATION = 100  # seconds


def by_duration(signals, fs, ids, duration=500):
    """Splits all signals into specified duration

    Args:
        signals (list): the signal to be split
        fs (float): the sampling rate
        ids (ndarray): the ids from each signal
        duration (int): the duration in ms, defaults to 500

    Returns:
        dataset (ndarray): splits of signals of the specified duration
    """
    samplesize = math.ceil(duration/1000 * fs) # Duration in number of samples
    splits = _split_all(signals, fs, ids, samplesize)
    return splits


def by_shortest(signals, fs, ids):
    """Splits all signals by the duration of the shortest signal.
    It calls 'by_scalar_shortest'"""
    splits = by_scalar_shortest(signals, fs, ids, 1)
    return splits


def by_scalar_shortest(signals, fs, ids, alpha=3):
    """Splits all signals by a divisor of the shortest signal

    Args:
        signals (list): the signals to be split
        fs (float): the sampling rate
        ids (ndarray): the ids from each signal
        alpha (int): the shortest signal divisor

    Returns:
        dataset (ndarray): splits of signals at the specified length
    """
    shortest = _find_shortest(signals)
    samplesize = math.ceil(shortest.size / alpha)
    splits = _split_all(signals, fs, ids, samplesize)
    return splits


def _split_all(signals, fs, ids, nsamples):
    nsamples = _fix_duration(signals, nsamples, fs)
    splits = []
    expanded_ids = []
    for signal, id_ in zip(signals, ids):
        sigsplits = _split(signal, nsamples)
        splits.extend(sigsplits)
        expanded_ids.extend([id_] * sigsplits.shape[0])
    return np.array(splits), np.array(expanded_ids)


def _fix_duration(signals, nsamples, fs):
    max_duration = _find_max_duration(signals, fs)
    duration = math.ceil(nsamples/fs * 1000)
    _warn_duration(duration, max_duration)
    wraped_duration = max(min(max_duration, duration), _MIN_DURATION)
    return math.ceil(wraped_duration/1000 * fs)


def _warn_duration(duration, max_duration):
    if max_duration < duration:
        raise Warning(f"Longest signal has {max_duration}ms, using this instead"
                      f" of {duration}ms")
    if duration < _MIN_DURATION:
        raise Warning(f"Min duration is {_MIN_DURATION}ms, using this instead"
                      f" of {duration}ms")


def _find_shortest(signals):
    shortest = signals[0]
    for signal in signals:
        if signal.size < shortest.size:
            shortest = signal
    return shortest


def _find_max_duration(signals, fs):
    max_duration = signals[0].size
    for signal in signals[1:]:
        if max_duration < signal.size:
            max_duration = signal.size
    return max_duration/fs * 1000


def _split(signal, nsamples):
    nsplits = math.ceil(signal.size / nsamples)
    padlen = nsplits*nsamples - signal.size
    paddedsignal = np.append(signal, np.zeros(padlen))
    return paddedsignal.reshape((nsplits, nsamples))

