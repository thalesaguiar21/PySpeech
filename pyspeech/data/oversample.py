import math
import numpy as np


from ..dsp import processing as sproc


def split_by_shorter(signals, fs):
    shortest = _find_shortest(signals)
    print(f"Shorter signal: {shortest.size / fs}")
    splits = _split_all(signals, shortest.size)
    return splits


def _find_shortest(signals):
    shortest = signals[0]
    for signal in signals:
        if signal.size < shortest.size:
            shortest = signal
    return shortest


def by_segment(signals, fs):
    segments = []
    for signal in signals:
        sig_segments = _voiced_segments(signals[0], fs)
        segments.extend(sig_segments)
    padded_segments = _fill_with_zeros(segments)
    return np.array(padded_segments)


def _voiced_segments(signal, fs, sil_thres=1e-2):
    normsignal = sproc.normalise(sproc.Signal(signal, fs))
    segments = []
    segment = []
    for amp in normsignal.amps:
        if abs(amp) > sil_thres:
            segment.append(amp)
        elif len(segment) > 0:
            segments.append(segment[:])
            segment.clear()
    return segments


def _fill_with_zeros(segments):
    largest = segments[0]
    for segment in segments[1:]:
        if len(segment) > len(largest):
            largest = segment

    zerofilled = []
    largest_size = len(largest)
    for segment in segments:
        padlen = largest_size - len(segment)
        padded_seg = np.append(segment, np.zeros(padlen))
        zerofilled.append(padded_seg)
    return zerofilled


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

