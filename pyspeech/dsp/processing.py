""" This module contains several signal processing functions, such as
emphasis, framing, and power spectrums. Besides a few data classes to
encapsulate arguments.
"""
import math
import numpy as np


from pyspeech.configs import confs


def frame_len(freq):
    return int(round(confs['frame_size']/1000. * freq))


def frame_step(freq):
    return int(round(confs['frame_stride']/1000. *freq))


def split(signals):
    """ Splits signals into frames

    Args:
        signals (list:Signal): The signals to be split
        frame (Frame): The frame size and stride

    Yields:
        frames (ndarray): The frammed signal

    Example:
        >>> sig = Signal(np.arange(24000), 80)
        >>> frame = Frame(25, 10)
        >>> list(split(signal, frame))
        [[1, 2],
         [2, 3],
         ...
         [22998, 22999]]
    """
    for signal in signals:
       yield _split(signal)


def _split(signal):
    flen = frame_len(signal.samplerate)
    fstride= frame_step(signal.samplerate)
    if fstride == 0:
        fstride = 1
        nframes = int(signal.amps.size / flen)
    else:
        nframes = 1 + int(math.ceil((signal.size - flen) / fstride))
    padding = (nframes-1)*fstride + flen - signal.size
    padded_amps = np.append(signal.amps, np.zeros((padding)))
    indices = np.tile(np.arange(0, flen), (nframes, 1)) + np.tile(np.arange(0, nframes * fstride, fstride), (flen, 1)).T
    indices.astype(dtype=np.int32)
    frames = padded_amps[indices]
    return frames


def remove_silence(signals, threshold=0.3):
    """ Removes silence from signal based on maximum aplitude

    Args:
        signal (list:Signal): The signals to remove silence
        frame (Frame): The frame size and stride

    Yield:
        The signal with amplitudes > threshold
    """
    for signal in signals:
        yield _remove_silence(signal, threshold)


def _remove_silence(signal, freq, threshold):
    or_frames = _split(signal)
    norm_frames = _split(normalise(signal))
    voiced_indexes = _get_voiced_indexes(norm_frames, threshold)
    voiced_frames = or_frames[voiced_indexes]
    return np.reshape(voiced_frames, voiced_frames.size)


def _get_voiced_indexes(frames, threshold):
    nframes = frames.shape[0]
    non_sil_indexes = []
    for i in range(nframes + 1):
        if np.absolute(frames[i]).max() > threshold:
            non_sil_indexes.append(i)
    return non_sil_indexes


def emphasize(signal, gain):
    return np.append(signal[0], signal[1:] - gain*signal[:-1])


def normalise(signal):
    max_amp = np.absolute(signal.amps).max()
    if max_amp == 0.0:
        amps = signal.amps
    else:
        amps = signal.amps / max_amp
    normalised_signal = Signal(amps, signal.samplerate)
    return normalised_signal


class Signal:
    """ A simple class to represent a digital signal

    Atributes:
        amps (ndarray): The amplitudes
        size (ndarray): The signal length
        freq (ndarray): The sampling rate
    """
    def __init__(self, amps, freq):
        self.amps = amps
        self.size = len(amps)
        self.samplerate= freq


def find_best_nfft(freq, flen=None):
    flen = frame_len(confs['frame_size'], freq) if flen is None else flen
    if freq > 0 and flen > 0:
        nfft = 1
        while nfft < freq * flen:
            nfft *= 2
        return nfft
    else:
        raise ValueError('Frame length and frequency must be greater than zero')

