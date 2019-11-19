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


def split_with_stride(signal):
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
    flen = frame_len(signal.samplerate)
    fstride= frame_step(signal.samplerate)
    if flen > signal.size:
        nframes = 1
    else:
        nframes = 1 + int(math.ceil((signal.size - flen) / fstride))
    padding = (nframes-1)*fstride + flen - signal.size
    padded_amps = np.append(signal.amps, np.zeros((padding)))
    base_idx = np.tile(np.arange(0, flen), (nframes, 1))
    idx_step = np.tile(np.arange(0, nframes * fstride, fstride), (flen, 1)).T
    indices = (base_idx + idx_step).astype(dtype=np.int32)
    frames = padded_amps[indices]
    return frames


def remove_silence(signal, threshold):
    """ Removes silence from signal based on maximum aplitude

    Args:
        signal (list:Signal): The signals to remove silence
        frame (Frame): The frame size and stride

    Returns:
        The signal with amplitudes > threshold
    """
    or_frames = _split(signal)
    norm_frames = _split(normalise(signal))
    voiced_indexes = _get_voiced_indexes(norm_frames, threshold)
    voiced_frames = or_frames[voiced_indexes]
    voiced_amps = np.reshape(voiced_frames, voiced_frames.size)
    return Signal(voiced_amps, signal.samplerate)


def _split(signal):
    flen = frame_len(signal.samplerate)
    nframes = math.ceil(signal.size / flen)
    padlen = nframes*flen - signal.size

    pad_amps = np.append(signal.amps, np.zeros(padlen))
    frames = np.reshape(pad_amps, (nframes, flen))
    return frames


def _get_voiced_indexes(frames, threshold):
    nframes = frames.shape[0]
    non_sil_indexes = []
    for i in range(nframes):
        if np.absolute(frames[i]).max() > threshold:
            non_sil_indexes.append(i)
    return non_sil_indexes


def emphasize(signals, gain):
    emph_amps = np.append(signal[0], signal[1:] - gain*signal[:-1])
    return Signal(emph_amps, signal.samplerate)


def normalise(signal):
    max_amp = np.absolute(signal.amps).max()
    if max_amp == 0.0:
        normalised_amps = signal.amps
    else:
        normalised_amps = signal.amps / max_amp
    return Signal(normalised_amps, signal.samplerate)


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

