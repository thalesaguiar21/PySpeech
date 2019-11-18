""" This module contains several signal processing functions, such as
emphasis, framing, and power spectrums. Besides a few data classes to
encapsulate arguments.
"""
import math
import numpy as np


from pyspeech.configs import confs


def frame_len(size, freq):
    return int(round(size/1000. * freq))


def frame_step(stride, freq):
    return int(round(stride/1000. *freq))


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


def norm_log_power_spectrum(signal):
    log_spec = _log_power_spectrum(signal)
    return log_spec - np.max(log_spec)


def log_power_spectrum(signal, frame, nfft):
    pow_spec = _power_spectrum(signal)
    bounded_psec = np.fmax(pow_spec, np.finfo(np.float64).eps)
    log_spec = np.log10(bounded_psec)


def power_spectrum(signal):
    mag_spec = _mag_spectrum(signal)
    return 1.0/nfft * mag_spec**2


def mag_spectrum(signal):
    wnd_amps = _split(signal)
    spectrum = np.fft.rfft(frames, confs['nfft'])
    return np.absolute(spectrum)


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
    nframes = int(math.floor(signal.size / flen))
    padding = flen - (signal.size - nframes*flen)
    padded_amps = np.append(signal.amps, np.zeros((padding)))
    frames = np.reshape(padded_amps, (nframes + 1, flen))
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


def find_best_nfft(freq, flen=None):
    flen = frame_len(confs['frame_size'], freq) if flen is None else flen
    if freq > 0 and flen > 0:
        nfft = 1
        while nfft < freq * flen:
            nfft *= 2
        return nfft
    else:
        raise ValueError('Frame length and frequency must be greater than zero')

