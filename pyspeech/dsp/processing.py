""" This module contains several signal processing functions, such as
emphasis, framing, and power spectrums. Besides a few data classes to
encapsulate arguments.
"""
import math
import numpy as np


class Processor:
    """ """
    def __init__(self, frame, emph, nfft):
        self.frame = frame
        self.emph = emph
        self.NFFT = nfft

    def preprocess(self, signal, freq):
        emph_signal = emphasize(signal, self.emph)
        framed_signal = self.make_frames(emph_signal, freq)
        framed_signal *= np.hamming(self.frame.length(freq))
        return sptransf.make_power_spectrum(framed_signal, self.NFFT)

    def make_frames(self, signal, sample_rate):
        frame_length = self.frame.length(sample_rate)
        frame_step = self.frame.step(sample_rate)
        signal_length = len(signal)
        qtd_frames = math.ceil((signal_length - frame_length) / frame_step)

        pad_signal_length = qtd_frames*frame_step + frame_length
        z = np.zeros(pad_signal_length - signal_length)
        padded_signal = np.append(signal, z)

        frame_begin = np.tile(np.arange(0, frame_length), (qtd_frames, 1))
        frm_end = np.tile(
            np.arange(0, qtd_frames * frame_step, frame_step),
            (frame_length, 1)).T
        indices = frame_begin + frm_end
        return padded_signal[indices.astype(np.int32, copy=False)]


class Frame:

    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def length(self, freq):
        return int(round(self.size/1000. * freq))

    def step(self, freq):
        return int(round(self.stride/1000. * freq))


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


def norm_log_pwoer_spectrum(signal, frame, nfft):
    log_spec = _log_power_spectrum(signal, frame, nfft)
    return log_spec - np.max(log_spec)


def log_power_spectrum(signal, frame, nfft):
    pow_spec = _power_spectrum(signal, frame, nfft)
    bounded_psec = np.fmax(pow_spec, np.finfo(np.float64).eps)
    log_spec = np.log10(bounded_psec)


def power_spectrum(signal, frame, nfft):
    mag_spec = _mag_spectrum(signal, frame, nfft)
    return 1.0/nfft * mag_spec**2


def mag_spectrum(signal, frame, nfft):
    wnd_amps = _split(signal, frame)
    spectrum = np.fft.rfft(frames, nfft)
    return np.absolute(spectrum)


def split(signals, frame):
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
       yield _split(signal, frame)


def _split(signal, frame):
    flen = frame.len(signal.samplerate)
    nframes = int(math.floor(signal.size / flen))
    padding = flen - (signal.size - nframes*flen)
    padded_amps = np.append(signal.amps, np.zeros((padding)))
    frames = np.reshape(padded_amps, (nframes + 1, flen))
    return frames


def remove_silence(signals, frame, threshold=0.3):
    """ Removes silence from signal based on maximum aplitude

    Args:
        signal (list:Signal): The signals to remove silence
        frame (Frame): The frame size and stride

    Yield:
        The signal with amplitudes > threshold
    """
    for signal in signals:
        yield _remove_silence(signal, frame, threshold)


def _remove_silence(signal, frame, freq, threshold):
    or_frames = _split(signal, frame)
    norm_frames = _split(normalise(signal), frame)
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
    return signal.amps / max_amp


def find_best_nfft(freq, frame_length):
    if freq > 0 and frame_length > 0:
        nfft = 1
        while nfft < freq * frame_length:
            nfft *= 2
        return nfft
    else:
        raise ValueError('Frame length and frequency must be greater than zero')

