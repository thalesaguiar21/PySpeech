import math
from dataclasses import dataclass
import numpy as np
import pyspeech.dsp.transform as sptransf


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

    def __init__(self, amps, freq):
        self.amps = amps
        self.size = len(amps)
        self.freq = freq


def split(signals, frame):
    for signal in signals:
       yield _split(signal, frame)


def _split(signal, frame):
    flen = frame.len(signal.freq)
    nframes = int(math.floor(signal.size / flen))
    padding = flen - (signal.size - nframes*flen)
    padded_amps = np.append(signal.amps, np.zeros((padding)))
    frames = np.reshape(padded_amps, (nframes + 1, flen))
    return frames


def remove_silence(signals, frame, threshold=0.3):
    for signal in signals:
        yield _remove_silence(signal, frame, threshold)


def _remove_silence(signal, frame, freq, threshold):
    or_frames = _split(signal, frame)
    norm_frames = _split(normalise(signal.amps), frame)
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
    max_amp = np.absolute(signal).max()
    return signal / max_amp

