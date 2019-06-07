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
        return int(round(self.size/1000. * freq))


def remove_silence(signal, frame, freq, threshold=0.3):
    frame_len = frame.length(freq)
    n_frames = int(math.floor(signal.size / frame_len))
    pad_length = frame_len - (signal.size - n_frames*frame_len)
    padded_signal = np.append(normalise(signal), np.zeros((pad_length)))
    frames = np.reshape(padded_signal, (n_frames + 1, frame_len))
    indices = [i for i in range(n_frames + 1) if frames[i].max() > threshold]
    non_silence_frames = frames[indices]
    return np.reshape(non_silence_frames, non_silence_frames.size)


def emphasize(signal, gain):
    return np.append(signal[0], signal[1:] - gain*signal[:-1])


def normalise(signal):
    return signal * 1.0/signal.max()
