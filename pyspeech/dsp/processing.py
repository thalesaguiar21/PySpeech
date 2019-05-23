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
        _hamming_window(framed_signal, self.frame.size)
        return sptransf.stfft(framed_signal, self.NFFT)

    def make_frames(self, signal, sample_rate):
        frame_length = int(round(self.frame.size/1000.*sample_rate))
        frame_step = int(round(self.frame.stride/1000.*sample_rate))
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


@dataclass
class Frame:
    size: int
    stride: int


def _hamming_window(frames, length):
    frames *= 0.54 - 0.46*math.cos(2.0*math.pi/(length))


def emphasize(signal, gain):
    return np.append(signal[0], signal[1:] - gain*signal[:-1])


