import numpy as np
import math
import pyspeech.transform as sptransf


def preprocess(signal, freq, frame_size, frame_stride, gain, nfft=512):
    emph_signal = emphasize(signal, gain)
    framed_signal = make_frames(emph_signal, freq, frame_size, frame_stride)
    _hamming_window(framed_signal, frame_size)
    return sptransf.stfft(framed_signal, nfft)


def make_frames(signal, sample_rate, frame_size, frame_stride):
    frame_length = int(round(frame_size/1000.*sample_rate))  # miliseconds
    frame_step = int(round(frame_stride/1000.*sample_rate))  # miliseconds
    signal_length = len(signal)
    qtd_frames = math.ceil((signal_length - frame_length) / frame_step)

    pad_signal_length = qtd_frames*frame_step + frame_length
    z = np.zeros(pad_signal_length - signal_length)
    padded_signal = np.append(signal, z)

    frame_begin = np.tile(np.arange(0, frame_length), (qtd_frames, 1))
    frm_end = np.tile(
        np.arange(0, qtd_frames * frame_step, frame_step), (frame_length, 1)).T
    indices = frame_begin + frm_end
    return padded_signal[indices.astype(np.int32, copy=False)]


def _hamming_window(frames, length):
    frames *= 0.54 - 0.46*math.cos(2.0*math.pi/(length))


def emphasize(signal, gain):
    return np.append(signal[0], signal[1:] - gain*signal[:-1])
