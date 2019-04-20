import numpy as np
import math
import pyspeech.folder as spfold
import pyspeech.transform as sptrans
from scipy.io import wavfile


def powerspectrum(datapath, emphasis, frame_size, frame_stride):
    windowed_signal = window_voice_dataset(
        datapath, emphasis, frame_size, frame_stride)
    return sptrans.fft(windowed_signal, 512)


def window_voice_dataset(data_path, emph_rate, frame_size, frame_stride):
    audio_files_path = spfold.find_wav_files(data_path)
    onfreq_signals = []
    for audio_path in audio_files_path:
        onfreq_signals.append(
            windowed_signal(audio_path, emph_rate, frame_size, frame_stride)
        )
    return onfreq_signals


def windowed_signal(audio, emph_rate, frame_size, frame_stride):
    rate, signal = wavfile.read(audio)
    emph_signal = _preemph(signal, emph_rate)
    frames = _split(rate, emph_signal, frame_size, frame_stride)
    frame_length = frames.shape[1]
    _hamming_window(frames, frame_length)
    return frames


def _preemph(signal, alpha):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def _split(sample_rate, signal, frame_size, frame_stride):
    frame_length = int(round(frame_size / 1000. * sample_rate))  # miliseconds
    frame_step = int(round(frame_stride / 1000. * sample_rate))  # miliseconds
    signal_length = len(signal)
    qtd_frames = math.ceil((signal_length - frame_length) / frame_step)

    pad_signal_length = qtd_frames * frame_step + frame_length
    z = np.zeros(pad_signal_length - signal_length)
    padded_signal = np.append(signal, z)

    frame_begin = np.tile(np.arange(0, frame_length), (qtd_frames, 1))
    frm_end = np.tile(
        np.arange(0, qtd_frames * frame_step, frame_step), (frame_length, 1)).T
    indices = frame_begin + frm_end
    return padded_signal[indices.astype(np.int32, copy=False)]


def _hamming_window(frames, length):
    frames *= 0.54 - 0.46 * math.cos(2.0 * math.pi / (length))


def equal_loudness_preemphasis():
    pass
