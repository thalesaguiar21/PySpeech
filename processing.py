import numpy as np
from scipy.io import wavfile
from math import ceil, cos, pi, sqrt
from folder import find_wav_files


def window_voice_dataset(folder, emph_rate, frame_size, frame_stride):
    audio_files_path = find_wav_files(folder)
    qtd_files = len(audio_files_path)
    crr_audio = 1
    onfreq_signals = []
    for audio_path in audio_files_path:
        print("Processing audio ", crr_audio, "/", qtd_files, "...",
              end="\r", sep="")
        onfreq_signals.append(
            windowed_signal(audio_path, emph_rate, frame_size, frame_stride)
        )
        crr_audio += 1
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
    qtd_frames = ceil((signal_length - frame_length) / frame_step)

    pad_signal_length = qtd_frames * frame_step + frame_length
    z = np.zeros(pad_signal_length - signal_length)
    padded_signal = np.append(signal, z)

    frame_begin = np.tile(np.arange(0, frame_length), (qtd_frames, 1))
    frm_end = np.tile(
        np.arange(0, qtd_frames * frame_step, frame_step), (frame_length, 1)).T
    indices = frame_begin + frm_end
    return padded_signal[indices.astype(np.int32, copy=False)]


def _hamming_window(frames, length):
    frames *= 0.54 - 0.46 * cos(2.0 * pi / (length))


def fft(frames, nfft):
    mag_spectrum = np.absolute(np.fft.rfft(frames, nfft))
    return 1.0 / nfft * mag_spectrum ** 2.0


def dctII_onedim(signal):
    indices = np.array([np.arange(signal.size)])
    arg = np.dot(pi * indices.T / (2 * signal.size), 2 * indices + 1)
    return np.dot(signal, np.cos(arg.T))


def dctII_onedim_ortho(signal):
    dcts = dctII_onedim(signal)
    dcts[0] *= 1 / sqrt(2)
    return sqrt(2 / signal.size) * dctII_onedim(signal)


def _gamma1d(i):
    if i == 0:
        return 1 / sqrt(2)
    else:
        return 1


def dct_2d(signal):
    if len(signal.shape) != 2:
        raise ValueError('Signal must be 2-dimensional!')

    mfcc = np.zeros(signal.shape)
    xN = signal.shape[0]
    yN = signal.shape[1]
    x_range = np.array([np.arange(xN) for _ in range(xN)])
    y_range = np.array([np.arange(yN) for _ in range(yN)])
    xcosines = np.cos(pi / xN * (x_range + 0.5) * x_range.T)
    ycosines = np.cos(pi / yN * (y_range + 0.5) * y_range.T)

    sep = 2 * sqrt(1 / (xN * yN))
    gammas = np.array([[_gamma2d(i, j) for j in range(yN)] for i in range(xN)])

    for i in range(xN):
        for j in range(yN):
            for x in range(xN):
                for y in range(yN):
                    mfcc[i, j] += gammas[i, j] * signal[x, y] * \
                        ycosines[j, y] * xcosines[i, x]
    return sep * mfcc


def _gamma2d(i, j):
    if i == 0 and j == 0:
        return 1 / 2
    elif (i > 0 and j == 0) or (i == 0 and j > 0):
        return 1 / sqrt(2)
    else:
        return 1
