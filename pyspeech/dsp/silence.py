import numpy as np

from . import processing as proc
from . import frame


def remove(signal, threshold):
    """ Removes silence from signal based on maximum aplitude

    Returns:
        The signal with amplitudes > threshold
    """
    or_frames = frame.striding(signal)
    norm_frames = frame.striding(proc.normalise(signal))
    voiced_indexes, __ = _detect_silence(norm_frames, threshold)
    voiced_frames = or_frames[voiced_indexes]
    voiced_amps = np.reshape(voiced_frames, voiced_frames.size)
    return proc.Signal(voiced_amps, signal.fs)


def _detect_silence(frames, threshold):
    non_sil_indexes = []
    sil_indexes = []
    for i, frm_energy in enumerate(_db_energy(frames)):
        if frm_energy > threshold:
            non_sil_indexes.append(i)
        else:
            sil_indexes.append(i)
    return non_sil_indexes, sil_indexes


def _db_energy(signals):
    for signal in signals:
        sqr_sum = np.sum(signal ** 2)
        yield 10 * np.log(sqr_sum)

