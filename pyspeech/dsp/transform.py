import numpy as np


from . import frame


def zr_rate(signal):
    pass


def log_energy(signal):
    st_energies = list(short_time_energy(signal))
    return 10 * np.log10(st_energies)


def short_time_energy(signal):
    window = np.hamming(frame.flength(signal))
    for fr in frame.striding(signal):
        windowed_sig = fr * window
        yield np.sum(windowed_sig ** 2)

