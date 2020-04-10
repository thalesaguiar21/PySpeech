import numpy as np


from . import frame


def zr_rate(signal):
    rates = []
    flen = frame.flength(signal)
    for fr in frame.striding(signal):
        ncrossings = _zr_crossing(fr)
        rates.append(0.5*flen * ncrossings)
    return rates


def _zr_crossing(signal):
    crossings = 0
    signs = [sgn(amp) for amp in signal]
    for i in range(1, len(signs)):
        sgnsum = signs[i] + signs[i-1]
        crossings += abs(sgnsum)
    return crossings


def sgn(a):
    if a >= 0:
        return 1
    else:
        return -1


def log_energy(signal):
    st_energies = list(short_time_energy(signal))
    return 10 * np.log10(st_energies)


def short_time_energy(signal):
    window = np.hamming(frame.flength(signal))
    for fr in frame.striding(signal):
        windowed_sig = fr * window
        yield np.sum(windowed_sig ** 2)

