import numpy as np


from . import frame


def log_energy(frame):
    energy = list(st_energy(frame))
    return 10 * np.log10(energy)


def st_energy(frame):
    energies = (signal * np.hamming(frame.size))** 2
    return np.sum(energies)



