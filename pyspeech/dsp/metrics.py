import numpy as np


def hz2mel(hz):
    return 1127 * np.log(1 + hz/700)


def mel2hz(mel):
    return 700*np.exp(mel/1127) - 700

