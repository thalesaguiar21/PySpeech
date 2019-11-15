import numpy as np


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz/700)


def mel_to_hz(mel):
    return 700 * (10**(mel/2595) - 1.0)

