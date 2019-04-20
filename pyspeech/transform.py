import numpy as np
import math


def fft(frames, nfft):
    mag_spectrum = np.absolute(np.fft.rfft(frames, nfft))
    return 1.0/nfft * mag_spectrum**2.0


def dctII_onedim(signal):
    indices = np.array([np.arange(signal.size)])
    arg = np.dot(math.pi*indices.T / (2*signal.size), 2*indices + 1)
    return np.absolute(np.dot(signal, np.cos(arg.T)))


def dctII_onedim_ortho(signal):
    dcts = dctII_onedim(signal)
    dcts[0] *= 1 / math.sqrt(2)
    return math.sqrt(2 / signal.size) * dctII_onedim(signal)


def _gamma1d(i):
    if i == 0:
        return 1 / math.sqrt(2)
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
    xcosines = np.cos(math.pi/xN * (x_range + 0.5) * x_range.T)
    ycosines = np.cos(math.pi/yN * (y_range + 0.5) * y_range.T)

    sep = 2 * math.sqrt(1.0 / (xN*yN))
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
        return 1 / math.sqrt(2)
    else:
        return 1
