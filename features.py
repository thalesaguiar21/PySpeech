import numpy as np
from math import pi, sqrt
import pdb


def mfcc(signal, qtd_ceps):
    # Applies a Discrete Correlation Transforma(DCT) on Filter Banks
    # pdb.set_trace()
    mfcc = np.zeros(signal.shape)
    xN = signal.shape[0]
    yN = signal.shape[1]
    x_range = np.array([np.arange(xN) for _ in range(xN)])
    y_range = np.array([np.arange(yN) for _ in range(yN)])
    xcosines = np.cos(pi / xN * (x_range + 0.5) * x_range.T)
    ycosines = np.cos(pi / yN * (y_range + 0.5) * y_range.T)

    sep = 2 * sqrt(1 / (xN * yN))
    gammas = np.array([[gamma(i, j) for j in range(yN)] for i in range(xN)])

    for i in range(xN):
        for j in range(yN):
            for x in range(xN):
                for y in range(yN):
                    mfcc[i, j] += gammas[i, j] * signal[x, y] * \
                        ycosines[j, y] * xcosines[i, x]
    return sep * mfcc


def gamma(i, j):
    if i == 0 and j == 0:
        return 1 / 2
    elif (i > 0 and j == 0) or (i == 0 and j > 0):
        return 1 / sqrt(2)
    else:
        return 1


def plp():
    raise NotImplementedError()


def lsf():
    raise NotImplementedError()


def jitter_abs():
    raise NotImplementedError()


def jitter_rap():
    raise NotImplementedError()


def jitter_ppqn(n=5):
    raise NotImplementedError()


def jitter_ddp():
    raise NotImplementedError()


def shimmer():
    raise NotImplementedError()


def shimmer_db():
    raise NotImplementedError()


def shimmer_apqn(n=3):
    raise NotImplementedError()


def shimmer_dda():
    raise NotImplementedError()


def rpde():
    raise NotImplementedError()


def dfa():
    raise NotImplementedError()


def ppe():
    raise NotImplementedError()


T = np.array([[1, 19, 37, 55, 73, 91, 109, 127],
              [19, 37, 55, 73, 91, 109, 127, 145],
              [37, 55, 73, 91, 109, 127, 145, 163],
              [55, 73, 91, 109, 127, 145, 163, 181],
              [73, 91, 109, 127, 145, 163, 181, 199],
              [91, 109, 127, 145, 163, 181, 199, 217],
              [109, 127, 145, 163, 181, 199, 217, 235],
              [127, 145, 163, 181, 199, 217, 235, 253]])

m = np.array([10, 20, 30])
M = np.array([m + i * 30 for i in range(5)])

print(mfcc(M, 12))
