import numpy as np
from processing import dctII_onedim


def mfcc(frames, qtd_ceps):
    # Applies a Discrete Correlation Transforma(DCT) on Filter Banks
    return np.array(
        [dctII_onedim(frame) for frame in frames])


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
