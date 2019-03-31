import numpy as np
from processing import dctII_onedim


def mfcc(frames, qtd_ceps):
    # Applies a Discrete Correlation Transforma(DCT) on Filter Banks
    mfccs = np.array([np.log(dctII_onedim(frame)) for frame in frames])
    return mfccs[:, 1:qtd_ceps + 1]


def mfcc_means(frames, qtd_ceps):
    # Applies a Discrete Correlation Transforma(DCT) on Filter Banks
    return np.mean(mfcc(frames, qtd_ceps), axis=0)


def deltas(mfccs):
    # Calculate Delta or Delta-DEltas from MFCC vector
    n_frames = mfccs.shape[1]
    f_frames = mfccs[:, 1:] - mfccs[:, :-1]
    denom = 2 * sum([i ** 2 for i in range(n_frames - 1)])
    coef = np.array([i / denom for i in range(n_frames - 1)])
    for i in range(f_frames.shape[0]):
        f_frames[i] *= coef
    return np.sum(f_frames, axis=1)


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


def mean_normalization(feature):
    feature -= np.mean(feature, axis=0) + 1e-8
