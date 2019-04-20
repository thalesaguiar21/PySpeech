import numpy as np
import pyspeech.transform as sptrans
from enum import Enum, auto


class Feats(Enum):
    MFCC = auto()
    PLP = auto()
    JITTER_ABS = auto()
    JITTER_RAP = auto()
    JITTER_PPQN = auto()
    JITTER_DDP = auto()
    SHIMER = auto()
    SHIMMER_DB = auto()
    SHIMMER_APQN = auto()
    SHIMMER_DDA = auto()
    RPDE = auto()
    DFA = auto()
    PPE = auto()


def extract(feats_types, frames, qtd_ceps):
    ''' Create a 2D array with appended features in the given order '''
    features = []
    if Feats.MFCC in feats_types:
        features.append(mfcc_means(frames, qtd_ceps))
    if Feats.PLP in feats_types:
        features.append(plp())
    if Feats.JITTER_ABS in feats_types:
        features.append(jitter_abs())
    if Feats.JITTER_DDP in feats_types:
        features.append(jitter_ddp())
    if Feats.JITTER_PPQN in feats_types:
        features.append(jitter_ppqn())
    if Feats.JITTER_DDP in feats_types:
        features.append(jitter_ddp())
    if Feats.SHIMMER in feats_types:
        features.append(shimmer())
    if Feats.SHIMMER_DB in feats_types:
        features.append(shimmer_db())
    if Feats.SHIMMER_DDA in feats_types:
        features.append(shimmer_dda())
    if Feats.RPDE in feats_types:
        features.append(rpde())
    if Feats.DFA in feats_types:
        features.append(dfa())
    if Feats.PPE in feats_types:
        features.append(ppe())
    return features


def mfcc(frames, qtd_ceps):
    # Applies a Discrete Correlation Transforma(DCT) on Filter Banks
    MFs = np.array([np.absolute(sptrans.dctII_onedim(f)) for f in frames])
    mfccs = np.log(MFs)
    return mfccs[:, 1:qtd_ceps + 1]


def mfcc_means(frames, qtd_ceps):
    # Applies a Discrete Correlation Transforma(DCT) on Filter Banks
    return np.mean(mfcc(frames, qtd_ceps), axis=0)


def deltas(feats):
    ''' Compute the first derivative of a given feature vector '''
    if len(feats.shape) == 1:
        return sum(_deltas(feats))
    elif len(feats.shape) == 2:
        deltas = np.zeros((feats.shape[1], ))
        for fi in feats:
            deltas += _deltas(fi)
        return deltas
    else:
        raise ValueError(
            'Feature dimension ({}) not supported!'.format(len(feats.shape)))


def _deltas(feats):
    ''' Compute the first derivative of a given array '''
    vec_size = feats.size
    forwarded_feats = feats[1:] - feats[:-1]
    denom = 2 * sum([i**2.0 for i in range(vec_size - 1)])
    coefs = np.array([i / denom for i in range(vec_size - 1)])
    return forwarded_feats * coefs


def log_energy(windowed_frames):
    epsilon = 10e-5
    frame_energies = []
    frame_size = windowed_frames.shape[1]
    for frame in windowed_frames:
        arg = 1.0/frame_size * frame**2
        frame_energies.append(10 * np.log10(epsilon + arg.sum()))
    return np.array(frame_energies)


def plp():
    # power spectrum
    # bark filter banks
    # equal-loudness preemphasis
    # intensity-to-loudness
    # linear prediction
    # cepstrum computation
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
