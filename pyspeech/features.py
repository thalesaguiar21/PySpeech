import numpy as np
import pyspeech.transform as sptrans


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
    denom = 2 * sum([i ** 2.0 for i in range(vec_size - 1)])
    coefs = np.array([i / denom for i in range(vec_size - 1)])
    return forwarded_feats * coefs


def log_energy(windowed_frames):
    epsilon = 10e-5
    frame_energies = []
    frame_size = windowed_frames.shape[1]
    for frame in windowed_frames:
        arg = (1.0 / frame_size) * frame ** 2
        frame_energies.append(10 * np.log10(epsilon + arg.sum()))
    return np.array(frame_energies)


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
