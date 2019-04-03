import numpy as np
import pyspeech.transform as sptrans
import pdb


def mfcc(frames, qtd_ceps):
    # Applies a Discrete Correlation Transforma(DCT) on Filter Banks
    MFs = np.array([np.absolute(sptrans.dctII_onedim(f)) for f in frames])
    mfccs = np.log(MFs)
    return mfccs[:, 1:qtd_ceps + 1]


def mfcc_means(frames, qtd_ceps):
    # Applies a Discrete Correlation Transforma(DCT) on Filter Banks
    return np.mean(mfcc(frames, qtd_ceps), axis=0)


def mfcc_deltas(mfccs):
    # Calculate Delta or Delta-DEltas from MFCC vector
    n_frames = mfccs.shape[1]
    f_frames = mfccs[:, 1:] - mfccs[:, :-1]
    denom = 2 * sum([i ** 2 for i in range(n_frames - 1)])
    coef = np.array([i / denom for i in range(n_frames - 1)])
    for i in range(f_frames.shape[0]):
        f_frames[i] *= coef
    return np.sum(f_frames, axis=1)


def log_energy(windowed_frames):
    epsilon = 10e-5
    frame_energies = []
    frame_size = windowed_frames.shape[1]
    for frame in windowed_frames:
        arg = (1.0 / frame_size) * frame ** 2
        frame_energies.append(10 * np.log10(epsilon + arg.sum()))
    return np.array(frame_energies)


def energy_delta(log_energies):
    qtd_energies = log_energies.size
    forwarded = log_energies[1:] - log_energies[:-1]
    denom = 2 * sum([i ** 2.0 for i in range(qtd_energies - 1)])
    coefs = np.array([i / denom for i in range(qtd_energies - 1)])
    return forwarded * coefs


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
