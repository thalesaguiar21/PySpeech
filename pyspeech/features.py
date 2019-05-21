import numpy as np
import pyspeech.transform as sptrans
import pyspeech.processing as spproc
import pyspeech.filters as spfilt


def mfcc(signal, frequency, nfilt):
    # Applies a Discrete Correlation Transforma(DCT) on Filter Banks
    power_spec = spproc.powerspectrum(signal, frequency, 25, 10, 0.97)
    filtered_frames = spfilt.mel_banks(power_spec, nfilt, frequency, 512)
    dctII_frames = [sptrans.dctII_onedim(frame) for frame in filtered_frames]
    melfrequencies = [np.absolute(dctframe) for dctframe in dctII_frames]
    npmelfrencies = np.array(melfrequencies)
    mfccs = np.log(npmelfrencies)
    return mfccs


def deltas(feats):
    ''' Compute the first derivative of a given feature vector '''
    dim = len(feats.shape)
    if dim == 1:
        return sum(_deltas(feats))
    elif dim == 2:
        deltas = np.zeros((feats.shape[1], ))
        for fi in feats:
            deltas += _deltas(fi)
        return deltas
    else:
        raise ValueError('Dimension ({}) not supported!'.format(dim))


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
        arg = 1.0/frame_size * frame**2.0
        frame_energies.append(10 * np.log10(epsilon + arg.sum()))
    return np.array(frame_energies)


def mean_normalization(feature):
    feature -= np.mean(feature, axis=0) + 1e-8
