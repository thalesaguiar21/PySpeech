from dataclasses import dataclass

import numpy as np
import scipy.fftpack as scifft

from .. import conf
from ..dsp import frame
from ..dsp import processing as sp
from ..dsp import spectrum as spec
from ..dsp import metrics as smet
from ..features import derivs as sder


def extract(signal, mfcc, melfilter, emph):
    """ Extract the Mel-Frequency Cepstrum Coefficients from a signal

    Note:
        If conf.append_energy is True, this function will append the log
        energy of each frame

    Args:
        emph (float): The emphasis to the signal

    Returns:
        The log-energy + MFCC or MFCC
    """
    powspec = _make_power_spectrum(signal, emph)
    srate = signal.fs
    if conf.append_energy:
        feats = _extract_mfcc_and_energy(powspec, mfcc, melfilter, srate)
    else:
        feats = _extract_mfcc(powspec, mfcc, melfilter, srate)
    return feats


def _extract_mfcc_and_energy(powspec, mfcc, melfilter, srate):
    mfccs = _extract_mfcc(powspec, mfcc, melfilter, srate)
    energies = sder.log_energy(powspec)
    return np.hstack((energies.reshape(energies.size, 1), mfccs))


def _extract_mfcc(powspec, mfcc, melfilter, srate):
    fbanks_log = _make_log_fbanks(powspec, melfilter, srate)
    cepstrums = scifft.dct(fbanks_log, type=2, axis=1, norm='ortho')
    cepstrums_cut = cepstrums[:, :mfcc.ncep]
    lifted_cepstrums = lifter(cepstrums_cut, mfcc.lift)
    return lifted_cepstrums


def _make_power_spectrum(signal, emph):
    wnd_signal = _make_frames_and_window(signal, emph)
    return spec.power(wnd_signal)


def _make_frames_and_window(signal, emph):
    emph_signal = sp.emphasize(signal, emph)
    frames = frame.apply(signal)
    ham_frames = frames * np.hamming(frames.shape[1])
    return ham_frames


def _make_log_fbanks(powspec, melfilter, srate):
    fbanks = _make_filter_banks(melfilter, srate)
    fbanks_energies = powspec @ fbanks.T
    bounded_fbanks = np.fmax(fbanks_energies, np.finfo(np.float64).eps)
    return 20 * np.log(bounded_fbanks)


def _make_filter_banks(melfilter, srate):
    bins = _make_bins(melfilter, srate)
    return _make_fbanks(melfilter, bins)


def _make_bins(melfilter, srate):
    if melfilter.highfreq is None:
        melfilter.highfreq = srate/2
    highmel = smet.hz_to_mel(melfilter.highfreq)
    lowmel = smet.hz_to_mel(melfilter.lowfreq)
    mel_points = np.linspace(lowmel, highmel, melfilter.nfilt + 2)
    bins = np.floor((conf.nfft + 1)*smet.mel_to_hz(mel_points) / srate)
    return bins


def _make_fbanks(melfilter, bins):
    fbanks = np.zeros((melfilter.nfilt, conf.nfft//2 + 1 ))
    for j in range(1, melfilter.nfilt + 1):
        fminus = int(bins[j - 1])
        f = int(bins[j])
        fplus = int(bins[j + 1])
        for i in range(fminus, f):
            fbanks[j - 1, i] = (i-bins[j - 1]) / (bins[j]-bins[j - 1])
        for i in range(f, fplus):
            fbanks[j - 1, i] = (bins[j + 1]-i) / (bins[j + 1]-bins[j])
    return fbanks


@dataclass
class MFCC:
    ncep: int
    lift: int
    powspec: np.ndarray = np.array([])


@dataclass
class MelFilter:
    nfilt: int
    highfreq: float
    lowfreq: float


def lifter(cepstrums, l=22):
    if l > 0:
        nframes, ncoefs = cepstrums.shape
        n = np.arange(ncoefs)
        lift = 1 + (l/2)*np.sin(np.pi*n / l)
        return lift * cepstrums
    else:
        return cepstrums

