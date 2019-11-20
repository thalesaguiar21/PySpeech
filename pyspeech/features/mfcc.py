from dataclasses import dataclass

import numpy as np
import scipy.fftpack as scifft

import pyspeech.dsp.processing as sp
import pyspeech.dsp.spectrum as spec
import pyspeech.dsp.metrics as smet
import pyspeech.features.derivs as sder
from pyspeech.configs import confs


def extract(signal, mfcc, melfilter, emph):
    powspec = _make_power_spectrum(signal, emph)
    srate = signal.samplerate
    if confs['append_energy']:
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
    frames = sp.split_with_stride(signal)
    ham_frames = frames * np.hamming(sp.frame_len(confs['frame_size']))
    return ham_frames


def _make_log_fbanks(powspec, melfilter, srate):
    fbanks = _make_filter_banks(melfilter, srate)
    fbanks_energies = powspec @ fbanks.T
    bounded_fbanks = np.fmax(fbanks_energies, np.finfo(np.float64).eps)
    return np.log(bounded_fbanks)


def _make_filter_banks(melfilter, srate):
    bins = _make_bins(melfilter, srate)
    return _make_fbanks(melfilter, bins)


def _make_bins(melfilter, srate):
    if melfilter.highfreq is None:
        melfilter.highfreq = srate/2
    highmel = smet.hz_to_mel(melfilter.highfreq)
    lowmel = smet.hz_to_mel(melfilter.lowfreq)
    mel_points = np.linspace(lowmel, highmel, melfilter.nfilt + 2)
    bins = np.floor((confs['nfft'] + 1)*smet.mel_to_hz(mel_points) / srate)
    return bins


def _make_fbanks(melfilter, bins):
    fbanks = np.zeros((melfilter.nfilt, confs['nfft']//2 + 1 ))
    for j in range(melfilter.nfilt):
        for i in range(int(bins[j]), int(bins[j + 1])):
            fbanks[j, i] = (i-bins[j]) / (bins[j + 1]-bins[j])
        for i in range(int(bins[j + 1]), int(bins[j + 2])):
            fbanks[j, i] = (bins[j + 2]-i) / (bins[j + 2]-bins[j + 1])
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

