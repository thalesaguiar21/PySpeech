import numpy as np
import scipy.fftpack as scifft

import pyspeech.dsp.processing as sp
import pyspeech.dsp.spectrum as spec
import pyspeech.dsp.metrics as smet
import pyspeech.features.derivs as sder
from pyspeech.configs import confs


class Extractor:

    def __init__(self, melfilter, ncep=13, lift=22):
        self.powspec = np.array([])
        self.ncep = ncep
        self.melfilter = melfilter
        self.lift = lift

    def extract(signal, emph):
        self.powsepc = _make_power_spectrum(signal, emph)
        if confs['append_energy']:
            feats = self.mfcc_and_energy(power_spectrum)
        else:
            feats = self.mfcc(power_spectrum)
        return feats

    def mfcc_and_energy(self):
        mfccs = self.mfcc()
        energies = sder.log_energy(self.powspec)
        return np.hstack((energies.reshape(energies.size, 1), mfccs))

    def mfcc(self):
        filter_banks = self.melfilter.make()
        fbanks_energies = self.powspec @ filter_banks.T
        bound_energies = np.fmax(fbanks_energies, np.finfo(np.float64).eps)
        fbanks_log = np.log(bound_energies)
        cepstrums = scifft.dct(fbanks_log, type=2, axis=1, norm='ortho')
        cepstrum_cut = cepstrums[:, :self.ncep]
        lifted_cepstrums = lifter(cepstrums_cut, self.ceplift)
        return lifted_cepstrums


class MelFilter:

    def __init__(self, nfilt, highfreq, lowfreq):
        self.nfilt = nfilt
        self.highfreq = highfreq if highfreq is not None else srate/2
        self.lowfreq = lowfreq

    def make(self):
        bins = self.bins()
        fbanks = self.fbanks(bins)
        return fbanks

    def bins(self):
        hihgmel = smet.hz_to_mel(self.highfreq)
        lowmel = smet.hz_to_mel(self.lowfreq)
        mel_points = np.linspace(lowmel, highmel, nfilt + 2)
        bins = np.floor((confs['nfft'] + 1)*smet.mel_to_hz(mel_points) / srate)
        return bins

    def fbanks(self, bins):
        fbanks = np.zeros((self.nfilt, confs['nfft']//2 + 1))
        for j in range(self.nfilt):
            for i in range(int(bins[j]), int(bins[j + 1])):
                fbanks[j, i] = (i-bins[j]) / (bins[j + 1]-bins[j])
            for i in range(int(bins[j + 1]), int(bins[j + 2])):
                fbanks[j, i] = (bins[j + 2]-i) / (bins[j + 2]-bins[j + 1])
        return fbanks


@dataclass
class MFCC:
    ncep: int
    lift: int
    powspec: np.ndarray


@dataclass
class MelFilter:
    nfilt: int
    highfreq: float
    lowfreq: float


def extract(signal, mfcc, melfilter, emph):
    powspec = _make_power_spectrum(signal, emph)
    if confs['append_energy']:
        feats = _extract_mfcc_and_energy(powspec, mfcc, melfilter, emph,
                                         signal.samplerate)
    else:
        feats = _extract_mfcc(powspec, nfilt, ncep, emph, ceplift,
                              signal.samplerate, lowfreq, highfreq)
    return feats


def _extract_mfcc_and_energy(powspec, nfilt, ncep, emph, ceplift, srate,
                             lowfreq, highfreq=None):
    mfccs = _extract_mfcc(powspec, nfilt, ncep, ceplift, srate,
                          lowfreq, highfreq)
    energies = sder.log_energy(powspec)
    return np.hstack((energies.reshape(energies.size, 1), mfccs))


def _extract_mfcc(powspec, nfilt, ncep, emph, ceplift, srate,
                  lowfreq, highfreq=None):
    filter_banks = make_filter_banks(powspec, highfreq, lowfreq, nfilt, srate)
    fbanks_energies = powspec @ filter_banks.T
    # Prevent zero for log
    fbanks_energies_cut = np.fmax(fbanks_energies, np.finfo(np.float64).eps)
    fbanks_log = np.log(fbanks_energies_cut)
    cepstrums = scifft.dct(fbanks_log, type=2, axis=1, norm='ortho')[:, :ncep]
    lifted_cepstrums = lifter(cepstrums, ceplift)
    return lifted_cepstrums


def _make_power_spectrum(signal, emph):
    wnd_signal = make_frames_and_window(signal, emph)
    return spec.power(wnd_signal)


def make_frames_and_window(signal, emph):
    emph_signal = sp.emphasize(signal, emph)
    frames = sp.split_with_stride(signal)
    ham_frames = frames * np.hamming(sp.frame_len(confs['frame_size']))
    return ham_frames


def make_filter_banks(power_spec, highfreq, lowfreq, nfilt, srate):
    highfreq = highfreq if highfreq is not None else srate/2
    highmel = smet.hz_to_mel(highfreq)
    lowmel = smet.hz_to_mel(lowfreq)
    mel_points = np.linspace(lowmel, highmel, nfilt + 2)
    bins = np.floor((confs['nfft'] + 1)*smet.mel_to_hz(mel_points) / srate)

    fbanks = np.zeros((nfilt, confs['nfft']//2 + 1 ))
    for j in range(nfilt):
        for i in range(int(bins[j]), int(bins[j + 1])):
            fbanks[j, i] = (i-bins[j]) / (bins[j + 1]-bins[j])
        for i in range(int(bins[j + 1]), int(bins[j + 2])):
            fbanks[j, i] = (bins[j + 2]-i) / (bins[j + 2]-bins[j + 1])
    return fbanks


def lifter(cepstrums, l=22):
    if l > 0:
        nframes, ncoefs = cepstrums.shape
        n = np.arange(ncoefs)
        lift = 1 + (l/2)*np.sin(np.pi*n / l)
        return lift * cepstrums
    else:
        return cepstrums

