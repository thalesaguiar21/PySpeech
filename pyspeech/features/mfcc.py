import numpy as np
import scipy.fftpack as scifft

import pyspeech.dsp.processing as sp
import pyspeech.dsp.spectrum as spec
import pyspeech.dsp.metrics as smet
import pyspeech.features.derivs as sder
from pyspeech.configs import confs


def extract(signal, nfilt, ncep, emph, ceplift, lowfreq, highfreq=None):
    features = np.array([])
    if confs['append_energy']:
        features = _extract_mfcc_and_energy(signal, nfilt, ncep, emph, ceplift,
                                            lowfreq, highfreq)
    else:
        features = _extract_mfcc(signal, nfilt, ncep, emph, ceplift, lowfreq,
                                 highfreq)
    return features


def _extract_mfcc(signal, nfilt, ncep, emph, ceplift, lowfreq, highfreq=None):
    wnd_signal = make_frames_and_window(signal, emph)
    power_spectrum = spec.power(wnd_signal)
    filter_banks = make_filter_banks(power_spectrum, highfreq, lowfreq,
                                     nfilt, signal.samplerate)

    fbanks_energies = power_spectrum @ filter_banks.T
    # Prevent zero for log
    fbanks_energies_cut = np.fmax(fbanks_energies, np.finfo(np.float64).eps)
    fbanks_log = np.log(fbanks_energies_cut)
    cepstrums = scifft.dct(fbanks_log, type=2, axis=1, norm='ortho')[:, :ncep]
    lifted_cepstrums = lifter(cepstrums, ceplift)
    return lifted_cepstrums

def _extract_mfcc_and_energy(signal, nfilt, ncep, emph, ceplift, lowfreq,
                             highfreq=None):
    wnd_signal = make_frames_and_window(signal, emph)
    power_spectrum = spec.power(wnd_signal)
    log_energies = sder.log_energy(power_spectrum)
    filter_banks = make_filter_banks(power_spectrum, highfreq, lowfreq,
                                     nfilt, signal.samplerate)

    fbanks_energies = power_spectrum @ filter_banks.T
    # Prevent zero for log
    fbanks_energies_cut = np.fmax(fbanks_energies, np.finfo(np.float64).eps)
    fbanks_log = np.log(fbanks_energies_cut)
    cepstrums = scifft.dct(fbanks_log, type=2, axis=1, norm='ortho')[:, :ncep]
    lifted_cepstrums = lifter(cepstrums, ceplift)
    return lifted_cepstrums


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

