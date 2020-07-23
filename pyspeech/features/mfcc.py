import numpy as np
import scipy

from ..dsp import processing
from ..dsp import frame
from ..dsp import spectrum
from ..dsp import shorttime
from ..dsp.metrics import hz2mel, mel2hz
from .. import conf


def extract(signal, emph, nfilt, spam, nceps=13, nlift=22):
    """ Extract Mel-Frequency Cepstrum Coefficients based on the HTK

    Args:
        signal (processing.Signal): The signal to extract
        emph (float): the pre-emphasis gain
        nfilt (int): the number of triangular filters
        spam (tuple): a (low, high) cutoff freqeuncies for the filter design
        nceps (int): the number of cepstrums to keep (excluding 0th), defaults
            to 13.
        nlift (int): the cepstral liftering, defaults to 22

    Returns:
        A Nframes X nfilt array of Mel-Frequency Cepstrum Coefficients
    """
    flen, fstride = frame.size(signal.fs), frame.stride(signal.fs)
    user_nfft = conf.nfft
    conf.nfft = _next_pow2(flen)
    K = conf.nfft//2 + 1

    emph_signal = processing.emphasize(signal, emph)
    frames = frame.apply(emph_signal)
    wnd_frames = frames * np.hamming(flen)
    magnitude_spec = spectrum.magnitude(wnd_frames)
    trifilters = _make_filter_banks(nfilt, K, signal.fs, spam)

    filter_banks = trifilters @ magnitude_spec.T
    # Log-fbanks converted back to frames as rows
    log_fbanks = np.log(filter_banks).T
    ceps = scipy.fft.dct(log_fbanks, type=3, n=nceps, norm='ortho', axis=1)
    lifts = _cep_lift(nceps, nlift)
    lifted_ceps = ceps * lifts

    mfccs = lifted_ceps[:, 1:]
    conf.nfft = user_nfft
    if conf.append_energy:
        egys = shorttime.log_energy(frames)[:, None]
        return np.hstack((mfccs, egys))
    # Rollback to user defined nfft
    return mfccs


def _next_pow2(x):
    return 1 << (x-1).bit_length()


def _make_filter_banks(nfilt, filt_len, fs, spam):
    fmin = 0
    f_low, f_high = spam
    fmax = 0.5 * fs
    mel_low, mel_high = hz2mel(f_low), hz2mel(f_high)

    f = np.linspace(fmin, fmax, filt_len)
    norm_factor = (mel_high-mel_low) / (nfilt + 2)
    mel_cut = mel_low + np.arange(0, nfilt + 2)*norm_factor
    hz_cut = mel2hz(mel_cut)

    trifilts = np.zeros((nfilt, filt_len))
    for m in range(nfilt):
        # Up
        k = (f >= hz_cut[m]) & (f <= hz_cut[m + 1])
        trifilts[m, k] = (f[k]-hz_cut[m]) / (hz_cut[m + 1]-hz_cut[m])
        # Down
        k = (f >= hz_cut[m + 1]) & (f <= hz_cut[m + 2])
        trifilts[m, k] = (hz_cut[m + 2]-f[k]) / (hz_cut[m + 2]-hz_cut[m + 1])
    return trifilts


def _cep_lift(size, nlift):
    return 1 + 0.5*nlift * np.sin(np.pi*np.arange(0, size) / nlift)

