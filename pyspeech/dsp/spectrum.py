import numpy as np

from pyspeech.configs import confs


def norm_log_power_spectrum(signal):
    log_spec = _log_power_spectrum(signal)
    return log_spec - np.max(log_spec)


def log_power_spectrum(signal, frame, nfft):
    pow_spec = _power_spectrum(signal)
    bounded_psec = np.fmax(pow_spec, np.finfo(np.float64).eps)
    log_spec = np.log10(bounded_psec)


def power_spectrum(signal):
    mag_spec = _mag_spectrum(signal)
    return 1.0/nfft * mag_spec**2


def mag_spectrum(signal):
    wnd_amps = _split(signal)
    spectrum = np.fft.rfft(frames, confs['nfft'])
    return np.absolute(spectrum)

