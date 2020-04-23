import os

from scipy.io import wavfile

from .context import pyspeech
from pyspeech.dsp.processing import Signal


SIGNALPATH01 = os.path.abspath('tests/voice/OSR_us_000_0011_8k.wav')
SIGNALPATH02 = os.path.abspath('tests/voice/OSR_us_000_0012_8k.wav')

fs1, amps01 = wavfile.read(SIGNALPATH01)
fs2, amps02 = wavfile.read(SIGNALPATH02)

signal01 = Signal(amps01, fs1)
signal02 = Signal(amps02, fs2)

