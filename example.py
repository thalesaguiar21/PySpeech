import pyspeech.features as spfeat
from scipy.io import wavfile


audiopath = 'samples/OSR_us_000_0011_8k.wav'
freq, signal = wavfile.read(audiopath)
signal = signal[:int(freq * 3.5)]
mfccs = spfeat.mfcc(signal, freq, nfilt=40)
mfccs13 = mfccs[:, 1:14]  # Keep 13 mfccs
