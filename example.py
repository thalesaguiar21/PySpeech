from scipy.io import wavfile
import pyspeech.features as spfeat
import pyspeech.folder as spfold
import pdb

# Extracting 13 MFCC from one audio
audiopath = 'samples/OSR_us_000_0011_8k.wav'
freq, signal = wavfile.read(audiopath)
signal = signal[:int(freq * 3.5)]
mfccs = spfeat.mfcc(signal, freq, nfilt=40)
mfccs13 = mfccs[0][:, 1:14]  # Keep 13 mfccs


# Extracting 13 MFCCs from several audios
audios_path = spfold.find_wav_files('samples')
frequencies = []
signals = []
for audio_path in audios_path:
    freq, signal = wavfile.read(audio_path)
    frequencies.append(freq)
    signals.append(signal)

pdb.set_trace()
feats = spfeat.mfcc(signals, frequencies, 40)
