from scipy.io import wavfile
import pyspeech.features as spfeat
import pyspeech.folder as spfold
import pyspeech.dsp.processing as spproc


nfilters = 40
# Build Processor
frame = spproc.Frame(size=25, stride=10)
processor = spproc.Processor(frame, emph=0.97, nfft=512)

# Extracting 13 MFCC from one audio
audiopath = 'samples/OSR_us_000_0011_8k.wav'
freq, signal = wavfile.read(audiopath)
signal = signal[:int(freq * 3.5)]
mfccs = spfeat.mfcc(signal, freq, nfilters, processor)
mfccs13 = mfccs[:, 1:14]  # Keep 13 mfccs


# Extracting 13 MFCCs from several audios
audios_path = spfold.find_wav_files('samples')
frequencies = []
signals = []
for audio_path in audios_path:
    freq, signal = wavfile.read(audio_path)
    frequencies.append(freq)
    signals.append(signal)

feats = spfeat.mfcc(signals, frequencies, nfilters, processor)
feats[0] # MFCCs for first sample
feats[1] # MFCCs for second sample
