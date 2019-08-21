from scipy.io import wavfile
import pyspeech.features.features as spfeat
import pyspeech.folder as spfold
import pyspeech.dsp.processing as spproc


# Configure the signal processing module
nfilters = 40
frame = spproc.Frame(size=25, stride=10)
processor = spproc.Processor(frame, emph=0.97, nfft=512)

# Extracting 13 MFCC from one audio
audiopath = 'samples/OSR_us_000_0011_8k.wav'
freq, signal = wavfile.read(audiopath)
mfccs = spfeat.extract_mfcc(signal, freq, nfilters, processor, cepstrums=13)
print(mfccs.shape)

# Extracting 13 MFCCs from several audios
audios_path = spfold.find_wav_files('samples')
frequencies = []
signals = []
for audio_path in audios_path:
    freq, signal = wavfile.read(audio_path)
    frequencies.append(freq)
    signals.append(signal)

feats = spfeat.extract_mfcc(signals, frequencies, nfilters, processor, cepstrums=13)
feats[0] # MFCCs for first audio
print(feats[0].shape)

