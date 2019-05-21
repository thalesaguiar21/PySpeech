import pyspeech.features as spfeat
from scipy.io import wavfile
import pdb

audiopath = 'samples/33711__acclivity__excessiveexposure.wav'
freq, signal = wavfile.read(audiopath)
mfccs = spfeat.mfcc(signal, freq, nfilt=40)
pdb.set_trace()
