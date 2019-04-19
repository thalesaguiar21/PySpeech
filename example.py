import pyspeech.processing as spproc
import pyspeech.features as spfeat
import pyspeech.filters as spfilt
import pyspeech.transform as sptrans


windowed_signals = spproc.window_voice_dataset(
    '/home/thalesaguiar/MEGA/artificial-intelligence/databases/speech/english_small/train/voice/',
    0.97,
    25,
    10
)

proc_fb = 1
total_fb = len(windowed_signals)
for frames in windowed_signals:
    print('Extracting ', proc_fb, '/', total_fb, '...', end='\r', sep='', flush=True)
    pow_frames = sptrans.fft(frames, 512)
    fbanks = spfilt.bark_banks(pow_frames, 40, 16000, 512)
    # spfeat.mfcc(fbanks, 12)
    proc_fb += 1
print()
