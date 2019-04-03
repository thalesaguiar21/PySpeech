import pyspeech.processing as spproc
import pyspeech.features as spfeat
import pyspeech.filters as spfilt
import pyspeech.transform as sptrans


windowed_signals = spproc.window_voice_dataset(
    'C:\\DATASETS\\english_small\\train\\voice',
    0.97,
    25,
    10
)

proc_fb = 1
total_fb = len(windowed_signals)

for frames in windowed_signals:
    print('AUDIO ', proc_fb, '/', total_fb, '...', end='\r', sep='')
    pow_frames = sptrans.fft(frames, 512)
    fbanks = spfilt.compute_filter_banks(pow_frames, 40, 16000, 512)
    spfeat.mfcc(fbanks, 12)
    proc_fb += 1
