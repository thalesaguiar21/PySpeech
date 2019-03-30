from processing import process_voice_dataset
from features import mfcc
from filters import compute_filter_banks


powframes = process_voice_dataset(
    'C:\\DATASETS\\english_small\\train\\voice',
    0.97,
    25,
    10,
    512
)

proc_fb = 1
total_fb = len(powframes)
print('')
for pow_sig in powframes:
    print('AUDIO ', proc_fb, '/', total_fb, '...', end='\r', sep='')
    fbanks = compute_filter_banks(pow_sig, 40, 16000, 512)
    mfcc(fbanks, 12)
    proc_fb += 1
