# PySpeech
<img align="left" width="150" src="/images/pyspeech_logo.png">
Several signal processing and feature extraction functions for automatic speech to text conversion, i.e Automatic Speech Recognition (ASR). This project aims to be a compilation of speech processing methods to facilitate the development of feature extraction techniques for Speech Recognition (specially for speech-to-text). It is still under testing and development, in the following section you will find issues and basic usage of this library.

## To-do
Notice that this project is still on developemnt, therefore bugs may occur and
changes to code will be frequent.

- [ ] Add other filters
- [ ] Implement the skeleton features in `features.py` module
- [ ] Add testing folder
- [X] Change project structure
- [X] Move signal transformations from `processing.py` module
- [ ] Improve interface
- [X] Fix mfcc log error

## Basic usage
For processing, that is, to apply a preemphasis, split and apply a windowing 
function to the speech signal you can process a single audio

```python
import pyspeech.processing as spproc
windowed_signals = spproc.windowed_signal(
    'C:\\DATASETS\\english_small\\train\\voice',
    0.97,
    25,
    10
)
```

Or a whole dataset by passing the folder full path
```python
import pyspeech.processing as spproc
# To process the whole dataset
windowed_signals = spproc.window_voice_dataset(
    'C:\\DATASETS\\english_small\\train\\voice',
    0.97,
    25,
    10
)
```

After processing the signal, you can then apply several filters from `filters`
module and features from `features.py` modules. As an example:

```python
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
for frames in windowed_signals:
    pow_frames = sptrans.fft(frames, 512)
    fbanks = spfilt.compute_filter_banks(pow_frames, 40, 16000, 512)
    spfeat.mfcc(fbanks, 12)
    proc_fb += 1
```

You can check the whole code in `example.py` file.
