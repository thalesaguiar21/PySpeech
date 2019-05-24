# PySpeech
<img align="left" width="150" src="/images/pyspeech_logo.png">
Several signal processing and feature extraction functions for automatic speech to text conversion, i.e Automatic Speech Recognition (ASR). This project aims to be a compilation of speech processing methods to facilitate the development of feature extraction techniques for Speech Recognition (specially for speech-to-text). It is still under testing and development, in the following section you will find issues and basic usage of this library.

## To-do
Notice that this project is still on developemnt, therefore bugs may occur and
changes to code will be frequent.

- [ ] Fix bark filter bank
- [ ] Add PLP features
- [ ] Add testing folder
- [ ] Compute minimum NFFT to better process signal
- [ ] Add better logging messages

## Basic usage
First, following the steps to speech processing, create a Frame and a Processor
to apply the preprocessing stages to the signal

```python
import pyspeech.processing as spproc
frame = spproc.Frame(size=25, stride=10)
proc = spproc.Processor(frame, emph=0.97, nfft=512)
```

Then, pass it to the feature extraction function you want, such as MFCC
```python
import pyspeech.features as spfeat

# For a single audio
spfeat.extract_mfcc(signal, frequency, nfilt=40, processor=proc)

# For several audios
spfeat.extract_mfcc(signals, freqencies, nfilt=40, processor=proc)
```

It is also possible to use reading functions from folder module

```python
import pyspeech.folder as spfoldS

# To read several audio samples
spfold.find_wav_files('folder_path')
```

Check the 'example.py' file for more details.
