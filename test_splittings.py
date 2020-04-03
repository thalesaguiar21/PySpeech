import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from pyspeech.data import oversample



# Create random signals
# Fs = 800 # 800Hz
# amp = 300
# siglengths = np.random.randint(3000, 48000, 200)
# signals = [amp * np.random.rand(length) for length in siglengths]

audiopath = 'tests/voice/OSR_us_000_0011_8k.wav'
freq, signal = wavfile.read(audiopath)
signals = [signal]
print(f"Signal has {signal.size / freq}s and {freq}KHz")
print('Splitting signals...')
#newdata = oversample.split_by_shorter(signals, Fs)
newdata = oversample.by_segment(signals, freq)
print('Finished')
print(f"New size is: {newdata.shape[0]}")
print(f"Length of samples: {newdata.shape[1]/freq}s")

time = np.arange(signal.size ) / freq
plt.plot(time, signal / max(signal))
plt.show()

