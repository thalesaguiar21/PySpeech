# The number of points for DFT
nfft = 512

# Adds energy to feature vector
append_energy = True

# Frame size and striding in msec
framing = {
  'size': 25,
  'stride': 10
}

# Cutoff freqeuncy for high pass filter design when removing silence
fir = {
    'fc': 0.02,
    'len': 0.08,
}

