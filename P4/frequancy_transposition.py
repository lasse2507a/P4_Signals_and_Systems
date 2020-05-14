# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:12:30 2020

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as ss
import librosa
import librosa.display

filename = 'sound/jacob_snak.wav'
y, sr = librosa.load(filename)

def transposition(data, start_frq):
    data_fft = np.fft.fft(data)
    return data_fft

hej = transposition(y, 2000)

w, h = ss.freqz(hej)

plt.plot(y)
plt.show()
plt.plot(w, 20 * np.log10(abs(h)))