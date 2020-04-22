# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:40:55 2020

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

#Navn paa lydklip
filename = "sound/jacob_snak.wav"

#Uddrager samplingsfrekvens og data fra lydfil
fs, data = sp.io.wavfile.read(filename)
time = len(data)/fs
#Downsampling
down = 10
signal = sp.signal.decimate(data, down)

#Short time Fourier transform
f, t, STFT  = sp.signal.stft(signal, fs/down, nperseg=1000)

#to plot over time
x_data = np.linspace(0, time, len(data))
x_signal = np.linspace(0, time, len(signal))

fft=np.fft.fft(signal)


plt.plot(x_data, data)
plt.show()
plt.plot(x_signal, signal)
plt.show()
plt.pcolormesh(t, f, np.abs(STFT))
plt.show()
plt.plot(np.abs(fft))

