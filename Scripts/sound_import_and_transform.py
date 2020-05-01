# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:40:55 2020

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.io import wavfile

#Navn paa lydklip
filename = "jacob_snak.wav"

#Uddrager samplingsfrekvens og data fra lydfil
fs, data1 = wavfile.read(filename)
time = len(data1)/fs
highest = np.amax(data1)
data = data1/highest

#FIR filter by windowing
fir = sp.signal.firwin(5, 4400, fs = fs)


#Downsampling
down = 5
signal = sp.signal.decimate(data, down, ftype='fir')

#Some windows (find more: https://bit.ly/2KxlMsO)
M = int(20e-3*fs)         #number of samples
hamming = sp.signal.windows.hamming(M)
boxcar = sp.signal.windows.boxcar(M)
flattop = sp.signal.windows.flattop(M)

#Short time Fourier transform
f, t, STFT  = sp.signal.stft(signal, fs/down, window = 'hamming', nperseg=882)

A = STFT

#to plot over time
x_data = np.linspace(0, time, len(data))
x_signal = np.linspace(0, time, len(signal))

#fft=np.fft.fft(signal)

#invert short time Fourier transform
T, ISTFT  = sp.signal.istft(STFT, fs/down, nperseg=256)

ISTFT_del = ISTFT[(ISTFT >= 1) & (ISTFT <= -1)]

B = ISTFT
#sp.io.wavfile.write('modi_sound/jacob_snak_down_stft.wav', int(fs/down), ISTFT)



plt.plot(x_data, data)
plt.show()
plt.plot(x_signal, signal)
plt.show()
plt.pcolormesh(t, f, np.abs(STFT))
plt.show()





