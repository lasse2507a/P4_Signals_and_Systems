# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 08:41:19 2020

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


#Signals
def sinew(J = 18, freq = 1000, phase = 0):
    """
    Signal consisting of a single sine wave with specified 
    frequencies, phases, and amount of points.
    """
    N = 2**J
    t = np.arange(N)/fs
    A = 2 * np.pi * t
    y = np.sin(A * freq + phase)
    return y, t, N


def fsinew(J = 18, freq1 = 1300, freq2 = 2000, freq3 = 400, freq4 = 1000, 
           phase1 = 0, phase2 = 0, phase3 = 0, phase4 = 0, phase5 = 0):
    """
    Signal consisting of four sine waves with specified 
    frequencies, phases, and amount of points.
    """
    N = 2**J
    t = np.arange(N)/fs
    A = 2 * np.pi * t
    x1 = np.sin(A * freq1 + phase1)
    x2 = np.sin(A * freq2 + phase2)
    x3 = np.sin(A * freq3 + phase3)
    x4 = np.sin(A * freq4 + phase4)
    x_sum = x1 + x2 + x3 + x4
    return x_sum, t, N

#Some windows (find more: https://bit.ly/2KxlMsO)
M = 100
hamming = sp.signal.windows.hamming(M)
boxcar = sp.signal.windows.boxcar(M)
flattop = sp.signal.windows.flattop(M)

#Sample frequency
fs = 10e3       

#The pure signal
y, time, N = fsinew()

#Adding white noise
noise = np.random.normal(0, 1, int(N)) 
y_noise = y + noise

#STFT
f, t, STFT = sp.signal.stft(y_noise, fs = fs, window='hamming', nperseg=1000)

#Plotting
plt.pcolormesh(t, f, np.abs(STFT)) 
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.show()

   



