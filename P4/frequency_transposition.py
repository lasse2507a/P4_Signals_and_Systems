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


def window(window_name, M):
    if window_name == 'boxcar':
        return ss.windows.boxcar(M)
    elif window_name == 'bartlett':
        return ss.windows.bartlett(M)
    elif window_name == 'hamming':
        return ss.windows.hamming(M)
    elif window_name == 'hann':
        return ss.windows.hann(M)
    elif window_name == 'blackman':
        return ss.windows.blackman(M)

filename = 'sound/jacob_snak.wav'
y, sr = librosa.load(filename)


def fsinew(J = 18, fs = 2**12 , freq1 = 200, freq2 = 400, freq3 = 500, freq4 = 800, 
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
    return x_sum



def transposition(data, start_frq):
    data_fft = np.fft.fft(data)
    return data_fft


hej = transposition(fsinew(), 600)

plt.plot(hej)


#def fir_bandfilter(window, M, fc_low, fc_high, fs):
#    cutoff = [fc_low, fc_high]
#    bandfilter = ss.firwin(M+1, cutoff, window = window, pass_zero = False, fs = fs)
#    return bandfilter
#
#
#def transposition(data, start_frq):
#    data_fft = np.fft.fft(data)
#    return data_fft
#
#
#fir_filter = fir_bandfilter('hamming', 200, 200, 6000, sr)
#fir_filter = np.fft.fft(fir_filter, 2**11)
#data = np.convolve(fir_filter, y)
#
#data1 = ss.decimate(y, 5)
#
#win = np.fft.fft(window('hamming', 50), 2**11)
#
#
#hej = transposition(y, 2000)
#hej1 = transposition(data, 2000)
#
#plt.plot(20*np.log10(abs(hej1[0:int(len(hej1)/2)])))
#plt.show()
#plt.plot(abs(hej1[0:int(len(hej1)/2)]))
#plt.show()
#plt.plot(np.linspace(0,sr/2,(int(len(fir_filter)/2))), \
#                     20*np.log10(abs(fir_filter[0:int(len(fir_filter)/2)])))

