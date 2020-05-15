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
    
def fir_bandfilter(window, M, fc_low, fc_high, fs):
    cutoff = [fc_low, fc_high]
    bandfilter = ss.firwin(M+1, cutoff, window = window, pass_zero = False, fs = fs)
    return bandfilter


def filtering(x, fir_filter):
    y = np.convolve(x, fir_filter)
    return y


filename = 'sound/Hey.wav'
y, sr = librosa.load(filename)

data1 = filtering(y, fir_bandfilter('hamming', 50, 200, 4000, sr))

data2 = ss.decimate(data1, 2, ftype = 'fir')


def fsinew(J = 13, fs = 2**13 , freq1 = 3100, freq2 = 1000, freq3 = 3300, freq4 = 500, 
           phase1 = 0, phase2 = 0, phase3 = 0, phase4 = 0, phase5 = 0):
    """
    Signal consisting of four sine waves with specified 
    frequencies, phases, and amount of points.
    """
    N = 2**J
    t = np.arange(N)/fs
    A = 2 * np.pi * t
    x1 = 0.5*np.sin(A * freq1 + phase1)
    x2 = np.sin(A * freq2 + phase2)
    x3 = np.sin(A * freq3 + phase3)
    x4 = np.sin(A * freq4 + phase4)
    x_sum = x1 + x2 + x3 + x4
    return x_sum


def transposition2(data, start_frq, slut_frq, fs):
    
    data_fft = abs(np.fft.fft(data))[0:int(len(data)/2)]
    start_frq = int(start_frq * (len(data)/fs))
    slut_frq = int(slut_frq * (len(data)/fs))
    data_del = data_fft[start_frq : slut_frq]
    max_punkt = np.where(data_del == np.amax(data_del))[0][0] + start_frq
    source_up = max_punkt + int(max_punkt/2)
    source_down = int(max_punkt/2) + int(max_punkt/4)
    octav_source = int(max_punkt/2)
    target_up = start_frq
    target_down = int(start_frq/2)
    data_source = data_fft[source_down : source_up]
    data_target = data_fft[target_down: target_up]
    k=0
    tjek = np.zeros(len(data_target))
    for i in range(len(data_source)):
        if source_down + i - octav_source < target_up and \
        source_down + i - octav_source > target_down:
            data_fft[k + target_down] = data_target[k] + data_source[i]
            tjek[k] = data_source[i]
            k += 1

    return data_fft


def transposition(data, start_frq, fs):
    data_fft = abs(np.fft.fft(data))[0:int(len(data)/2)]
    start_frq = int(start_frq * (len(data)/fs))
    source_up = start_frq*2
    target_down = int(start_frq/2)
    data_source = data_fft[start_frq : source_up]
    data_target = data_fft[target_down: start_frq]
    max_punkt = np.where(data_source == np.amax(data_source))[0][0] + start_frq
    octav_down = int(max_punkt/2)
    k=0
    for i in range(len(data_source)):
        if start_frq + i - octav_down < start_frq and \
        start_frq + i - octav_down > target_down:
            data_fft[k + target_down] = data_target[k] + data_source[i]
            k += 1
    return data_fft


start_frq = 2000
slut_frq = 4000
hej = transposition2(data2, start_frq, slut_frq, sr/2)
data = abs(np.fft.fft(data2))[0:int(len(data2)/2)]


plt.plot(np.linspace(0,sr/4, len(hej)), abs(hej))
plt.plot(np.linspace(0,sr/4, len(data)), abs(data))



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

