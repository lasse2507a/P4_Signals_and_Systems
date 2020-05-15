# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as ss


# =============================================================================
# Synthetic Data Generation
# =============================================================================
def sinew(J = 12, fs = 2**11, freq = 100, phase = 0):
    """
    Signal consisting of a single sine wave with specified 
    frequencies, phases, and amount of points.
    """
    N = 2**J
    t = np.arange(N)/fs
    A = 2 * np.pi * t
    x = np.sin(A * freq + phase)
    return x


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


# =============================================================================
# Signal Modification Functions
# =============================================================================
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


def filtering(x, fir_filter):
    y = np.convolve(x, fir_filter)
    return y


def zeropad_fft(h, zeros=44100):
    h_pad = np.zeros(zeros)
    h_pad[0:len(h)] = h
    H_pad = np.abs(np.fft.fft(h_pad))
    H_pad = H_pad[0:int(len(H_pad)/2)]
    return H_pad


def h_d(N = 87, fc = 4000/44100):
    h_d = np.zeros(N+1)
    for k in range(N+1):
        h_d[k] = (np.sin(fc*(k-N/2)))/(np.pi*(k-N/2))
    return h_d


# =============================================================================
# Plotting
# =============================================================================
h_hann = h_d()*window('hann', 88)
h_hamming = h_d()*window('hamming', 88)
h_blackman = h_d(131)*window('blackman', 132)

H_hann = zeropad_fft(h_hann)
H_hamming = zeropad_fft(h_hamming)
H_blackman = zeropad_fft(h_blackman)

plt.figure(figsize=(14,10))
plt.subplot(211)
plt.plot(h_hann, ':', color = 'r')
plt.plot(h_hamming, ':', color = 'm')
plt.plot(h_blackman, ':', color = 'k')
plt.subplot(212)
plt.plot(H_hann, color = 'r')
plt.plot(H_hamming, color = 'm')
plt.plot(H_blackman,  color = 'k')