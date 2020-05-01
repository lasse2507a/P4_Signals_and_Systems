# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:22:09 2020

@author: lasse
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as ss


# =============================================================================
# Import of Data
# =============================================================================


# =============================================================================
# Synthetic Data Generation
# =============================================================================
def sinew(J = 18, freq = 1000, phase = 0):
    """
    Signal consisting of a single sine wave with specified 
    frequencies, phases, and amount of points.
    """
    N = 2**J
    t = np.arange(1 , N+1)
    A = 2 * np.pi * t / N
    x = np.sin(A * freq + phase)
    return x


def fsinew(J = 12, sampling_frequency = 2**11, freq1 = 200, freq2 = 400, freq3 = 500, freq4 = 800, 
           phase1 = 0, phase2 = 0, phase3 = 0, phase4 = 0, phase5 = 0):
    """
    Signal consisting of four sine waves with specified 
    frequencies, phases, and amount of points.
    """
    N = 2**J
    t = np.arange(N)/sampling_frequency
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
    elif window_name == 'hamming':
        return ss.windows.hamming(M)
    elif window_name == 'hann':
        return ss.windows.hann(M)
    elif window_name == 'blackman':
        return ss.windows.blackman(M)


def spectrogram(x, sampling_frequency, window, nperseg):
    f, t, STFT = ss.stft(x, sampling_frequency, window, nperseg)
    plt.pcolormesh(t, f, np.abs(STFT)) 
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    

def fir_bandfilter(window, M, fc_low, fc_high, fs):
    cutoff = [fc_low, fc_high]
    bandfilter = ss.firwin(M+1, cutoff, window = window, pass_zero = False, fs = fs)
    return bandfilter


# =============================================================================
# Application of Signal Modification Functions
# =============================================================================
def filtering(x, fir_filter):
    y = np.convolve(x, fir_filter)
    return y

bandfilter = fir_bandfilter('boxcar', 12, 0.766990*2, 1.53398*2, fs = 2*np.pi)

w, h = ss.freqz(bandfilter)

# =============================================================================
# Plotting
# =============================================================================
plt.plot(bandfilter)
plt.show()

fig = plt.figure()
plt.title('Digital filter frequency response')
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Frequency [rad/sample]')
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
angles = np.unwrap(np.angle(h))
plt.plot(w, angles, 'g')
plt.ylabel('Angle (radians)', color='g')
plt.grid()
plt.axis('tight')
plt.show()



spectrogram(fsinew(), 2**11, 'boxcar', 1024)

x_filtered = filtering(fsinew(), bandfilter)

spectrogram(x_filtered, 2**11, 'boxcar', 1024)