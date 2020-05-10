# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:37:45 2020

@author: Jacob
"""
from __future__ import division, print_function

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# =============================================================================
# Synthetic Data Generation
# =============================================================================

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
# Function
# =============================================================================



def plot_frq_response_dB(trans_width, numtaps , fs=8000, cutoff=1000):
    taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs],
                    [1, 0], Hz=fs)
    w, h = signal.freqz(taps, [1], worN=2000)
    plt.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)), \
             label='trans_width={}, N={}'.format(trans_width,numtaps))
    plt.ylim(-40, 5)
    plt.xlim(0, 0.5*fs)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain [dB]')
    plt.grid(True)


def plot_frq_response(trans_width, numtaps , fs=8000, cutoff=1000):
    taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs],
                    [1, 0], Hz=fs)
    w, h = signal.freqz(taps, [1], worN=2000)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), \
             label='trans_width={}, N={}'.format(trans_width,numtaps))
    plt.ylim(-0.1, 1.3)
    plt.xlim(0, 0.5*fs)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain')
    plt.grid(True)


def plot_imp_response(trans_width, numtaps , fs=8000, cutoff=1000):
    taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs],
                    [1, 0], Hz=fs)
    plt.plot(taps, label='trans_width={}, N={}'.format(trans_width,numtaps))
    plt.ylim(-0.1, 1)
    #plt.xlim(0, 0.5*fs)
    plt.xlabel('Sample number')
    plt.ylabel('Amplitude')
    plt.grid(True)

# Low-pass filter design parameters
fs = 8000.0       # Sample rate, Hz
cutoff = 1000.0    # Desired cutoff frequency, Hz
trans_width = 100  # Width of transition from pass band to stop band, Hz
numtaps = 50      # Size of the FIR filter.

plt.figure(figsize=(16,9))
plt.title('Frequency response', fontsize = 20)
plot_imp_response(trans_width=1, numtaps=50)
plot_imp_response(trans_width=100, numtaps=50)
plot_imp_response(trans_width=150, numtaps=50)
plot_imp_response(trans_width=200, numtaps=50)
plt.legend()


plt.figure(figsize=(16,9))
plt.title('Frequency response', fontsize = 20)
plot_frq_response(trans_width=100, numtaps=50)
plot_frq_response(trans_width=100, numtaps=50)
plot_frq_response(trans_width=100, numtaps=50)
plot_frq_response(trans_width=100, numtaps=50)
plt.legend()


plt.figure(figsize=(16,9))
plt.title('Frequency response', fontsize = 20)
plot_frq_response_dB(trans_width=300, numtaps=100)
plot_frq_response_dB(trans_width=100, numtaps=50)
plot_frq_response_dB(trans_width=200, numtaps=50)
plot_frq_response_dB(trans_width=300, numtaps=50)
plt.plot([750], [-1], '*', label = '-1dB at 750Hz', color = 'black')
plt.plot([1000], [-3], '*', label = '-3dB at 1000Hz', color = 'blue')
plt.plot([1500], [-10], '*', label = '-10dB at 1500Hz', color = 'red')
plt.legend()






