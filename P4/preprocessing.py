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


def zeropad_fft(h, zeros=2**13):
    h_pad = np.zeros(zeros)
    h_pad[0:len(h)] = h
    H_pad = np.abs(np.fft.fft(h_pad))
    H_pad = H_pad[0:int(len(H_pad)/2)]
    return H_pad


# =============================================================================
# Plotting
# =============================================================================
h_hann = ss.firwin(192, 4000, window = 'hann', fs = 48000)
h_hamming = ss.firwin(192, 4000, window = 'hamming', fs = 48000)

H_hann = zeropad_fft(h_hann)
H_hamming = zeropad_fft(h_hamming)

plt.figure(figsize=(12,4))
plt.plot(h_hann, ':', color = 'r', label = 'Hann, N = 191')
plt.plot(h_hamming, ':', color = 'b', label = 'Hamming, N = 191')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.grid()
plt.savefig('figures/preprocessing_filter_coefficients.pdf')

plt.figure(figsize=(12,10))
plt.subplot(311)
plt.plot(np.linspace(0,24000,len(H_hann)), H_hann, color = 'r', label = 'Hann, N = 191')
plt.plot(np.linspace(0,24000,len(H_hamming)), H_hamming, color = 'b', label = 'Hamming, N = 191')
plt.legend()
plt.xlim(0, 24000)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid()
plt.subplot(312)
plt.plot(np.linspace(0,24000,len(H_hann)), H_hann, color = 'r', label = 'Hann, N = 191')
plt.plot(np.linspace(0,24000,len(H_hamming)), H_hamming, color = 'b', label = 'Hamming, N = 191')
plt.legend()
plt.xlim(0, 5000)
plt.ylim(0.95,1.05)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid()
plt.subplot(313)
plt.plot(np.linspace(0, 24000, len(H_hann)), 20*np.log10(H_hann), color = 'r', label = 'Hann, N = 191')
plt.plot(np.linspace(0, 24000, len(H_hamming)), 20*np.log10(H_hamming), color = 'b', label = 'Hamming, N = 191')
plt.legend()
plt.ylim(-100,10)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.savefig('figures/preprocessing_magnitude_dB.pdf')