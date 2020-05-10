# -*- coding: utf-8 -*-
"""
Created on Fri May  8 08:50:10 2020

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

# =============================================================================
# Frequense sampling method
# =============================================================================

def H_ideal(fs = 8000, cutoff = 1000):
    ones = np.ones(cutoff)
    zeros = np.zeros(fs-2*cutoff)
    H_ideal = np.concatenate((ones, zeros, ones))
    return H_ideal


def H(fs = 8000, cutoff = 1000, a = 1000):
    def f(x):
        return -1*x + 1
    def g(x):
        return 1*x
    x = np.linspace(0,1, a)
    y = f(x)
    y2 = g(x)
    ones = np.ones(cutoff)
    zeros = np.zeros(fs-2*cutoff-2*len(x))
    H = np.concatenate((ones, y, zeros, y2, ones))  
    return H



def H_sampled(H_ideal, N):
    T = int(len(H_ideal)/(N))
    H_sampled = np.zeros(N)
    x = np.zeros(N)
    H_sampled[0] = H_ideal[0]
    for i in range(1, N):
        H_sampled[i] = H_ideal[i*T]
        x[i] = i*T
    return H_sampled, x

def H_sampled2(H_ideal, N):
    T = len(H_ideal)/(N)
    H_sampled = np.zeros(N)
    x = np.zeros(N) 
    for i in range(N):
        H_sampled[i] = H_ideal[int(i*T)+int((T*(1/2)))]
        x[i] = int(i*T)+int((T*(1/2)))
    return H_sampled, x


def h(H_sampled):
    N = len(H_sampled)
    if N % 2 == 0:
        upper = int(N/2-1)
    else:
        upper = int((N-1)/2)

    alpha = (N-1)/2
    h = np.zeros(N)
    for n in range(N):
        for k in range(1, upper):
            h[n] += (1/N)*(2*np.abs(H_sampled[k])*np.cos(2*np.pi*k*(n-alpha)/N))
        h[n] = h[n] + H_sampled[0]*(1/N)
    return h

def zeropad_fft(h, zeros=2**13):
    h_pad = np.zeros(zeros)
    h_pad[0:len(h)] = h
    H_pad = np.abs(np.fft.fft(h_pad))
    H_pad = H_pad[0:int(len(H_pad)/2)]
    return H_pad
# =============================================================================
# Fir by window
# =============================================================================

def fir_low(window, M, fc, fs):
    low = ss.firwin(M+1, fc, window = window, fs = fs)
    return low



# =============================================================================
# Comparisson
# =============================================================================

H_omega2 = H(cutoff = 1000 ,a = 1000)
H2_k11, x21 = H_sampled2(H_omega2, 11)
h11_2 = h(H2_k11)
H2_pad11 = zeropad_fft(h11_2)
fir = fir_low('hamming', 12, 1600, 8000)
H_fir = zeropad_fft(fir)
w, h = ss.freqz(fir)


plt.figure(figsize = (16,9))
plt.suptitle('Frequency response', fontsize = 20)
plt.subplot(211)
plt.plot(np.linspace(0, 4000, len(H2_pad11)), H2_pad11, label =  'Freq sam')
plt.plot(np.linspace(0, 4000, len(H_fir)), H_fir, label =  'Fir by win')
#plt.plot(w*4000, np.abs(h), label =  'fir by win')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.ylim(-0.1,1.1)
#plt.xlim(0, 1000)
plt.legend()
plt.grid()
#plt.subplots_adjust(hspace= 0.3)
plt.subplot(212)
plt.plot(np.linspace(0, 4000, len(H2_pad11)), 20*np.log10(H2_pad11), label =  'Freq sam')
plt.plot(np.linspace(0, 4000, len(H_fir)), 20*np.log(H_fir), label =  'Fir by win')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain [dB]')
plt.ylim(-100, 10)
#plt.xlim(500, 2000)
plt.plot([750], [-1], '*', label = '-1dB at 750Hz', color = 'black')
plt.plot([1000], [-3], '*', label = '-3dB at 1000Hz', color = 'blue')
plt.plot([1500], [-10], '*', label = '-10dB at 1500Hz', color = 'red')
plt.legend()
plt.grid()
