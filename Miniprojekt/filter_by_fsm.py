# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:37:45 2020

@author: Jacob
"""

import numpy as np
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

def H_ideal(fs = 8000, cutoff = 1000):
    ones = np.ones(cutoff)
    zeros = np.zeros(fs-cutoff)
    H_ideal = np.concatenate((ones, zeros))
    return H_ideal


def H(fs = 8000, cutoff = 1000, a = 1000):
    def f(x):
        return -(1)*x + 1
    x = np.linspace(0,1, a)
    y = f(x)
    ones = np.ones(cutoff)
    zeros = np.zeros(fs-cutoff-len(x))
    H = np.concatenate((ones, y, zeros))  
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


def plt_magnitude_dB(H_pad):
    plt.figure()
    plt.plot(np.linspace(0, 4000, len(H_pad)), H_pad)
    plt.show()
    plt.plot(np.linspace(0, 4000, len(H_pad)),20*np.log10(H_pad))
    plt.plot([750, 1000, 1500], [-1, -3, -10], '*')
    plt.ylim(-20, 10)
    plt.show()
    

# =============================================================================
# Using functions
# =============================================================================
    

H_omega = H_ideal(cutoff = 750)
H_k, x = H_sampled(H_omega, 20)
h = h(H_k)
H_pad = zeropad_fft(h)


# =============================================================================
# Plotting
# =============================================================================

# H(omega) and H(k)
plt.figure(figsize=(10,5))
plt.title('Ideal lowpass filter')
plt.plot(H_omega, label ='$|H(\omega)$|')
plt.plot(x, H_k, '*', label = '$|H(k)|$')
plt.xlabel('Frequency [Hz]')
plt.legend()
plt.grid()
plt.savefig('figure/ideal_sam.pdf')

# Impulse response
plt.figure(figsize = (10,5))
plt.title('Impulse response')
plt.plot(h, '*', label = 'Imulse response')
plt.xlabel('Sample number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
#plt.save('figure/impulse.pdf')

# Magnitude
plt.figure(figsize = (10,5))
plt.title('Magnitude')
plt.plot(np.linspace(0, 4000, len(H_pad)), H_pad, label = 'Magnitude')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.plot(1000, 1/np.sqrt(2), '*')
plt.legend()
plt.grid()
#plt.save('figure/magnitude.pdf')

# Frequency response
plt.figure(figsize = (10,5))
plt.title('Frequency response')
plt.plot(np.linspace(0, 4000, len(H_pad)), 20*np.log10(H_pad), label = 'Frequency response $|H(f)|$')
plt.xlabel('Frequency [Hz]')
plt.ylabel('$|H(f)|$ [dB]')
plt.ylim(-100, 10)
plt.plot([750, 1000, 1500], [-1, -3, -10], '*', label = 'Goals')
plt.legend()
plt.grid()
#plt.save('figure/freq_response.pdf')