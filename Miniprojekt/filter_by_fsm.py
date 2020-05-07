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
    
points = 20

H_omega0 = H_ideal()
H_omega1 = H(a = 500)
H_omega2 = H(a = 1000)
H_omega3 = H(a = 1500)
H_k0, x = H_sampled(H_omega0, points)
H_k1, x = H_sampled(H_omega1, points)
H_k2, x = H_sampled(H_omega2, points)
H_k3, x = H_sampled(H_omega3, points)
h0 = h(H_k0)
h1 = h(H_k1)
h2 = h(H_k2)
h3 = h(H_k3)
H_pad0 = zeropad_fft(h0)
H_pad1 = zeropad_fft(h1)
H_pad2 = zeropad_fft(h2)
H_pad3 = zeropad_fft(h3)


# =============================================================================
# Plotting
# =============================================================================

# H(omega) and H(k)
plt.figure(figsize=(10,5))
plt.title('Ideal lowpass filter', fontsize = 20)
plt.plot(H_omega0, label ='$|H(\omega)$|')
plt.plot(x, H_k0, '*', label = '$|H(k)|$')
plt.plot(H_omega1, label ='$|H(\omega)$|')
plt.plot(x, H_k1, '*', label = '$|H(k)|$')
plt.plot(H_omega2, label ='$|H(\omega)$|')
plt.plot(x, H_k2, '*', label = '$|H(k)|$')
plt.plot(H_omega3, label ='$|H(\omega)$|')
plt.plot(x, H_k3, '*', label = '$|H(k)|$')
plt.xlabel('Frequency [Hz]')
plt.legend()
plt.grid()
#   plt.savefig('figure/ideal_sam.pdf')

# Impulse response
plt.figure(figsize = (10,5))
plt.title('Impulse response', fontsize = 20)
plt.plot(h0, '*', label = 'Imulse response')
plt.plot(h1, '*', label = 'Imulse response')
plt.plot(h2, '*', label = 'Imulse response')
plt.plot(h3, '*', label = 'Imulse response')
plt.xlabel('Sample number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
#plt.save('figure/impulse.pdf')

# Magnitude
plt.figure(figsize = (10,5))
plt.title('Magnitude', fontsize = 20)
plt.plot(np.linspace(0, 4000, len(H_pad0)), H_pad0, label = 'Magnitude')
plt.plot(np.linspace(0, 4000, len(H_pad1)), H_pad1, label = 'Magnitude')
plt.plot(np.linspace(0, 4000, len(H_pad2)), H_pad2, label = 'Magnitude')
plt.plot(np.linspace(0, 4000, len(H_pad3)), H_pad3, label = 'Magnitude')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.plot(1000, 1/np.sqrt(2), '*')
plt.legend()
plt.grid()
#plt.save('figure/magnitude.pdf')

# Frequency response
plt.figure(figsize = (10,5))
plt.title('Frequency response', fontsize = 20)
plt.plot(np.linspace(0, 4000, len(H_pad0)), 20*np.log10(H_pad0), label = 'Frequency response $|H(f)|$')
plt.plot(np.linspace(0, 4000, len(H_pad1)), 20*np.log10(H_pad1), label = 'Frequency response $|H(f)|$')
plt.plot(np.linspace(0, 4000, len(H_pad2)), 20*np.log10(H_pad2), label = 'Frequency response $|H(f)|$')
plt.plot(np.linspace(0, 4000, len(H_pad3)), 20*np.log10(H_pad3), label = 'Frequency response $|H(f)|$')
plt.xlabel('Frequency [Hz]')
plt.ylabel('$|H(f)|$ [dB]')
plt.ylim(-100, 10)
plt.plot([750, 1000, 1500], [-1, -3, -10], '*', label = 'Goals')
plt.legend()
plt.grid()
#plt.save('figure/freq_response.pdf')