# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:37:45 2020

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as plt



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


def H_ideal(fs = 8000, cutoff = 1000):
    ones = np.ones(cutoff)
    zeros = np.zeros(fs-2*cutoff)
    H_ideal = np.concatenate((ones, zeros, ones))
    return H_ideal


def H_sampled(H_ideal, N):
    T = int(len(H_ideal)/(N))
    H_sampled = np.zeros(N)
    H_sampled[0] = H_ideal[0]
    for i in range(1, N):
        H_sampled[i] = H_ideal[i*T]
    return H_sampled


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
    plt.ylim(-100, 10)
    plt.show()
    

H_ideal = H_ideal()
H_sampled = H_sampled(H_ideal, 30)
h = h(H_sampled)


plt.plot(H_ideal, '*')
plt.show()
plt.plot(H_sampled, '*')
plt.show()
plt.plot(h, '*')
plt.show()
plt_magnitude_dB(zeropad_fft(h))