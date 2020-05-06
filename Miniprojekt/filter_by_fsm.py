# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:37:45 2020

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as ss



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


def H_ideal(fs = 18000, cutoff = 5000):
    ones = np.ones(cutoff)
    zeros = np.zeros(fs-cutoff-cutoff)
    data = np.concatenate((ones, zeros, ones))
    return data#[0:int(len(data)/2)]


def H_sampled(H_ideal, N):    
    T = int(len(H_ideal)/(N))
    sample = np.zeros(N)
    sample[0] = H_ideal[0]
    for i in range(1, N):
        sample[i] = H_ideal[i*T]
    return sample


def h(H_sampled):
    N = len(H_sampled)
    if N % 2 == 0:
        upper = int(N/2-1)
    else:
        upper = int((N-1)/2)
        
    alpha = (N-1)/2
    h = np.zeros(N)
    for n in range(N):
        for i in range(upper):
            h[n] += 2*abs(H_sampled[i])*np.cos(2*np.pi*i*(n-alpha)/N)+H_sampled[0]
    return (1/N)*h


def zeropad_fft(h, zeros=2**10):
    h_pad = np.zeros(zeros)
    h_pad[0:len(h)] = h
    H_pad = np.abs(np.fft.fft(h_pad))
    return H_pad[0:int(len(H_pad)/2)]


def plt_dB(H_pad):
    plt.figure()
    plt.plot(H_pad)
    plt.show()
    plt.plot(np.linspace(0,9000, len(H_pad)),20*np.log10(H_pad))
    plt.ylim(-100, 100)
    plt.show()
    

H_ideal = H_ideal()
H_sampled = H_sampled(H_ideal, 9)
h_n = h(H_sampled)

plt.plot(H_ideal, '*')
plt.show()
plt.plot(H_sampled, '*')
plt.show()
plt.plot(h_n, '*')
plt.show()
plt_dB(zeropad_fft(h_n))


