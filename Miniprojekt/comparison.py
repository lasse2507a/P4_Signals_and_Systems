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

# =============================================================================
# Fir by window
# =============================================================================

def fir_low(window, M, fc, fs):
    low = ss.firwin(M+1, fc, window = window, pass_zero = False, fs = fs)
    return low

