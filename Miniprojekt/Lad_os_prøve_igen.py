# -*- coding: utf-8 -*-
"""
Created on Sun May 10 09:45:38 2020

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Function
# =============================================================================

#ideal lowpass filter
def H_ideal(fs = 8000, cutoff = 1000):
    ones = np.ones(cutoff)
    zeros = np.zeros(fs-2*cutoff)
    H_ideal = np.concatenate((ones, zeros, ones))
    return H_ideal

#Sampling 
def H_sampled(H_ideal, N):
    T = int(len(H_ideal)/(N))
    H_sampled = np.zeros(N)
    x = np.zeros(N)
    for i in range(N):
        H_sampled[i] = H_ideal[i*T]
        x[i] = i*T
    return H_sampled #, x

#Sampling type 2
def H_sampled2(H_ideal, N):
    T = len(H_ideal)/(N)
    H_sampled = np.zeros(N)
    x = np.zeros(N) 
    for i in range(N):
        H_sampled[i] = H_ideal[int(i*T)+int((T*(1/2)))]
        x[i] = int(i*T)+int((T*(1/2)))
    return H_sampled, x

#rework type 2 
def rework(H_sample, N, M):
    matrix_16 = np.array([[0.40397949, 0, 0], [0.62291631, 0.12384644, 0], \
                          [0.70432347, 0.22385191, 0.01951294], [0,0,0]])
    matrix_32 = np.array([[0.38925171, 0, 0], [2, 3, 0], \
                          [0.73350248, 0.26135787, 0.02770996], [0,0,0]])
    matrix_64 = np.array([[0, 0, 0], [0, 0, 0], \
                          [0.74181077, 0.27213724, 0.03010864], [0,0,0]])
    matrix_128 = np.array([[0, 0, 0], [0, 0, 0], \
                          [0.72460093, 0.25347440, 0.02633057], [0,0,0]])
    matrix_15 = np.array([[0.41793823, 0, 0], [0.59357118, 0.10319824, 0], \
                          [0.65951526, 0.17360713, 0.01000977], [0,0,0]])
    matrix_33 = np.array([[0,39641724, 0, 0], [2, 3, 0], \
                          [0.70374222, 0,22577646, 0.01990967], [0,0,0]])
    matrix_65 = np.array([[2, 0, 0], [2, 3, 0], [2, 3, 4], [0,0,0]])
    where = np.where(H_sample == 0)[0][0]
    if N == 16 or N == 32 or N == 64 or N == 128:
        if N == 16:
            for i in range(3):
                H_sample[where + i] = matrix_16[M-1, i]
                H_sample[-(where + i + 1)] = matrix_16[M-1, i]
        elif N == 32:
            for i in range(3):
                H_sample[where + i] = matrix_32[M-1, i]
                H_sample[-(where + i + 1)] = matrix_32[M-1, i]
        elif N == 64:
            for i in range(3):
                H_sample[where + i] = matrix_64[M-1, i]
                H_sample[-(where + i + 1)] = matrix_64[M-1, i]
        elif N == 128:
            for i in range(3):
                H_sample[where + i] = matrix_128[M-1, i]
                H_sample[-(where + i + 1)] = matrix_128[M-1, i]           
    
    elif N == 15 or N == 33 or N == 65:
        if N == 15:
            for i in range(3):
                H_sample[where + i] = matrix_15[M-1, i]
                H_sample[-(where + i + 1)] = matrix_16[M-1, i]
        elif N == 33:
            for i in range(3):
                H_sample[where + i] = matrix_33[M-1, i]
                H_sample[-(where + i + 1)] = matrix_32[M-1, i]
        elif N == 65:
            for i in range(3):
                H_sample[where + i] = matrix_65[M-1, i]
                H_sample[-(where + i + 1)] = matrix_64[M-1, i]
    
    return H_sample

#rework type 2 
def rework2(H_sample, N, M):
    matrix_16 = np.array([[0.32149048, 0, 0], [0.4936921, 0.07175293, 0], \
                          [0.54899404, 0.11504207, 0.00474243], [0,0,0]])
    matrix_32 = np.array([[0.34217529, 0, 0], [0, 0, 0], \
                          [0.66114353, 0.20058013, 0.01828613], [0,0,0]])
    matrix_64 = np.array([[2, 0, 0], [2, 3, 0], [2, 3, 4], [0,0,0]])
    where = np.where(H_sample == 0)[0][0]
    if N == 16:
        for i in range(3):
            H_sample[where + i] = matrix_16[M-1, i]
            H_sample[-(where + i + 1)] = matrix_16[M-1, i]
    elif N == 32:
        for i in range(3):
            H_sample[where + i] = matrix_32[M-1, i]
            H_sample[-(where + i + 1)] = matrix_32[M-1, i]
    elif N == 64:
        for i in range(3):
            H_sample[where + i] = matrix_64[M-1, i]
            H_sample[-(where + i + 1)] = matrix_64[M-1, i]
    return H_sample


#computing the impulse response
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

#zeropadding and fft of impuls response
def zeropad_fft(h, zeros=2**15):
    h_pad = np.zeros(zeros)
    h_pad[0:len(h)] = h
    H_pad = np.abs(np.fft.fft(h_pad))
    H_pad = H_pad[0:int(len(H_pad)/2)]
    return H_pad


# =============================================================================
# Plot funktioner
# =============================================================================
def plot_reworked(Tn, N = 16 , fs=8000, cutoff=1000):
    H_pad = rework(H_sampled(H_ideal(fs, cutoff), N = N), N=N, M = Tn)                       
    plt.plot(np.linspace(0, fs, len(H_pad)), (H_pad), '*', \
             label = 'M = {}, N = {}'.format(Tn, N))
    plt.ylim(-0.1, 1.1)
    #plt.xlim(0, 0.5*fs)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain')
    plt.grid(True)
    
def plot_imp_response(M = 0, N = 16 , fs=8000, cutoff=1000, color = 'C1'):
    H_pad = h(rework(H_sampled(H_ideal(fs, cutoff), N = N), N=N, M = M))                       
    plt.plot(H_pad, '*', label = 'M = {}, N = {}'.format(M, N), color = color)
    plt.plot(H_pad, color = color)
    plt.ylim(-0.1, 0.5)
    #plt.xlim(0, 0.5*fs)
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.grid(True)

def plot_frq_response(M = 0, N = 16 , fs=8000, cutoff=1000,  label = 'label'):
    H_pad = zeropad_fft(h(rework(H_sampled(H_ideal(fs, cutoff), N = N), N = N, M = M)))                       
    plt.plot(np.linspace(0, 4000, len(H_pad)), np.abs(H_pad), \
             label = label)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain')
    plt.grid(True)

def plot_frq_response2(M = 0, N = 16 , fs=8000, cutoff=1000,  label = 'label'):
    H_pad = zeropad_fft(h(rework2(H_sampled(H_ideal(fs, cutoff), N = N), N = N, M = M)))                       
    plt.plot(np.linspace(0, 4000, len(H_pad)), np.abs(H_pad), \
             label = label)
    plt.ylim(0, 1.2)
    plt.xlim(0, 0.5*fs)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain')
    plt.grid(True)

def plot_frq_response_dB(M = 0, N = 16 , fs=8000, cutoff=1000,  label = 'label'):
    H_pad = zeropad_fft(h(rework(H_sampled(H_ideal(fs, cutoff), N = N), N=N, M = M)))                       
    plt.plot(np.linspace(0, 4000, len(H_pad)), 20*np.log10(np.abs(H_pad)), \
             label = label)
    plt.ylim(-100, 10)
    plt.xlim(0, 0.5*fs)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain [dB]')
    plt.grid(True)

def plot_frq_response_dB2(M = 0, N = 16 , fs=8000, cutoff=1000, label = 'label'):
    H_pad = zeropad_fft(h(rework2(H_sampled(H_ideal(fs, cutoff), N = N), N=N, M = M)))                       
    plt.plot(np.linspace(0, 4000, len(H_pad)), 20*np.log10(np.abs(H_pad)), \
             label = label)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain [dB]')
    plt.grid(True)

# =============================================================================
# plotting
# =============================================================================

H = H_ideal(8000, 1000)
N = 16
N1 = 32
N2 = 64
N3 = 128
M = 3
M1 = 3
M2 = 3
M3 = 3
H_sam =  np.amax(zeropad_fft(h(rework(H_sampled(H_ideal(8000, 1000), N = N), N = N, M = M))))
H_sam1 = np.amax(zeropad_fft(h(rework(H_sampled(H_ideal(8000, 1000), N = N1), N = N1, M = M))))
H_sam2 = np.amax(zeropad_fft(h(rework(H_sampled(H_ideal(8000, 1000), N = N2), N = N2, M = M))))
H_sam3 = np.amax(zeropad_fft(h(rework(H_sampled(H_ideal(8000, 1000), N = N3), N = N3, M = M))))
print([H_sam, H_sam1, H_sam2, H_sam3])
plt.figure(figsize = (10,5))
plt.plot(H_ideal(), label = 'Ideal filter')
plot_reworked(0)
plot_reworked(1)
plot_reworked(2)
plot_reworked(3)
plt.legend()
plt.savefig('figure/ideal_sam.pdf')

plt.figure(figsize=(10,5))
plt.title('Impulse Response' , fontsize = 20)
plot_imp_response(M = 0, N = 16, color = 'C0')
plot_imp_response(M = 1, N = 16, color = 'C1')
plot_imp_response(M = 2, N = 16, color = 'C2')
plot_imp_response(M = 3, N = 16, color = 'C3')
plt.legend()
plt.savefig('figure/impulse.pdf')

N = 16
N1 = 32
N2 = 64
N3 = 128
M = 3
M1 = 3
M2 = 3
M3 = 3
plt.figure(figsize = (16,9))
plt.suptitle('Frequency Response', fontsize = 20)
plt.subplot(221)
plot_frq_response(M = M, N = N, label = 'N = {}, M = {}'.format(N, M))
#plt.plot(1000, 1/np.sqrt(2), '*')
plot_frq_response(M = M1, N = N1, label = 'N = {}, M = {}'.format(N1, M1))
plot_frq_response(M = M2, N = N2, label = 'N = {}, M = {}'.format(N2, M2))
plot_frq_response(M = M3, N = N3, label = 'N = {}, M = {}'.format(N, M3))
plt.ylim(0, 1.2)
plt.xlim(0, 0.5*8000)
plt.legend()

plt.subplot(222)
plot_frq_response_dB(M = M, N = N, label = 'N = {}, M = {}'.format(N, M))
plot_frq_response_dB(M = M1, N = N1, label = 'N = {}, M = {}'.format(N1, M1))
plot_frq_response_dB(M = M2, N = N2, label = 'N = {}, M = {}'.format(N2, M2))
plot_frq_response_dB(M = M3, N = N3, label = 'N = {}, M = {}'.format(N3, M3))
plt.plot([750], [-1], '*', label = '-1dB at 750Hz', color = 'black')
plt.plot([1000], [-3], '*', label = '-3dB at 1000Hz', color = 'blue')
plt.plot([1500], [-10], '*', label = '-10dB at 1500Hz', color = 'red')
plt.ylim(-100, 10)
plt.xlim(0, 0.5*8000)
plt.legend()
#plt.savefig('figure/frq_response_sam.pdf')

plt.subplot(223)
plot_frq_response(M = M, N = N, label = 'N = {}, M = {}'.format(N, M))
#plt.plot(1000, 1/np.sqrt(2), '*')
plot_frq_response(M = M1, N = N1, label = 'N = {}, M = {}'.format(N1, M1))
plot_frq_response(M = M2, N = N2, label = 'N = {}, M = {}'.format(N2, M2))
plot_frq_response(M = M3, N = N3, label = 'N = {}, M = {}'.format(N3, M3))
plt.ylim(0.8, 1.2)
plt.xlim(0, 1200)
plt.legend()

plt.subplot(224)
plot_frq_response_dB(M = M, N = N, label = 'N = {}, M = {}'.format(N, M))
plot_frq_response_dB(M = M1, N = N1, label = 'N = {}, M = {}'.format(N1, M1))
plot_frq_response_dB(M = M2, N = N2, label = 'N = {}, M = {}'.format(N2, M2))
plot_frq_response_dB(M = M3, N = N3, label = 'N = {}, M = {}'.format(N3, M2))
plt.plot([750], [-1], '*', label = '-1dB at 750Hz', color = 'black')
plt.plot([1000], [-3], '*', label = '-3dB at 1000Hz', color = 'blue')
plt.plot([1500], [-10], '*', label = '-10dB at 1500Hz', color = 'red')
plt.legend()
plt.ylim(-20, 2)
plt.xlim(0, 1700)
plt.savefig('figure/frq_response_sam_M3.pdf')