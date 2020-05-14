# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


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


def fsinew(J = 14, fs = 2**14 , freq1 = 0, freq2 = 2000, freq3 = 3000, freq4 = 4000, 
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
# Frequency Lowering
# =============================================================================
def linear_freq_comp(signal, tau):
    signal = np.abs(np.fft.fft(signal))[0:int(len(signal)/2)]
    region_comp = np.zeros(len(signal))
    for i in range(len(signal)):
        region_comp[int(i*tau)] += signal[i]
    signal_comp = np.fft.ifft(region_comp)
    return signal_comp


def nonlinear_freq_comp(signal, fc, tau):
    signal = np.abs(np.fft.fft(signal))[0:int(len(signal)/2)]
    region = signal[fc:]
    region_comp = np.zeros(len(region))
    for i in range(len(region)):
        region_comp[int(((i+fc)**(1/tau))*(fc**(1-1/tau)))] += region[i]
    signal_comp = np.fft.ifft(np.append(signal[0:fc], region_comp))
    return signal_comp


# =============================================================================
# Application of Functions
# =============================================================================
signal = fsinew()
signal_comp = linear_freq_comp(signal, 0.5)
signal_comp2 = nonlinear_freq_comp(signal, 1000, 2)

#plt.figure(figsize=(16,8))
#plt.subplots_adjust(hspace = 0.45)
#plt.subplot(411)
#plt.plot(signal)
#plt.ylabel('Amplitude')
#plt.xlabel('Time [s]')
#plt.subplot(412)
#plt.plot(np.abs(np.fft.fft(signal))[0:int(len(signal)/2)])
#plt.ylabel('Magnitude')
#plt.xlabel('Frequency [Hz]')
#plt.subplot(413)
#plt.plot(signal_comp)
#plt.ylabel('Amplitude')
#plt.xlabel('Time [s]')
#plt.subplot(414)
#plt.plot(np.abs(np.fft.fft(signal_comp))[0:int(len(signal)/2)])
#plt.ylabel('Magnitude')
#plt.xlabel('Frequency [Hz]')
#
#plt.figure(figsize=(16,8))
#plt.subplots_adjust(hspace = 0.45)
#plt.subplot(411)
#plt.plot(signal)
#plt.ylabel('Amplitude')
#plt.xlabel('Time [s]')
#plt.subplot(412)
#plt.plot(np.abs(np.fft.fft(signal))[0:int(len(signal)/2)])
#plt.ylabel('Magnitude')
#plt.xlabel('Frequency [Hz]')
#plt.subplot(413)
#plt.plot(signal_comp2)
#plt.ylabel('Amplitude')
#plt.xlabel('Time [s]')
#plt.subplot(414)
#plt.plot(np.abs(np.fft.fft(signal_comp2))[0:int(len(signal)/2)])
#plt.ylabel('Magnitude')
#plt.xlabel('Frequency [Hz]')

plt.figure(figsize=(10,4))
plt.plot(np.linspace(0,8000), np.linspace(0,int(8000*0.7)), 'b', label = '$\\tau = 0.7$')
plt.plot(np.linspace(0,8000), np.linspace(0,int(8000*0.6)), 'g', label = '$\\tau = 0.6$')
plt.plot(np.linspace(0,8000), np.linspace(0,int(8000*0.5)), 'r', label = '$\\tau = 0.5$')
plt.legend()
plt.ylabel('$f_{out}$ [Hz]')
plt.xlabel('$f_{in}$ [Hz]')
plt.grid()
plt.savefig('figures/linear_frequency_compression.pdf')

plt.figure(figsize=(10,4))
plt.axhline(y=1000, xmax=0.16, color = 'k', linestyle = '--')
plt.axvline(x=1000, ymax=0.27, color = 'k', linestyle = '--')
plt.plot(np.linspace(0,1000), np.linspace(0,1000), 'k')
plt.plot(np.linspace(1000,8000), (np.linspace(1000,8000)**(1/1.5))*(1000**(1-1/1.5)), 'b', label = '$\\tau = 1.5$')
plt.plot(np.linspace(1000,8000), (np.linspace(1000,8000)**(1/2))*(1000**(1-1/2)), 'g', label = '$\\tau = 2$')
plt.plot(np.linspace(1000,8000), (np.linspace(1000,8000)**(1/2.5))*(1000**(1-1/2.5)), 'r', label = '$\\tau = 2.5$')
plt.legend()
plt.ylabel('$f_{out}$ [Hz]')
plt.xlabel('$f_{in}$ [Hz]')
plt.grid()
plt.savefig('figures/nonlinear_frequency_compression.pdf')