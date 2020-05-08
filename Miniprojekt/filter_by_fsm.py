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


def zeropad_fft(h, zeros=2**15):
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
    
points = 15

#type 1
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

#type 2
H_omega0 = H_ideal()
H_omega1 = H(a = 500)
H_omega2 = H(a = 1000)
H_omega3 = H(a = 1500)
H2_k0, x2 = H_sampled2(H_omega0, points)
H2_k1, x2 = H_sampled2(H_omega1, points)
H2_k2, x2 = H_sampled2(H_omega2, points)
H2_k3, x2 = H_sampled2(H_omega3, points)
h0_2 = h(H2_k0)
h1_2 = h(H2_k1)
h2_2 = h(H2_k2)
h3_2 = h(H2_k3)
H2_pad0 = zeropad_fft(h0_2)
H2_pad1 = zeropad_fft(h1_2)
H2_pad2 = zeropad_fft(h2_2)
H2_pad3 = zeropad_fft(h3_2)

#dif points
#5, 15, 25, 35
H_omega0 = H_ideal()
H_omega1 = H(a = 500)
H_omega2 = H(a = 1000)
H_omega3 = H(a = 1500)
H2_k5, x21 = H_sampled2(H_omega2, 5)
H2_k15, x22 = H_sampled2(H_omega2, 15)
H2_k25, x23 = H_sampled2(H_omega2, 25)
H2_k35, x24 = H_sampled2(H_omega2, 35)
h5_2 = h(H2_k5)
h15_2 = h(H2_k15)
h25_2 = h(H2_k25)
h35_2 = h(H2_k35)
H2_pad5 = zeropad_fft(h5_2)
H2_pad15 = zeropad_fft(h15_2)
H2_pad25 = zeropad_fft(h25_2)
H2_pad35 = zeropad_fft(h35_2)

max5 = np.amax(H2_pad5)
max15 = np.amax(H2_pad15)
max25 = np.amax(H2_pad25)
max35 = np.amax(H2_pad35)


#11, 13, 15, 17
H_omega0 = H_ideal()
H_omega1 = H(a = 500)
H_omega2 = H(a = 1000)
H_omega3 = H(a = 1500)
H2_k11, x21 = H_sampled2(H_omega2, 11)
H2_k13, x22 = H_sampled2(H_omega2, 13)
H2_k15, x23 = H_sampled2(H_omega2, 15)
H2_k17, x24 = H_sampled2(H_omega2, 17)
h11_2 = h(H2_k11)
h13_2 = h(H2_k13)
h15_2 = h(H2_k15)
h17_2 = h(H2_k17)
H2_pad11 = zeropad_fft(h11_2)
H2_pad13 = zeropad_fft(h13_2)
H2_pad15 = zeropad_fft(h15_2)
H2_pad17 = zeropad_fft(h17_2)

max11 = np.amax(H2_pad11)
max13 = np.amax(H2_pad13)
max15 = np.amax(H2_pad15)
max17 = np.amax(H2_pad17)


# =============================================================================
# Plotting
# =============================================================================

## H(omega) and H(k)
#
##plt.savefig('figure/ideal_sam_t2.pdf')
#
## Impulse response
#plt.figure(figsize = (16,9))
#plt.title('Impulse Response Coefficients (Type 2)', fontsize = 20)
##plt.subplot(121)
##plt.title('Type 1', fontsize = 15)
##plt.plot(h0)
##plt.plot(h1)
##plt.plot(h2)
##plt.plot(h3)
##plt.plot(h0, '*', label = '0 Hz transition band', color='C0')
##plt.plot(h1, '*', label = '500 Hz transition band', color='C1')
##plt.plot(h2, '*', label = '1000 Hz transition band', color='C2')
##plt.plot(h3, '*', label = '1500 Hz transition band', color='C3')
##plt.xlabel('Sample number')
##plt.ylabel('Amplitude')
##plt.ylim(-0.1, 0.5)
##plt.legend()
##plt.grid()
###plt.subplots_adjust(hspace= 0.3)
##plt.subplot(122)
##plt.title('Type 2', fontsize = 15)
#plt.plot(h0_2)
#plt.plot(h1_2)
#plt.plot(h2_2)
#plt.plot(h3_2)
#plt.plot(h0_2, '*', label = '0 Hz transition band', color='C0')
#plt.plot(h1_2, '*', label = '500 Hz transition band', color='C1')
#plt.plot(h2_2, '*', label = '1000 Hz transition band', color='C2')
#plt.plot(h3_2, '*', label = '1500 Hz transition band', color='C3')
#plt.xlabel('Sample number')
#plt.ylabel('Amplitude')
#plt.ylim(-0.1, 0.5)
#plt.legend()
#plt.grid()
#plt.savefig('figure/impulse.pdf')
#
## Magnitude
#plt.figure(figsize = (16,9))
#plt.suptitle('Magnitude', fontsize = 20)
#plt.subplot(211)
#plt.title('Type 1', fontsize = 15)
#plt.plot(np.linspace(0, 4000, len(H_pad0)), H_pad0, label = 'Magnitude')
#plt.plot(np.linspace(0, 4000, len(H_pad1)), H_pad1, label = 'Magnitude')
#plt.plot(np.linspace(0, 4000, len(H_pad2)), H_pad2, label = 'Magnitude')
#plt.plot(np.linspace(0, 4000, len(H_pad3)), H_pad3, label = 'Magnitude')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Magnitude')
#plt.ylim(-0.1,1.1)
##plt.plot(1000, 1/np.sqrt(2), '*')
#plt.legend()
#plt.grid()
#plt.subplots_adjust(hspace= 0.3)
#plt.subplot(212)
#plt.title('Type 2', fontsize = 15)
#plt.plot(np.linspace(0, 4000, len(H2_pad0)), H2_pad0, label = 'Magnitude')
#plt.plot(np.linspace(0, 4000, len(H2_pad1)), H2_pad1, label = 'Magnitude')
#plt.plot(np.linspace(0, 4000, len(H2_pad2)), H2_pad2, label = 'Magnitude')
#plt.plot(np.linspace(0, 4000, len(H2_pad3)), H2_pad3, label = 'Magnitude')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Magnitude')
#plt.ylim(-0.1,1.1)
##plt.plot(1000, 1/np.sqrt(2), '*')
#plt.legend()
#plt.grid()
##plt.save('figure/magnitude.pdf')
#
## Frequency response
#plt.figure(figsize = (16,9))
#plt.suptitle('Frequency response', fontsize = 20)
#plt.subplot(211)
#plt.title('Type 1', fontsize = 15)
#plt.plot(np.linspace(0, 4000, len(H_pad0)), 20*np.log10(H_pad0), label = 'Frequency response $|H(f)|$')
#plt.plot(np.linspace(0, 4000, len(H_pad1)), 20*np.log10(H_pad1), label = 'Frequency response $|H(f)|$')
#plt.plot(np.linspace(0, 4000, len(H_pad2)), 20*np.log10(H_pad2), label = 'Frequency response $|H(f)|$')
#plt.plot(np.linspace(0, 4000, len(H_pad3)), 20*np.log10(H_pad3), label = 'Frequency response $|H(f)|$')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('$|H(f)|$ [dB]')
#plt.ylim(-100, 10)
#plt.plot([750, 1000, 1500], [-1, -3, -10], '*', label = 'Goals')
#plt.legend()
#plt.grid()
#plt.subplots_adjust(hspace= 0.3)
#plt.subplot(212)
#plt.title('Type 2', fontsize = 15)
#plt.plot(np.linspace(0, 4000, len(H2_pad0)), 20*np.log10(H2_pad0), label = 'Frequency response $|H(f)|$')
#plt.plot(np.linspace(0, 4000, len(H2_pad1)), 20*np.log10(H2_pad1), label = 'Frequency response $|H(f)|$')
#plt.plot(np.linspace(0, 4000, len(H2_pad2)), 20*np.log10(H2_pad2), label = 'Frequency response $|H(f)|$')
#plt.plot(np.linspace(0, 4000, len(H2_pad3)), 20*np.log10(H2_pad3), label = 'Frequency response $|H(f)|$')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('$|H(f)|$ [dB]')
#plt.ylim(-100, 10)
#plt.plot([750, 1000, 1500], [-1, -3, -10], '*', label = 'Goals')
#plt.legend()
#plt.grid()
#plt.save('figure/freq_response.pdf')


plt.figure(figsize = (16,9))
plt.suptitle('Frequency response', fontsize = 20)
plt.subplot(211)
#plt.title('Gain', fontsize = 15)
plt.plot(np.linspace(0, 4000, len(H2_pad0)), H2_pad0, label =  'N = 11')
plt.plot(np.linspace(0, 4000, len(H2_pad1)), H2_pad1, label =  'N = 13')
plt.plot(np.linspace(0, 4000, len(H2_pad2)), H2_pad2, label =  'N = 15')
plt.plot(np.linspace(0, 4000, len(H2_pad3)), H2_pad3, label =  'N = 17')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.ylim(0.9,1.1)
plt.xlim(0, 1000)
#plt.plot(1000, 1/np.sqrt(2), '*')
plt.legend()
plt.grid()
#plt.subplots_adjust(hspace= 0.3)
plt.subplot(212)
#plt.title('Gain in dB', fontsize = 15)
plt.plot(np.linspace(0, 4000, len(H2_pad0)), 20*np.log10(H2_pad11), label =  'N = 11')
plt.plot(np.linspace(0, 4000, len(H2_pad1)), 20*np.log10(H2_pad13), label =  'N = 13')
plt.plot(np.linspace(0, 4000, len(H2_pad2)), 20*np.log10(H2_pad15), label =  'N = 15')
plt.plot(np.linspace(0, 4000, len(H2_pad3)), 20*np.log10(H2_pad17), label =  'N = 17')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain [dB]')
#plt.ylim(-12, 2)
#plt.xlim(500, 2000)
plt.plot([750], [-1], '*', label = '-1dB at 750Hz', color = 'black')
plt.plot([1000], [-3], '*', label = '-3dB at 1000Hz', color = 'blue')
plt.plot([1500], [-10], '*', label = '-10dB at 1500Hz', color = 'red')
plt.legend()
plt.grid()
plt.savefig('figure/freq_response_test.pdf')

print([max5, max15, max25, max35])
print([max11, max13, max15, max17])
db_dis = [20*np.log10(H2_pad11[int((2**15)/8)]), 20*np.log10(H2_pad13[int(2**15/8)]), \
       20*np.log10(H2_pad15[int((2**15)/8)]), 20*np.log10(H2_pad17[int(2**15/8)])]
print([20*np.log10(H2_pad11[int((2**15)/8)]), 20*np.log10(H2_pad13[int(2**15/8)]), \
       20*np.log10(H2_pad15[int((2**15)/8)]), 20*np.log10(H2_pad17[int(2**15/8)])])

    
    
    
    
    
    
plt.figure(figsize = (16,9))
plt.suptitle('Frequency response', fontsize = 20)
plt.subplot(221)
#plt.title('Gain', fontsize = 15)
plt.plot(np.linspace(0, 4000, len(H2_pad5)), H2_pad5, label =  'N = 5')
plt.plot(np.linspace(0, 4000, len(H2_pad15)), H2_pad15, label =  'N = 15')
plt.plot(np.linspace(0, 4000, len(H2_pad25)), H2_pad25, label =  'N = 25')
plt.plot(np.linspace(0, 4000, len(H2_pad35)), H2_pad35, label =  'N = 35')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.ylim(0.9,1.1)
plt.xlim(0, 1000)
#plt.plot(1000, 1/np.sqrt(2), '*')
plt.legend()
plt.grid()
#plt.subplots_adjust(hspace= 0.3)
plt.subplot(222)
#plt.title('Gain in dB', fontsize = 15)
plt.plot(np.linspace(0, 4000, len(H2_pad5)), 20*np.log10(H2_pad5), label =  'N = 5')
plt.plot(np.linspace(0, 4000, len(H2_pad15)), 20*np.log10(H2_pad15), label =  'N = 15')
plt.plot(np.linspace(0, 4000, len(H2_pad25)), 20*np.log10(H2_pad25), label =  'N = 25')
plt.plot(np.linspace(0, 4000, len(H2_pad35)), 20*np.log10(H2_pad35), label =  'N = 35')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain [dB]')
plt.ylim(-12, 2)
plt.xlim(500, 2000)
plt.plot([750], [-1], '*', label = '-1dB at 750Hz', color = 'black')
plt.plot([1000], [-3], '*', label = '-3dB at 1000Hz', color = 'blue')
plt.plot([1500], [-10], '*', label = '-10dB at 1500Hz', color = 'red')
plt.legend()
plt.grid()
plt.subplot(223)
#plt.title('Gain', fontsize = 15)
plt.plot(np.linspace(0, 4000, len(H2_pad11)), H2_pad11, label =  'N = 11')
plt.plot(np.linspace(0, 4000, len(H2_pad13)), H2_pad13, label =  'N = 13')
plt.plot(np.linspace(0, 4000, len(H2_pad15)), H2_pad15, label =  'N = 15')
plt.plot(np.linspace(0, 4000, len(H2_pad17)), H2_pad17, label =  'N = 17')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.ylim(0.9,1.1)
plt.xlim(0, 1000)
#plt.plot(1000, 1/np.sqrt(2), '*')
plt.legend()
plt.grid()
#plt.subplots_adjust(hspace= 0.3)
plt.subplot(224)
#plt.title('Gain in dB', fontsize = 15)
plt.plot(np.linspace(0, 4000, len(H2_pad11)), 20*np.log10(H2_pad11), label =  'N = 11')
plt.plot(np.linspace(0, 4000, len(H2_pad13)), 20*np.log10(H2_pad13), label =  'N = 13')
plt.plot(np.linspace(0, 4000, len(H2_pad15)), 20*np.log10(H2_pad15), label =  'N = 15')
plt.plot(np.linspace(0, 4000, len(H2_pad17)), 20*np.log10(H2_pad17), label =  'N = 17')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain [dB]')
plt.ylim(-12, 2)
plt.xlim(500, 2000)
plt.plot([750], [-1], '*', label = '-1dB at 750Hz', color = 'black')
plt.plot([1000], [-3], '*', label = '-3dB at 1000Hz', color = 'blue')
plt.plot([1500], [-10], '*', label = '-10dB at 1500Hz', color = 'red')
plt.legend()
plt.grid()
plt.savefig('figure/freq_response_test.pdf')