# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:22:09 2020

@author: lasse
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as ss
#import librosa
#import librosa.display


# =============================================================================
# Import of Data
# =============================================================================

filename = 'sound/vokaler.wav'
#y, sr = librosa.load(filename)
## trim silent edges
#whale_song, _ = librosa.effects.trim(y)
##librosa.display.waveplot(whale_song, sr=sr)

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


def spectrogram(x, sampling_frequency, window, nperseg):
    f, t, STFT = ss.stft(x, sampling_frequency, window, nperseg)    
    plt.pcolormesh(t, f, np.abs(STFT), vmin = 0) 
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    cb=plt.colorbar(orientation="horizontal")
    cb.set_label("dB")
    plt.tight_layout()
    plt.show()
    
def spectrogram_lib(data, sr, n_fft=2048, hop_length=512, window='hann'):
    D = np.abs(librosa.stft(data, n_fft=n_fft,  hop_length=hop_length, window=window))
    #f, t, K = ss.stft(data, fs, 'hamming', n_fft)
    #librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
    DB = librosa.amplitude_to_db(D, ref=np.max)
    librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    #plt.savefig('spectrogram/vokaler_{}.pdf'.format(int(n_fft)))
    plt.show()

    

def fir_bandfilter(window, M, fc_low, fc_high, fs):
    cutoff = [fc_low, fc_high]
    bandfilter = ss.firwin(M+1, cutoff, window = window, pass_zero = False, fs = fs)
    return bandfilter


# =============================================================================
# Application of Signal Modification Functions
# =============================================================================
def filtering(x, fir_filter):
    y = np.convolve(x, fir_filter)
    return y

#bandfilter = fir_bandfilter('hamming', 50, 0.766990*2, 1.53398*2, fs = 4*np.pi)


bandfilter = fir_bandfilter('hamming', 50, 500, 1000, fs = 2**13)

#lowpass = ss.firwin(30, fs = 2*np.pi, )

w, h = ss.freqz(bandfilter)

# =============================================================================
# Plotting
# =============================================================================

def zeropad_fft(h, zeros=2**15):
    h_pad = np.zeros(zeros)
    h_pad[0:len(h)] = h
    H_pad = np.abs(np.fft.fft(h_pad))
    H_pad = H_pad[0:int(len(H_pad)/2)]
    return H_pad

H1 = (zeropad_fft(window('boxcar', 50)))
H2 = (zeropad_fft(window('bartlett', 50)))
H3 = (zeropad_fft(window('hann', 50)))
H4 = (zeropad_fft(window('hamming', 50)))
H5 = (zeropad_fft(window('blackman', 50)))

plt.figure(figsize=(8,16))
plt.subplots_adjust(hspace = 0.25)
plt.subplot(511)
plt.plot(np.linspace(0, np.pi, len(H1)), 20*np.log10(np.abs(H1/ abs(H1).max())))
plt.ylabel('Magnitude [dB]')
plt.xlabel('Radian Frequency [$\omega$]')
plt.legend(['Rectangular'])
plt.ylim(-100, 0)
plt.xlim(0, np.pi)
plt.grid()
plt.subplot(512)
plt.plot(np.linspace(0, np.pi, len(H2)), 20*np.log10(np.abs(H2/ abs(H2).max())), 'g')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Radian Frequency [$\omega$]')
plt.legend(['Triangular'])
plt.ylim(-100, 0)
plt.xlim(0, np.pi)
plt.grid()
plt.subplot(513)
plt.plot(np.linspace(0, np.pi, len(H3)), 20*np.log10(np.abs(H3/ abs(H3).max())), 'r')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Radian Frequency [$\omega$]')
plt.legend(['Hann'])
plt.ylim(-100, 0)
plt.xlim(0, np.pi)
plt.grid()
plt.subplot(514)
plt.plot(np.linspace(0, np.pi, len(H4)), 20*np.log10(np.abs(H4/ abs(H4).max())), 'm')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Radian Frequency [$\omega$]')
plt.legend(['Hamming'])
plt.ylim(-100, 0)
plt.xlim(0, np.pi)
plt.grid()
plt.subplot(515)
plt.plot(np.linspace(0, np.pi, len(H5)), 20*np.log10(np.abs(H5/ abs(H5).max())), 'k')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Radian Frequency [$\omega$]')
plt.legend(['Blackman'])
plt.ylim(-100, 0)
plt.xlim(0, np.pi)
plt.grid()
plt.savefig('figures/window_comparison.pdf')
plt.show()



#plt.plot(bandfilter)
#plt.show()
#
#fig = plt.figure()
#plt.title('Digital filter frequency response')
#plt.plot(w, 20 * np.log10(abs(h)), 'b')
#plt.ylabel('Amplitude [dB]', color='b')
#plt.xlabel('Frequency [rad/sample]')
#ax1 = fig.add_subplot(111)
#ax2 = ax1.twinx()
#angles = np.unwrap(np.angle(h))
#plt.plot(w, angles, 'g')
#plt.ylabel('Angle (radians)', color='g')
#plt.grid()
#plt.axis('tight')
#plt.show()
#
#
#fs = 2**13 #4*np.pi 
#
#frq1 = 1500 #2*np.pi/(2**13)*100
#frq2 = 300 #2*np.pi/(2**13)*300
#frq3 = 800 #2*np.pi/(2**13)*500
#frq4 = 1000 #2*np.pi/(2**13)*800
#
#spectrogram_lib(fsinew(fs = fs, freq1 = frq1, freq2 = frq2, freq3 = frq3, freq4 = frq4), fs, n_fft=int(2048/2), hop_length=512, window='hann')
#
#x_filtered = filtering(fsinew(fs=fs, freq1 = frq1, freq2 = frq2, freq3 = frq3, freq4 = frq4), bandfilter)
#
#spectrogram_lib(x_filtered, fs, n_fft=int(2048/2), hop_length=512, window='hann')