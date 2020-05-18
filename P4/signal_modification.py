# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as ss
import librosa
import librosa.display



# =============================================================================
# Import of Data
# =============================================================================
filename = 'sound/jacob_snak.wav'
y, sr = librosa.load(filename, sr = None)


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


def zeropad_fft(h, zeros=2**15):
    h_pad = np.zeros(zeros)
    h_pad[0:len(h)] = h
    H_pad = np.abs(np.fft.fft(h_pad))
    H_pad = H_pad[0:int(len(H_pad)/2)]
    return H_pad


def filtering(x, fir_filter):
    y = np.convolve(x, fir_filter)
    return y


def transposition(data, start_frq, fs):
    #D = np.abs(librosa.stft(data, n_fft=n_fft,  hop_length=512, window=window))
    data_fft = data# abs(np.fft.fft(data))[0:int(len(data)/2)]
    start_frq = int(start_frq * (len(data)/fs))
    source_up = start_frq*2
    target_down = int(start_frq/2)
    data_source = data_fft[start_frq : source_up]
    data_target = data_fft[target_down: start_frq]
    max_punkt = np.where(data_source == np.amax(data_source))[0][0] + start_frq
    octav_down = int(max_punkt/2)
    k=0
    for i in range(len(data_source)):
        if start_frq + i - octav_down < start_frq and \
        start_frq + i - octav_down > target_down:
            data_fft[k + target_down] = data_target[k] + data_source[i]
            k += 1
    return data_fft


def linear_freq_comp(signal, tau):
    signal = np.abs(np.fft.fft(signal))[0:int(len(signal)/2)]
    region_comp = np.zeros(len(signal))
    for i in range(len(signal)):
        region_comp[int(i*tau)] += signal[i]
    signal_comp = np.fft.ifft(region_comp)
    return signal_comp


def nonlinear_freq_comp(signal, fc, tau):
    signal = np.abs(np.fft.fft(signal))[0:int(len(signal)/2)]
    signal_comp = np.zeros(len(signal))
    for i in range(len(signal)-fc-1):
        signal_comp[int(((i+fc+1)**(1/tau))*(fc**(1-1/tau)))] += signal[i+fc+1]
    signal_comp = np.fft.ifft(np.append(signal[0:fc], signal_comp[fc:]))
    return signal_comp


# =============================================================================
# Plotting
# =============================================================================

def transposition_short(data, start_frq, fs, f):
    #D = np.abs(librosa.stft(data, n_fft=n_fft,  hop_length=512, window=window))
    data_fft = data# abs(np.fft.fft(data))[0:int(len(data)/2)]
    start_frq = int(start_frq/(fs/2/len(f)))
    source_up = start_frq*2
    target_down = int(start_frq/2)
    for n in range(len(data_fft[:,0])):
        data_source = data_fft[start_frq : source_up, n]
        data_target = data_fft[target_down: start_frq, n]
        max_punkt = np.where(data_source == np.amax(data_source))[0][0] + start_frq
        octav_down = int(max_punkt/2)
        k=0
        for i in range(len(data_source)):
            if start_frq + i - octav_down < start_frq and \
            start_frq + i - octav_down > target_down:
                data_fft[k + target_down, n] = data_target[k] + data_source[i]
                k += 1
        return data_fft

data = y
fs = sr
down_with = 5
data_filtered = filtering(data, fir_bandfilter('hamming', 50, 200, 4410, fs))
data_down = ss.decimate(data, down_with)
window_length = 20e-3 #s
number_samp = int(fs/down_with*(window_length))
D = np.abs(librosa.core.stft(data, n_fft=number_samp,  hop_length=512, window='hamming'))

f, t, F = ss.stft(data_down, sr/5, 'hamming', nperseg = number_samp, noverlap = 0)


h = np.zeros(len(F[:,0]), dtype = complex)
for i in range(len(F[:,0])):
    h += F[:,i] 

fft = np.fft.fft(data_down)[0:int(len(data_down)/2)]
plt.plot(np.linspace(0,sr/5/2,int(len(data_down)/2)),np.abs(fft))
plt.show()
plt.plot(f,np.abs(h))
plt.show()
plt.plot(f,np.abs(F[:,160]), color = 'C0')

trans = transposition_short(F, 2000, fs/down_with, f)

t1, Fi = ss.istft(trans, sr/5,'hamming')

librosa.output.write_wav('sound/ny_lyd.wav', Fi, int(sr/5))
