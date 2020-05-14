import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as ss
import librosa
import librosa.display


# =============================================================================
# Import of Data
# =============================================================================
filename = 'sound/vokaler.wav'
y, sr = librosa.load(filename)


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
# Plotting
# =============================================================================

