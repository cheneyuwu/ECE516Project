import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fftshift



import transforms


def momentum(a, b, c, d):
    return np.sum(np.conj(a) * b * c * d)


def effective_td_bw(tf, signal, d):
    m2 = momentum(signal, tf ** 2, signal, d)
    m1 = momentum(signal, tf, signal, d)
    m0 = momentum(signal, 1, signal, d)
    return 2 * np.pi * np.sqrt(m2 / m0 - (m1 / m0) ** 2)


fs = 1e3
time = np.linspace(-10, 10, int(20 * fs))
sig_time = transforms.wavelet(time, 0, 10, 0.1 * np.sqrt(2) / 3)

freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=1 / fs))
sig_freq = np.fft.fftshift(np.fft.fft(sig_time))

sigma_time = effective_td_bw(time, sig_time, d=1e-1)
sigma_freq = effective_td_bw(freq, sig_freq, d=1e+1)




# plt.plot(freq, sig_freq.imag)


# # plt.plot(time, sig_time.real)
# plt.show()
