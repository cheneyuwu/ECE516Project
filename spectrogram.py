import numpy as np
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt

import transforms

fs = 1e4
N = 1e4

time = np.arange(N) / float(fs)

# amp = 2 * np.sqrt(2)
# noise_power = 0.01 * fs / 2


# mod = 500 * np.cos(2 * np.pi * 0.25 * time)
# carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)

# noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
# noise *= np.exp(-time / 5)

# x = carrier + noise
# x = carrier
# x = amp * np.cos(2*np.pi*3e3*time) + 1j * amp * np.sin(2*np.pi*3e3*time)

# x = transforms.wavelet(time, 0.5, 0, 0.1)
# x = transforms.wave(time, 0, 0)
x = np.ones_like(time)

plt.plot(x.real)
plt.show()

f, t, Sxx = signal.spectrogram(x, fs, return_onesided=False)
plt.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0))
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.show()
