import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy import signal, stats
from scipy.fft import fftshift

import plot_utils


def plot_signal_real(ax, time, signal_time, fs, title="Signal (Real)"):
  ax.clear()
  ax.plot(time, signal_time.real, color="r")
  ax.set_xlim([time[0], time[-1]])
  ax.set_ylim([-3, 3])
  ax.set_xlabel("Time [Second]")
  ax.set_ylabel("Amplitude")
  ax.set_title(title)


def plot_signal_imag(ax, time, signal_time, fs, title="Signal (Imaginary)"):
  ax.clear()
  ax.plot(time, signal_time.imag, color="g")
  ax.set_xlim([time[0], time[-1]])
  ax.set_ylim([-3, 3])
  ax.set_xlabel("Time [Second]")
  ax.set_ylabel("Amplitude")
  ax.set_title(title)


def plot_spectrogram(ax, signal_time, fs, title="Spectrogram of Signal"):
  ax.clear()
  f, t, Sxx = signal.spectrogram(
      signal_time,
      return_onesided=False,
      fs=fs,
      window=("tukey", 0.25),
      nperseg=70,
      noverlap=60,
  )
  ax.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0), cmap="Greys_r")
  ax.set_ylabel("Frequency [Hz]")
  ax.set_xlabel("Time [sec]")
  ax.set_title(title)
