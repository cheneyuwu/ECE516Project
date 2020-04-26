import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy import signal, stats
from scipy.fft import fftshift

import plot_utils
import transforms


def plot_signal(ax, time, signal_time, fs, title="Time Series", sep=2.5, ylabel="Imag               Real"):
  ax.plot(time, signal_time.real + sep, color="r", linewidth=0.5)
  ax.plot(time, signal_time.imag - sep, color="g", linewidth=0.5)
  ax.set_xlim([time[0], time[-1]])
  ax.set_ylim([-2 * sep, 2 * sep])
  ax.set_xlabel("")
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.get_xaxis().set_visible(False)
  ax.tick_params(axis='y', labelsize=0, length=0)
  # ax.get_yaxis().set_visible(False)

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)


def plot_signal_overlap(ax, time, signal_time, fs, title="Time Series"):
  ax.plot(time, signal_time.real, color="r", linewidth=0.5)
  ax.plot(time, signal_time.imag, color="g", linewidth=0.5)
  # ax.set_xlim([time[0], time[-1]])
  # ax.set_ylim([-5, 5])
  ax.set_xlabel("Time")
  ax.set_ylabel("Amplitude")
  ax.set_title(title)
  ax.tick_params(axis='x', labelsize=0, length=0)
  # ax.get_yaxis().set_visible(False)

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)


def plot_signal_real(ax, time, signal_time, fs, title="Signal (Real)"):
  ax.plot(time, signal_time.real, color="r")
  ax.set_xlim([time[0], time[-1]])
  ax.set_ylim([-5, 5])
  ax.set_xlabel("Time [Second]")
  ax.set_ylabel("Amplitude")
  ax.set_title(title)


def plot_signal_imag(ax, time, signal_time, fs, title="Signal (Imaginary)"):
  ax.plot(time, signal_time.imag, color="g")
  ax.set_xlim([time[0], time[-1]])
  ax.set_ylim([-5, 5])
  ax.set_xlabel("Time [Second]")
  ax.set_ylabel("Amplitude")
  ax.set_title(title)


def plot_spectrogram(ax, signal_time, fs, title="Spectrogram"):
  ax.clear()
  ax.specgram(
      signal_time,
      Fs=1.0,
      window=signal.get_window(("gaussian", 20), 128),
      noverlap=112,
      NFFT=128,
      scale="linear",
      # cmap="Greys",
  )
  ax.set_xlabel("Time")
  ax.set_ylabel("Frequency")
  ax.tick_params(axis='x', labelsize=0, length=0)
  ax.set_yticks([-0.5, 0, 0.5])
  ax.set_title(title)


def plot_spectrogram_log(ax, signal_time, fs, title=None):
  ax.clear()
  ax.specgram(
      signal_time,
      Fs=1.0,
      window=signal.get_window(("gaussian", 20), 128),
      noverlap=112,
      NFFT=128,
      # cmap="Greys",
  )
  ax.set_xlabel("Time")
  ax.set_ylabel("Frequency")
  ax.tick_params(axis='x', labelsize=0, length=0)
  ax.set_yticks([-0.5, 0, 0.5])
  ax.set_title(title)


def plot_chirplet_ff(ax, time, signal_time, duration, fs, title="F-F Plane"):
  # Chirplet transform analysis
  sample = 200
  xy_range = int(fs / 2)  # half of the sampling frequency to satisfy Nyquist boundary, for chirplet transform

  # Chirplet Transform adaptation
  fb = np.linspace(-xy_range, xy_range, sample)
  fe = np.linspace(-xy_range, xy_range, sample)
  fb, fe = np.meshgrid(fb, fe)
  window = transforms.q_chirplet_fbe(time, duration / 2, fb, fe,
                                     np.sqrt(2) / 3)  # dt set to np.sqrt(2) / 3 so that 3*sigma is 1
  res = np.inner(np.conj(window), signal_time)
  res = np.absolute(res)
  # res = np.clip(res, 80, np.inf)
  # ax.contour(fb, fe, res)
  ax.pcolormesh(fb / fs, fe / fs, res)
  ax.ticklabel_format(style="sci", scilimits=(-2, 2), axis="both")
  ax.set_xlabel("Beginning Freq.")
  ax.set_ylabel("Ending Freq.")
  ax.set_xticks([-0.5, 0, 0.5])
  ax.set_yticks([-0.5, 0, 0.5])
  ax.set_title(title)


def plot_warblet_dd(ax, time, signal_time, duration, fs, title="Dilation-Dilation Plane"):

  # Chirplet transform analysis
  sample = 200
  xy_range = int(fs / 2)  # half of the sampling frequency to satisfy Nyquist boundary, for chirplet transform

  # Warblet Transform adaptation
  bm = np.linspace(0.0, xy_range, sample)
  fm = np.linspace(0, 10, sample)
  fm, bm = np.meshgrid(fm, bm)
  window = transforms.warblet(time, duration / 2, 0, fm, bm, 0, duration * np.sqrt(2) / 3)
  res = np.inner(np.conj(window), signal_time)
  res = np.absolute(res)
  # res = np.clip(res, 100, np.inf)
  # ax.contour(fm, bm, res)
  ax.pcolormesh(fm / fs, bm / fs, res)
  ax.ticklabel_format(style="sci", scilimits=(-2, 2), axis="both")
  ax.set_xlabel("Modulation Freq.")
  ax.set_ylabel("Modulation Index")
  ax.set_xticks([0, 0.5e-2, 1e-2])
  ax.set_yticks([0, 0.5])
  ax.set_title(title)