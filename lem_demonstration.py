import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy import signal, stats
from scipy.fft import fftshift

import plot
import plot_utils
import sim_radar_data
import transforms
from mplem import lem, mp, mplem

data_name = "Visualize LEM"
data = sim_radar_data.data[data_name]

fs = 1e3
duration = 1.0
time = np.arange(duration * fs) / float(fs)
signal_time = data["signal"]
freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=1 / fs))
signal_freq = np.fft.fftshift(np.fft.fft(signal_time))

ncenter = data["ncenter"]  # true number of centers
results = lem(time, signal_time, 20, ncenter=ncenter)

# plot true signal
nrow = 1
ncoln = 2
fig, axs = plt.subplots(nrow, ncoln)
if nrow == 1:
  axs = [axs]
fig.suptitle("True Signal")
fig.set_size_inches(ncoln * 5, nrow * 3.5)
fig.subplots_adjust(left=0.07, right=0.95, bottom=0.15, top=0.78, wspace=0.4, hspace=0.4)
plot.plot_signal(axs[0][0], time, signal_time, fs, sep=4.0, ylabel="Imag     Real")
plot.plot_spectrogram(axs[0][1], signal_time, fs)
plt.savefig("LEMTrue.pdf", format="pdf")

# plot initial guess
nrow = 1
ncoln = 2
fig, axs = plt.subplots(nrow, ncoln)
if nrow == 1:
  axs = [axs]
fig.suptitle("Initial Guess")
fig.set_size_inches(ncoln * 5, nrow * 3.5)
fig.subplots_adjust(left=0.07, right=0.95, bottom=0.15, top=0.78, wspace=0.4, hspace=0.4)
mu_t, mu_f, sigma_t, sigma_f, c = results[0]
fake_signal = np.sum(transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t), axis=0)
plot.plot_signal(axs[0][0], time, fake_signal, fs, sep=4.0, ylabel="Imag     Real")
plot.plot_spectrogram(axs[0][1], fake_signal, fs)
plt.savefig("LEMInit.pdf", format="pdf")

# plot after 5 iterations
nrow = 1
ncoln = 2
fig, axs = plt.subplots(nrow, ncoln)
if nrow == 1:
  axs = [axs]
fig.suptitle("Approximated Signal (After 5 Iterations)")
fig.set_size_inches(ncoln * 5, nrow * 3.5)
fig.subplots_adjust(left=0.07, right=0.95, bottom=0.15, top=0.78, wspace=0.4, hspace=0.4)

mu_t, mu_f, sigma_t, sigma_f, c = results[5]
fake_signal = np.sum(transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t), axis=0)
plot.plot_signal(axs[0][0], time, fake_signal, fs, sep=4.0, ylabel="Imag     Real")
plot.plot_spectrogram(axs[0][1], fake_signal, fs)
plt.savefig("LEM5Iter.pdf", format="pdf")

# plot after 10 iterations
nrow = 1
ncoln = 2
fig, axs = plt.subplots(nrow, ncoln)
if nrow == 1:
  axs = [axs]
fig.suptitle("Approximated Signal (After 10 Iterations)")
fig.set_size_inches(ncoln * 5, nrow * 3.5)
fig.subplots_adjust(left=0.07, right=0.95, bottom=0.15, top=0.78, wspace=0.4, hspace=0.4)

mu_t, mu_f, sigma_t, sigma_f, c = results[10]
fake_signal = np.sum(transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t), axis=0)
plot.plot_signal(axs[0][0], time, fake_signal, fs, sep=4.0, ylabel="Imag     Real")
plot.plot_spectrogram(axs[0][1], fake_signal, fs)
plt.savefig("LEM10Iter.pdf", format="pdf")

plt.show()
