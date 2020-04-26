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

data_name = "Stabbing"  # try: "Start Walking", "Stabbing", "PickPocket"
data = sim_radar_data.data[data_name]

nrow = 1
ncoln = 3
fig, axs = plt.subplots(nrow, ncoln)
if nrow == 1:
  axs = [axs]
fig.suptitle(data_name)
fig.set_size_inches(ncoln * 5, nrow * 5)
fig.subplots_adjust(left=0.08, right=0.95, bottom=0.16, top=0.83, wspace=0.35, hspace=0.35)

fs = 1e3
duration = 1.0
time = np.arange(duration * fs) / float(fs)
signal_time = data["signal"]
freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=1 / fs))
signal_freq = np.fft.fftshift(np.fft.fft(signal_time))

# apply mplem
result = mplem(time, signal_time)

# plot data spectrogram
plot.plot_spectrogram(axs[0][0], signal_time, fs, title="Data Spectrogram")

# plot approximated signal spectrogram
a, mu_t, mu_f, sigma_t, sigma_f, c = result
fake_signal = np.sum(a[..., np.newaxis] * transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t), axis=0)
plot.plot_spectrogram(axs[0][1], fake_signal, fs, title="ACT Spectrogram")

# plot in freq-freq plane
fake_signal = np.sum(
    a[..., np.newaxis] * transforms.q_chirplet(
        time,
        duration / 2,
        mu_f,
        c,
        np.sqrt(2) * sigma_t,
    ),
    axis=0,
)
plot.plot_chirplet_ff(axs[0][2], time, fake_signal, duration, fs, title="Freq.-Freq. Plane")

plt.savefig(data_name.replace(" ", "") + ".pdf", format="pdf")
plt.show()
