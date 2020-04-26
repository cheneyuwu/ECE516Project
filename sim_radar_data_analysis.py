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
from lem import lem, matching_pursuit, mplem

# for i, data_name in enumerate(["Start Walking", "Stabbing", "PickPocket"]):
data_name = "Stabbing"
i = 0
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
duration = 1.0  # 5 second
time = np.arange(duration * fs) / float(fs)
signal_time = data["signal"]
freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=1 / fs))
signal_freq = np.fft.fftshift(np.fft.fft(signal_time))

ncenter = data["ncenter"]

result = mplem(time, signal_time)

# Plot Spectrogram
plot.plot_spectrogram(axs[i][0], signal_time, fs, title=None if i else "Data Spectrogram")

# Plot Spectrogram
a, mu_t, mu_f, sigma_t, sigma_f, c = result
fake_signal = np.sum(a[..., np.newaxis] * transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t), axis=0)
plot.plot_spectrogram(axs[i][1], fake_signal, fs, title=None if i else "ACT Spectrogram")

# Generate Frequency Frequency Plot
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
plot.plot_chirplet_ff(axs[i][2], time, fake_signal, duration, fs, title=None if i else "Freq.-Freq. Plane")

plt.savefig(data_name.replace(" ", "") + ".pdf", format="pdf")
plt.show()
