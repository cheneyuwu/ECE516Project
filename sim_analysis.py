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

nrow = 3
ncoln = 3
fig, axs = plt.subplots(nrow, ncoln)
# fig.suptitle("Simulated Radar Dataset")
fig.set_size_inches(ncoln * 5, nrow * 5)
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.3, hspace=0.3)

data_name = "PickPocket"

for i, data_name in enumerate(["Start Walking", "Stabbing", "PickPocket"]):
  data = sim_radar_data.data[data_name]

  fs = 1e3
  duration = 1.0  # 5 second
  time = np.arange(duration * fs) / float(fs)
  signal_time = data["signal"]
  freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=1 / fs))
  signal_freq = np.fft.fftshift(np.fft.fft(signal_time))

  ncenter = data["ncenter"]

  result = mplem(time, signal_time)
  # result = [matching_pursuit(time, signal_time)]
  # result = lem(time, signal_time, 400, ncenter=ncenter)

  # mu_t, mu_f, sigma_t, sigma_f, c = result[-1]
  # print("Final: ", mu_t, mu_f, sigma_t, sigma_f, c)

  # plot.plot_signal(axs[i, 0], time, signal_time, fs)

  # Plot Spectrogram
  plot.plot_spectrogram(axs[i, 0], signal_time, fs)

  # Plot Spectrogram
  a, mu_t, mu_f, sigma_t, sigma_f, c = result
  print(result)
  print(np.absolute(a))
  fake_signal = np.sum(a[..., np.newaxis] * transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t), axis=0)
  # fake_signal = a * transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t)
  plot.plot_spectrogram(axs[i, 1], fake_signal, fs)

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
  plot.plot_chirplet_ff(axs[i, 2], time, fake_signal, duration, fs)

# plt.savefig(data_name + ".pdf", format="pdf")
plt.show()
