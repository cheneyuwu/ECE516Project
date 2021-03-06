import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy import signal, stats
from scipy.fft import fftshift

import plot
import plot_utils
import sim_radar_data

fs = 1e3
duration = 1.0
time = np.arange(duration * fs) / float(fs)

nrow = 2
ncoln = 2
fig, axs = plt.subplots(nrow, ncoln)
fig.set_size_inches(ncoln * 5, nrow * 4)
fig.subplots_adjust(left=0.12, right=0.95, bottom=0.05, top=0.95, wspace=0.35, hspace=0.20)
# visualize q-chirplet
signal_time = sim_radar_data.data["Chirplet"]["signal"]
plot.plot_signal(axs[0][0], time, signal_time, fs, title="Q-Chirplet Time Series")
plot.plot_spectrogram(axs[1][0], signal_time, fs, title="Q-Chirplet Spectrogram")
# visualize warblet
signal_time = sim_radar_data.data["Warblet"]["signal"]
plot.plot_signal(axs[0][1], time, signal_time, fs, title="W-Chirplet Time Series")
plot.plot_spectrogram(axs[1][1], signal_time, fs, title="W-Chirplet Spectrogram")

plt.savefig("VisualizeChirplets.pdf", format="pdf")
plt.show()
