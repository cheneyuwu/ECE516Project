import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy import signal, stats
from scipy.fft import fftshift

import plot
import plot_utils
import sim_radar_data

data_name = "Chirplet"
data = sim_radar_data.data[data_name]

fs = 1e3
duration = 1.0  # 5 second
time = np.arange(duration * fs) / float(fs)

# Create animation
nrow = 3
ncoln = 2
fig, axs = plt.subplots(nrow, ncoln)
fig.suptitle("LEM", fontsize=BIGGER_SIZE)
fig.set_size_inches(ncoln * 5, nrow * 5)
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.85, wspace=0.4, hspace=0.4)

# Plot Spectrogram
signal_time = sim_radar_data.data["Chirplet"]["signal"]
plot.plot_signal_real(axs[0][0], time, signal_time, fs)
plot.plot_signal_imag(axs[1][0], time, signal_time, fs)
plot.plot_spectrogram(axs[2][0], signal_time, fs)

# Plot Spectrogram
signal_time = sim_radar_data.data["Warblet"]["signal"]
plot.plot_signal_real(axs[0][1], time, signal_time, fs)
plot.plot_signal_imag(axs[1][1], time, signal_time, fs)
plot.plot_spectrogram(axs[2][1], signal_time, fs)

plt.show()
plt.savefig("VisualizeChirplets.pdf", format="pdf")