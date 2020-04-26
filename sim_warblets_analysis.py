import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Ellipse
from scipy import signal, stats
from scipy.fft import fftshift

import plot
import plot_utils
import sim_radar_data

fs = 1e3
duration = 1.0
time = np.arange(duration * fs) / float(fs)

# Create animation
nrow = 1
ncoln = 2
fig, axs = plt.subplots(nrow, ncoln)
if nrow == 1:
  axs = [axs]
# fig.suptitle("Waving, 2 Hands")
fig.set_size_inches(ncoln * 6, nrow * 5)
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.16, top=0.9, wspace=0.35, hspace=0.35)

# Plot Spectrogram
signal_time = sim_radar_data.data["Waving"]["signal"]
# plot.plot_signal(axs[0][0], time, signal_time, fs)
plot.plot_spectrogram(axs[0][0], signal_time, fs)
plot.plot_warblet_dd(axs[0][1], time, signal_time, duration, fs)

circle = Ellipse((0.2e-2, 0.2), 1e-3, 0.5e-1, color='r', fill=False)
axs[0][1].add_artist(circle)
circle = Ellipse((0.3e-2, 0.3), 1e-3, 0.5e-1, color='r', fill=False)
axs[0][1].add_artist(circle)

plt.savefig("WavingSimulation.pdf", format="pdf")
plt.show()
