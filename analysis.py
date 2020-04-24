import csv
import os
import os.path as osp

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.fft import fftshift
from scipy.ndimage.interpolation import shift

import transforms
import plot_utils
import plot

# Load data from file
fs = 1e3  # the assumed sample frequency of all data
data_dir = osp.join(os.getcwd(), "HDR_audio_24GHz_complex_radar")
data_dict = dict()  # a dictionary containing different gains for HDR measurement

for file_name in ["dropping_1_object", "spin_warblet"]:
  csvdata = np.empty((8, 0))
  with open(osp.join(data_dir, file_name + ".txt"), newline="") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
      row = np.array([float(r) for r in row[:8]]).reshape(-1, 1)
      csvdata = np.append(csvdata, row, axis=1)  # data of shape [8, number of samples]
  data_dict[file_name + "_DC10"] = csvdata[0, ...] + 1j * csvdata[0 + 4, ...]
  data_dict[file_name + "_AC100"] = csvdata[1, ...] + 1j * csvdata[1 + 4, ...]
  data_dict[file_name + "_AC1000"] = csvdata[2, ...] + 1j * csvdata[2 + 4, ...]
  data_dict[file_name + "_AC10000"] = csvdata[3, ...] + 1j * csvdata[3 + 4, ...]
  # ###
  # # normalize data
  # for k, data in data_dict.items():
  #     mean = data.mean(axis=1, keepdims=True)
  #     data = data - mean
  #     # center the data
  #     cov = np.cov(data)
  #     L = np.linalg.cholesky(cov)
  #     data_dict[k] = np.linalg.inv(L) @ data
  # ###
  data_dict[file_name + "_Combined"] = np.mean([v for k, v in data_dict.items() if k.startswith(file_name)], axis=0)

# Working on data
data_name = "spin_warblet_Combined"
data = data_dict[data_name]

# Update duration time
data = data[5000:10000]  # adjust duration
time = np.arange(data.shape[0]) / float(fs)
duration = len(time) / fs

# Generate Plots
#########################################################################

nrow = 2
ncoln = 4
fig, axs = plt.subplots(nrow, ncoln)
if nrow == 1:
  axs = [axs]
fig.suptitle(data_name)
fig.set_size_inches(ncoln * 5, nrow * 5)
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.9, wspace=0.4, hspace=0.4)

# Step 0
# visualize radar data
ax = axs[0][0]
ax.plot(data.real, label="Real", color="r")
ax.plot(data.imag, label="Imag", color="g")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Sample Value")
ax.set_title("Radar Data, Uncalibrated")
ax.legend()

ax = axs[0][1]
# ax.scatter(data.real, data.imag)
h = ax.hist2d(data.real, data.imag, bins=100)
ax.set_xlabel("Real")
ax.set_ylabel("Imag")
ax.set_title("Real vs Imag, Uncalibrated")
plt.colorbar(h[3], ax=ax)

plot.plot_spectrogram_log(axs[0][2], data, fs)

# # Chirplet Transform adaptation
# plot.plot_chirplet_ff(axs[0][3], time, data, duration, fs, title="Frequency-Frequency Plane")

# Warblet Transform adaptation
# plot.plot_warblet_dd(axs[0][3], time, data, duration, fs, title="Frequency-Frequency Plane")

# return

# Step 1: calibrate data
# subtract mean
mean = data.mean(axis=0, keepdims=True)
data = data - mean
data = np.array((data.real, data.imag))
# center the data
cov = np.cov(data)
L = np.linalg.cholesky(cov)
data = np.linalg.inv(L) @ data
data = data[0, ...] + 1j * data[1, ...]

# plot uncalibrated data
ax = axs[1][0]
ax.plot(data.real, label="real", color="r")
ax.plot(data.imag, label="imag", color="g")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Sample Value")
ax.set_title("Radar Data, Calibrated")
ax.legend()

ax = axs[1][1]
h = ax.hist2d(data.real, data.imag, bins=100)
ax.set_xlabel("real")
ax.set_ylabel("imag")
ax.set_title("Real vs Imag, Calibrated")
plt.colorbar(h[3], ax=ax)

plot.plot_spectrogram_log(axs[1][2], data, fs)

# # Chirplet Transform adaptation
# plot.plot_chirplet_ff(axs[1][3], time, data, duration, fs, title="Frequency-Frequency Plane")

# Warblet Transform adaptation
# plot.plot_warblet_dd(axs[1][3], time, data, duration, fs, title="Frequency-Frequency Plane")

plt.savefig(data_name + ".pdf", format="pdf")
plt.show()
