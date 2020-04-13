import csv
import os
import os.path as osp

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.fft import fftshift

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# set sizes
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 16
plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize


data_dir = osp.join(os.getcwd(), "HDR_audio_24GHz_complex_radar")

# file_name = "roomtone.txt"
# file_name = "dropping_1_object.txt"
# file_name = "dropping_2_objects.txt"
# file_name = "dropping_digikey_resistor_bags.txt"
file_name = "drop_resistor_tape.txt"

ith_data = 0

# Load Data
data = np.empty((8, 0))
with open(osp.join(data_dir, file_name), newline="") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        row = np.array([float(r) for r in row[:8]]).reshape(-1, 1)
        data = np.append(data, row, axis=1)
        # data of shape [8, number of samples]


fig = plt.figure()
fig.set_size_inches(3 * 5, 2 * 5)
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.25, hspace=0.25)  # gym envs

# data = data[:, 5500:7000]

# Step 0
# visualize radar data
ax = fig.add_subplot(231)
ax.plot(data[ith_data, ...], label="real", color="r")
ax.plot(data[ith_data + 4, ...], label="imag", color="g")
ax.set_xlabel("sample index")
ax.set_ylabel("sample value")
ax.set_title("Radar Data, Uncalibrated")
ax.legend()

# plot uncalibrated data
ax = fig.add_subplot(232)
ax.scatter(data[ith_data, ...], data[ith_data + 4, ...])
ax.set_xlabel("real")
ax.set_ylabel("imag")
ax.set_title("Real vs Imag, Uncalibrated")

f, t, Sxx = signal.spectrogram(data[ith_data, ...] + 1j * data[ith_data + 4, ...], return_onesided=False)
ax = fig.add_subplot(233)
ax.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0))
ax.set_ylabel("Frequency [Hz]")
ax.set_xlabel("Time [sec]")
ax.set_title("Spectrogram, Uncalibrated")


# Step 1: calibrate data
# subtract mean
mean = data.mean(axis=1)
data = data - mean.reshape(8, -1)
# center the data
for i in range(4):
    cov = np.cov(data[[i, i + 4], ...])
    L = np.linalg.cholesky(cov)
    data[[i, i + 4], ...] = np.linalg.inv(L) @ data[[i, i + 4], ...]

# plot uncalibrated data
ax = fig.add_subplot(234)
ax.plot(data[ith_data, ...], label="real", color="r")
ax.plot(data[ith_data + 4, ...], label="imag", color="g")
ax.set_xlabel("sample index")
ax.set_ylabel("sample value")
ax.set_title("Radar Data, Calibrated")
ax.legend()

# plot calibrated data
ax = fig.add_subplot(235)
ax.scatter(data[ith_data, ...], data[ith_data + 4, ...])
ax.set_xlabel("real")
ax.set_ylabel("imag")
ax.set_title("Real vs Imag, Calibrated")

f, t, Sxx = signal.spectrogram(data[ith_data, ...] + 1j * data[ith_data + 4, ...], return_onesided=False)
ax = fig.add_subplot(236)
ax.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0))
ax.set_ylabel("Frequency [Hz]")
ax.set_xlabel("Time [sec]")
ax.set_title("Spectrogram, Calibrated")

plt.savefig(os.path.splitext(file_name)[0] + ".pdf")
plt.show()
