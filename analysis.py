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

# Set Figure sizes
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


# # Load data from file
# data_dir = osp.join(os.getcwd(), "HDR_audio_24GHz_complex_radar")

# file_name = "roomtone.txt"
# # file_name = "dropping_1_object.txt"
# # file_name = "dropping_2_objects.txt"
# # file_name = "dropping_digikey_resistor_bags.txt"
# # file_name = "drop_resistor_tape.txt"

# csvdata = np.empty((8, 0))
# with open(osp.join(data_dir, file_name), newline="") as f:
#     reader = csv.reader(f, delimiter="\t")
#     for row in reader:
#         row = np.array([float(r) for r in row[:8]]).reshape(-1, 1)
#         csvdata = np.append(csvdata, row, axis=1)  # data of shape [8, number of samples]
data = dict()  # a dictionary containing different gains for HDR measurement
# data["10"] = np.stack((csvdata[0, ...], csvdata[0 + 4, ...]))
# data["100"] = np.stack((csvdata[1, ...], csvdata[1 + 4, ...]))
# data["1000"] = np.stack((csvdata[2, ...], csvdata[2 + 4, ...]))
# data["10000"] = np.stack((csvdata[3, ...], csvdata[3 + 4, ...]))

# Generate some arbitrary data
fs = 1e4
N = 1e5
time = np.arange(N) / float(fs)

# signal 1
amp = 2 * np.sqrt(2)
mod = 500 * np.cos(2 * np.pi * 0.25 * time)
carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
data["100"] = np.stack((carrier, signal.hilbert(carrier)))

# signal 2
fc = 1000
tc = 0
c = 0
sig = np.exp(1j * 2 * np.pi * (c * ((time - tc) ** 2) + fc * (time - tc)))
data["100"] = np.stack((sig.real, sig.imag))


# Select data of format (2, n) real and imaginary
data = data["100"]






nrow = 2
ncoln = 3
fig, axs = plt.subplots(nrow, ncoln)
fig.set_size_inches(ncoln * 5, nrow * 5)
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.25, hspace=0.25)  # gym envs

# Step 0
# visualize radar data
ax = axs[0][0]
ax.plot(data[0, ...], label="real", color="r")
ax.plot(data[1, ...], label="imag", color="g")
ax.set_xlabel("sample index")
ax.set_ylabel("sample value")
ax.set_title("Radar Data, Uncalibrated")
ax.legend()

ax = axs[0][1]
ax.scatter(data[0, ...], data[1, ...])
ax.set_xlabel("real")
ax.set_ylabel("imag")
ax.set_title("Real vs Imag, Uncalibrated")

f, t, Sxx = signal.spectrogram(data[0, ...] + 1j * data[1, ...], return_onesided=False, fs=fs)
ax = axs[0][2]
ax.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0))
ax.set_ylabel("Frequency [Hz]")
ax.set_xlabel("Time [sec]")
ax.set_title("Spectrogram, Uncalibrated")


# # Step 1: calibrate data
# # subtract mean
# mean = data.mean(axis=1, keepdims=True)
# data = data - mean
# # center the data
# cov = np.cov(data)
# L = np.linalg.cholesky(cov)
# data = np.linalg.inv(L) @ data


# plot uncalibrated data
ax = axs[1][0]
ax.plot(data[0, ...], label="real", color="r")
ax.plot(data[1, ...], label="imag", color="g")
ax.set_xlabel("sample index")
ax.set_ylabel("sample value")
ax.set_title("Radar Data, Calibrated")
ax.legend()

ax = axs[1][1]
ax.scatter(data[0, ...], data[1, ...])
ax.set_xlabel("real")
ax.set_ylabel("imag")
ax.set_title("Real vs Imag, Calibrated")

f, t, Sxx = signal.spectrogram(data[0, ...] + 1j * data[1, ...], return_onesided=False, fs=fs)
ax = axs[1][2]
ax.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0))
ax.set_ylabel("Frequency [Hz]")
ax.set_xlabel("Time [sec]")
ax.set_title("Spectrogram, Calibrated")

# plt.savefig(os.path.splitext(file_name)[0] + ".pdf")
plt.show()
