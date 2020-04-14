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
fs = 1e3  # the assumed sample frequency of all data
data_dir = osp.join(os.getcwd(), "HDR_audio_24GHz_complex_radar")
data_dict = dict()  # a dictionary containing different gains for HDR measurement

for file_name in ["dropping_1_object"]:
    csvdata = np.empty((8, 0))
    with open(osp.join(data_dir, file_name + ".txt"), newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            row = np.array([float(r) for r in row[:8]]).reshape(-1, 1)
            csvdata = np.append(csvdata, row, axis=1)  # data of shape [8, number of samples]
    data_dict[file_name + "_DC10"] = np.stack((csvdata[0, ...], csvdata[0 + 4, ...]))
    data_dict[file_name + "_AC100"] = np.stack((csvdata[1, ...], csvdata[1 + 4, ...]))
    data_dict[file_name + "_AC1000"] = np.stack((csvdata[2, ...], csvdata[2 + 4, ...]))
    data_dict[file_name + "_AC10000"] = np.stack((csvdata[3, ...], csvdata[3 + 4, ...]))
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
    data_dict[file_name + "_AC10000"] = np.mean([k for k in data_dict.values()], axis=0)

# Generate some arbitrary data
N = 1e4
time = np.arange(N) / float(fs)
sig = transforms.q_chirplet(time, 0.5, 250, 500, 0.1)  #  + transforms.q_chirplet(time, 0.5, -250, -500, 0.1)
# Select data of format (2, n) real and imaginary
data_dict["Signal: Chirplet tc=0.5, fc=250, c=500, delta_t=0.1"] = np.stack((sig.real, sig.imag))


# Working on data
# data_name = "Signal: Chirplet tc=0.5, fc=250, c=500, delta_t=0.1"
data_name = "dropping_1_object_AC10000"
data = data_dict[data_name]
# data = data[:, 4500:5500]
time = np.arange(data.shape[1]) / float(fs)


nrow = 2
ncoln = 3
fig, axs = plt.subplots(nrow, ncoln)
fig.suptitle(data_name, fontsize=BIGGER_SIZE)
fig.set_size_inches(ncoln * 5, nrow * 5)
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)  # gym envs

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


f, t, Sxx = signal.spectrogram(
    data[0, ...] + 1j * data[1, ...], return_onesided=False, fs=fs, window=("tukey", 0.25), nperseg=128, noverlap=112,
)
ax = axs[0][2]
ax.pcolormesh(t, fftshift(f), np.log10(fftshift(Sxx, axes=0)))
ax.set_ylabel("Frequency [Hz]")
ax.set_xlabel("Time [sec]")
ax.set_title("Spectrogram, Uncalibrated")

# # generate chirp
# sample = 200
# xy_range = 10000

# fb = np.linspace(-xy_range, xy_range, sample)
# fe = np.linspace(-xy_range, xy_range, sample)
# fb, fe = np.meshgrid(fb, fe)
# window = transforms.q_chirplet_fbe(time, 0.5, fb, fe, 0.1)
# res = np.inner(np.conj(window), data[0] + 1j * data[1])
# ax = axs[0][3]
# ax.contour(fb, fe, np.absolute(res))
# ax.set_xlabel("f begin")
# ax.set_ylabel("f end")
# ax.set_title("Freq Begin vs Freq End, Uncalibrated")

# fc = np.linspace(-xy_range, xy_range, sample)
# c = np.linspace(-xy_range, xy_range, sample)
# fc, c = np.meshgrid(fc, c)
# window = transforms.q_chirplet(time, 0.5, fc, c, 0.1)
# res = np.inner(np.conj(window), data[0] + 1j * data[1])
# ax = axs[0][4]
# ax.contour(fc, c, np.absolute(res))
# ax.set_xlabel("center freq")
# ax.set_ylabel("slope")
# ax.set_title("Center Freq vs Slope, Uncalibrated")


# Step 1: calibrate data
# subtract mean
mean = data.mean(axis=1, keepdims=True)
data = data - mean
# center the data
cov = np.cov(data)
L = np.linalg.cholesky(cov)
data = np.linalg.inv(L) @ data


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

f, t, Sxx = signal.spectrogram(
    data[0, ...] + 1j * data[1, ...], return_onesided=False, fs=fs, window=("tukey", 0.25), nperseg=128, noverlap=112,
)
ax = axs[1][2]
ax.pcolormesh(t, fftshift(f), np.log(fftshift(Sxx, axes=0)))
ax.set_ylabel("Frequency [Hz]")
ax.set_xlabel("Time [sec]")
ax.set_title("Spectrogram, Calibrated")

plt.savefig(data_name + ".jpg")
plt.show()
