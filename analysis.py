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

# Generate some arbitrary data
duration = 5.0  # 5 second
time = np.arange(duration * fs) / float(fs)
# Rest Clutter
data_dict["Walking"] = transforms.q_chirp(time, duration / 2, -100, 0)
# Car Hazard
data_dict["Car_Hazard"] = data_dict["Walking"] + transforms.q_chirp(time, duration / 2, 200, 20)
# Floating Iceberg Fragment
data_dict["Floating_Iceberg_Fragment"] = transforms.warble(time, duration / 2, 0, 0.5, 200, 0)


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
fig.suptitle(data_name, fontsize=BIGGER_SIZE)
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
ax.scatter(data.real, data.imag)
ax.set_xlabel("Real")
ax.set_ylabel("Imag")
ax.set_title("Real vs Imag, Uncalibrated")


f, t, Sxx = signal.spectrogram(
    data,
    return_onesided=False,
    fs=fs,
    window=("tukey", 0.25),
    nperseg=128,
    noverlap=112,
    # window=("tukey", 1.0),
    # nperseg=64,
    # noverlap=48,
)
intensity = fftshift(Sxx, axes=0)
intensity = np.log10(fftshift(Sxx, axes=0))
ax = axs[0][2]
ax.pcolormesh(t, fftshift(f), intensity)
ax.set_ylabel("Frequency [Hz]")
ax.set_xlabel("Time [sec]")
ax.set_title("Spectrogram, Uncalibrated")

# Chirplet transform analysis
sample = 200
xy_range = int(fs / 2)  # half of the sampling frequency to satisfy Nyquist boundary, for chirplet transform

# # Chirplet Transform adaptation
# fb = np.linspace(-xy_range, xy_range, sample)
# fe = np.linspace(-xy_range, xy_range, sample)
# fb, fe = np.meshgrid(fb, fe)
# window = transforms.q_chirplet_fbe(
#     time, duration / 2, fb, fe, np.sqrt(2) / 3
# )  # dt set to np.sqrt(2) / 3 so that 3*sigma is 1
# res = np.inner(np.conj(window), data)
# res = np.absolute(res)
# res = np.clip(res, 80, np.inf)
# ax = axs[0][3]
# ax.contour(fb, fe, res)
# ax.set_xlabel("Frequency Begin")
# ax.set_ylabel("Frequency End")
# ax.set_title("Frequency Begin vs Frequency End, Uncalibrated")

# Warblet Transform adaptation
bm = np.linspace(-xy_range, xy_range, sample)
fm = np.linspace(0.1, 1, sample)
fm, bm = np.meshgrid(fm, bm)
window = transforms.warblet(time, duration / 2, 0, fm, bm, 0, duration / 2 * np.sqrt(2) / 3)
res = np.inner(np.conj(window), data)
res = np.absolute(res)
# res = np.clip(res, 100, np.inf)
ax = axs[0][3]
ax.contour(fm, bm, res)
ax.set_xlabel("Frequency of Modulation")
ax.set_ylabel("Amplitude of Modulation")
ax.set_title("Frequency vs Amplitude of Modulation, Uncalibrated")


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
ax.scatter(data.real, data.imag)
ax.set_xlabel("real")
ax.set_ylabel("imag")
ax.set_title("Real vs Imag, Calibrated")

f, t, Sxx = signal.spectrogram(
    data,
    return_onesided=False,
    fs=fs,
    window=("tukey", 0.25),
    nperseg=128,
    noverlap=112,  #  window=("tukey", 0.25), nperseg=128, noverlap=112,
)
intensity = fftshift(Sxx, axes=0)
intensity = np.log10(fftshift(Sxx, axes=0))
ax = axs[1][2]
ax.pcolormesh(t, fftshift(f), intensity)
ax.set_ylabel("Frequency [Hz]")
ax.set_xlabel("Time [sec]")
ax.set_title("Spectrogram, Calibrated")

# # Chirplet Transform adaptation
# fb = np.linspace(-xy_range, xy_range, sample)
# fe = np.linspace(-xy_range, xy_range, sample)
# fb, fe = np.meshgrid(fb, fe)
# window = transforms.q_chirplet_fbe(time, duration / 2, fb, fe, duration / 2 * np.sqrt(2) / 3)
# res = np.inner(np.conj(window), data)
# ax = axs[1][3]
# ax.contour(fb, fe, np.absolute(res))
# ax.set_xlabel("f begin")
# ax.set_ylabel("f end")
# ax.set_title("Freq Begin vs Freq End, Calibrated")

# Warblet Transform adaptation
bm = np.linspace(-xy_range, xy_range, sample)
fm = np.linspace(0.1, 1, sample)
fm, bm = np.meshgrid(fm, bm)
window = transforms.warblet(time, duration / 2, 0, fm, bm, 0, duration / 2 * np.sqrt(2) / 3)
res = np.inner(np.conj(window), data)
res = np.absolute(res)
# res = np.clip(res, 100, np.inf)
ax = axs[1][3]
ax.contour(fm, bm, res)
ax.set_xlabel("frequency of modulation")
ax.set_ylabel("amplitude")
ax.set_title("Frequency of Modulation vs Amplitude, Calibrated")

plt.savefig(data_name + ".pdf", format="pdf")
plt.show()
