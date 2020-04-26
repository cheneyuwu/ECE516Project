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

from lem import lem, matching_pursuit, mplem

from matplotlib.patches import Ellipse


def normalize_data(data):
  # subtract mean
  mean = data.mean(axis=0, keepdims=True)
  data = data - mean
  data = np.array((data.real, data.imag))
  # center the data
  cov = np.cov(data)
  L = np.linalg.cholesky(cov)
  data = np.linalg.inv(L) @ data
  data = data[0, ...] + 1j * data[1, ...]
  return data


# Load data from file
fs = 1e3  # the assumed sample frequency of all data
data_dir = osp.join(os.getcwd(), "HDR_audio_24GHz_complex_radar")
data_dict = dict()  # a dictionary containing different gains for HDR measurement

# "dropping_1_object", "spin_warblet", "ruler_16_inches", "ruler_8_inches"
for file_name in ["dropping_1_object"]:
  csvdata = np.empty((8, 0))
  with open(osp.join(data_dir, file_name + ".txt"), newline="") as f:
    reader = csv.reader(f, delimiter="\t")
    i = 0
    for row in reader:
      i += 1
      if i > 10000:
        break
      row = np.array([float(r) for r in row[:8]]).reshape(-1, 1)
      try:
        csvdata = np.append(csvdata, row, axis=1)  # data of shape [8, number of samples]
      except:
        print("error on row: ", row)
        print("previous row: ", csvdata[-1])
        exit(1)
  data_dict[file_name + "_DC10"] = csvdata[0, ...] + 1j * csvdata[0 + 4, ...]
  data_dict[file_name + "_AC100"] = csvdata[1, ...] + 1j * csvdata[1 + 4, ...]
  data_dict[file_name + "_AC1000"] = csvdata[2, ...] + 1j * csvdata[2 + 4, ...]
  data_dict[file_name + "_AC10000"] = csvdata[3, ...] + 1j * csvdata[3 + 4, ...]
  data_dict[file_name + "_Combined"] = np.mean([v for k, v in data_dict.items() if k.startswith(file_name)], axis=0)

# Generate Calibration Plots
#########################################################################

# Working on data
data_name = "dropping_1_object_Combined"
data = data_dict[data_name]

# Update duration time
data = data[3500:5500]  # adjust duration
time = np.arange(data.shape[0]) / float(fs)
duration = len(time) / fs

nrow = 2
ncoln = 2
fig, axs = plt.subplots(nrow, ncoln)
if nrow == 1:
  axs = [axs]
fig.suptitle("Dropping 1 Object Data Calibration")
fig.set_size_inches(ncoln * 5, nrow * 4)
fig.subplots_adjust(left=0.12, right=0.95, bottom=0.05, top=0.87, wspace=0.35, hspace=0.35)

ax = axs[0][0]
h = ax.hist2d(data.real, data.imag, bins=100)
ax.set_xlabel("Real")
ax.set_ylabel("Imag")
ax.set_xlim([-3000, 3000])
ax.set_ylim([-3000, 3000])
ax.ticklabel_format(style="sci", scilimits=(-2, 2), axis="both")
ax.set_title("Before Calibration")

plot.plot_spectrogram_log(axs[1][0], data, fs, title=None)

# Step 1: calibrate data
data = normalize_data(data)

ax = axs[0][1]
h = ax.hist2d(data.real, data.imag, bins=100)
ax.set_xlabel("Real")
ax.set_ylabel("Imag")
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_title("After Calibration")

plot.plot_spectrogram_log(axs[1][1], data, fs, title=None)

plt.savefig("Dropping1ObjectCalibration.pdf", format="pdf")
plt.show()

# Generate Chirplet Plots
###################################################################################################

# # Working on data
# data_name = "dropping_1_object_Combined"
# data = data_dict[data_name]

# # Update duration time
# print(len(data))
# data = data[4500:5500]  # adjust dropping 1 object
# # data = data[5500:6500]  # adjust dropping 2 objects
# time = np.arange(data.shape[0]) / float(fs)
# duration = len(time) / fs

# nrow = 1
# ncoln = 3
# fig, axs = plt.subplots(nrow, ncoln)
# if nrow == 1:
#   axs = [axs]
# fig.suptitle("Dropping 1 Object")
# fig.set_size_inches(ncoln * 5, nrow * 5)
# fig.subplots_adjust(left=0.08, right=0.95, bottom=0.16, top=0.83, wspace=0.35, hspace=0.35)

# # Step 1: calibrate data
# data = normalize_data(data)

# plot.plot_spectrogram_log(axs[0][0], data, fs, title="Data Spectrogram (Db)")
# # plot.plot_chirplet_ff(axs[0][2], time, data, duration, fs, title="Freq.-Freq. Plane")

# result = mplem(time, data)
# # Plot Spectrogram
# a, mu_t, mu_f, sigma_t, sigma_f, c = result
# fake_signal = np.sum(a[..., np.newaxis] * transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t), axis=0)
# plot.plot_spectrogram(axs[0][1], fake_signal, fs, title="ACT Spectrogram")

# # Generate Frequency Frequency Plot
# fake_signal = np.sum(
#     a[..., np.newaxis] * transforms.q_chirplet(
#         time,
#         duration / 2,
#         mu_f,
#         c,
#         np.sqrt(2) * sigma_t,
#     ),
#     axis=0,
# )
# plot.plot_chirplet_ff(axs[0][2], time, fake_signal, duration, fs, title="Freq.-Freq. Plane")

# plt.savefig("Dropping1Object.pdf", format="pdf")
# plt.show()

# Generate Chirplet Plots
###################################################################################################

# # Working on data
# data_name = "spin_warblet_Combined"
# data = data_dict[data_name]

# # Update duration time
# print(len(data))
# data = data[4500:8500]  # adjust spinning warblet
# # data = data[4000:8000]  # adjust duration 8 inches
# # data = data[0:4000]  # adjust duration 16 inches
# time = np.arange(data.shape[0]) / float(fs)
# duration = len(time) / fs

# nrow = 1
# ncoln = 2
# fig, axs = plt.subplots(nrow, ncoln)
# if nrow == 1:
#   axs = [axs]
# fig.suptitle("Spin Warblet")
# fig.set_size_inches(ncoln * 6, nrow * 5)
# fig.subplots_adjust(left=0.12, right=0.95, bottom=0.16, top=0.83, wspace=0.35, hspace=0.35)

# # Step 1: calibrate data
# data = normalize_data(data)

# plot.plot_spectrogram_log(axs[0][0], data, fs, title="Data Spectrogram (Db)")
# plot.plot_warblet_dd(axs[0][1], time, data, duration, fs)

# # # 16 inch
# # circle = Ellipse((0.74e-2, 0.02), 1e-3, 0.5e-1, color='r', fill=False)
# # axs[0][1].add_artist(circle)
# # 8 inch
# # circle = Ellipse((0.5e-2, 0.05), 1e-3, 0.5e-1, color='r', fill=False)
# # axs[0][1].add_artist(circle)
# # # spinning warb
# circle = Ellipse((0.07e-2, 0.27), 1e-3, 0.5e-1, color='r', fill=False)
# axs[0][1].add_artist(circle)

# plt.savefig("SpinWarblet.pdf", format="pdf")
# plt.show()