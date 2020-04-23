import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import animation
from scipy import signal, stats
from scipy.fft import fftshift
import itertools

import transforms
import plot_utils
import plot


def signal_momentum(a, b, c, dtf, weight=1.0):
  """compute <weight|a*|b|c> """
  return np.sum(weight * np.conj(a) * b * c, axis=-1)


def signal_mean(tf, signal, dtf, weight=1.0):
  m1 = signal_momentum(signal, tf, signal, dtf, weight)
  m0 = signal_momentum(signal, 1.0, signal, dtf, weight)
  return m1 / m0


def signal_stddev(tf, signal, dtf, weight=1.0):
  m2 = signal_momentum(signal, tf**2, signal, dtf, weight)
  m1 = signal_momentum(signal, tf, signal, dtf, weight)
  m0 = signal_momentum(signal, 1.0, signal, dtf, weight)
  return np.sqrt((tf.shape[-1] - 1) / tf.shape[-1] * (m2 / m0 - (m1 / m0)**2))


def matching_pursuit(time, signal):
  """
  """
  N = len(signal)  # signal size
  i_0 = 1  # the first level to chirp/rotate logons
  a = 2  # radix of scales
  gamma = np.arange(0, N, 1)  # signal range
  T = N  # normalized time range
  F = 2 * np.pi  # normalized frequency range
  D = np.ceil(0.5 * np.log(N) / np.log(a))  # number of levels of decomposition
  k = np.arange(0, D - i_0, 1)  # scale index
  m_k = 4 * (a**(2 * k)) - 1  # number of chirplets at each scale
  M = N**2 * (i_0 + np.sum(m_k))
  m = np.arange(0, m_k, 1)
  alpha_m = np.arctan(m / (a**(2 * k)))
  exit()


def lem(time, signal, niter=5, ncenter=2):
  # infer sample frequency from time
  fs = (len(time) - 1) / (time[-1] - time[0])
  signal_time = signal
  dt = 1 / fs
  # fft to convert to freq domain
  freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=dt), axes=-1)
  signal_freq = np.fft.fftshift(np.fft.fft(signal_time), axes=-1)
  df = 1 / ((len(freq) - 1) / (freq[-1] - freq[0]))

  # set initial guess
  if ncenter == 1:
    mu_t = np.array([0.5])
    mu_f = np.array([0.0])
    sigma_t = np.array([np.sqrt(2) * 1 / 3])
    sigma_f = np.array([10.0])
    c = np.array([0.0])
  if ncenter == 2:
    mu_t = np.array([0.3, 0.8])
    # mu_f = np.array([-10.0, 10.0]) # 2 center toy example
    # mu_f = np.array([-100.0, 100.0])  # stabbing
    # mu_f = np.array([100.0, -100.0])  # pick pocket
    # mu_f = np.array([-50.0, -100.0])  # pick pocket
    sigma_t = np.array([0.1 * np.sqrt(2) * 1 / 3, 0.1 * np.sqrt(2) * 1 / 3])
    sigma_f = np.array([20.0, 20.0])
    c = np.array([0.0, 0.0])
  elif ncenter == 3:
    mu_t = np.array([0.1, 0.5, 0.9])
    mu_f = np.array([-50.0, -120.0, -200.0])
    sigma_t = np.array([0.1 * np.sqrt(2) * 1 / 3, 0.3 * np.sqrt(2) * 1 / 3, 0.1 * np.sqrt(2) * 1 / 3])
    sigma_f = np.array([20.0, 20.0, 20.0])
    c = np.array([0.0, 0.0, 0.0])

  assert mu_t.shape[0] == ncenter
  assert mu_f.shape[0] == ncenter

  result = [(mu_t, mu_f, sigma_t, sigma_f, c)]

  for i in range(niter):
    print("Iteration: ", i)

    # compute weight in time
    c_t = transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t)
    b_t = np.conj(c_t) * c_t / np.sum((np.conj(c_t) * c_t), axis=0, keepdims=True)
    # update mu, sigma in time
    mu_t = signal_mean(time, signal_time, dt, b_t).real
    sigma_t = signal_stddev(time, signal_time, dt, b_t).real

    # compute weight in freq
    c_f = np.fft.fftshift(np.fft.fft(c_t), axes=-1)
    b_f = np.conj(c_f) * c_f / np.sum((np.conj(c_f) * c_f), axis=0, keepdims=True)
    # update mu, sigma in freq
    mu_f = signal_mean(freq, signal_freq, df, b_f).real
    sigma_f = signal_stddev(freq, signal_freq, df, b_f).real

    # compute correlation between uncertainty in time and frequency
    sigma_tf_pow4 = sigma_f**2 * sigma_t**2 - 1 / (4 * np.pi)**2
    sigma_tf = np.sqrt(np.sqrt(np.where(sigma_tf_pow4 > 0.0, sigma_tf_pow4, np.zeros_like(sigma_tf_pow4))))
    S = np.array([[sigma_t**2, sigma_tf**2], [sigma_tf**2, sigma_f**2]])
    S = np.swapaxes(S, -1, 0)
    w, v = np.linalg.eig(S)
    c_abs = np.absolute(v[:, 0, 0] / v[:, 1, 0]) / 2
    c_abs = np.where(np.isposinf(c_abs), np.zeros_like(c_abs), c_abs)

    # reconstruct signal and estimate chirprate
    perms = np.array(list(itertools.product([-1.0, 1.0], repeat=len(c_abs))))
    best_correlation = -np.inf
    for perm in perms:
      g_t = np.sum(transforms.q_chirplet(time, mu_t, mu_f, perm * c_abs, np.sqrt(2) * sigma_t), axis=0)
      correlation = np.absolute(signal_momentum(g_t, 1.0, signal_time, dt))
      if correlation > best_correlation:
        best_correlation = correlation
        best_perm = perm
    c = best_perm * c_abs

    # g_pos_t = transforms.q_chirplet(time, mu_t, mu_f, c_abs, np.sqrt(2) * sigma_t)
    # b_t = np.conj(g_pos_t) * g_pos_t / np.sum((np.conj(g_pos_t) * g_pos_t), axis=0, keepdims=True)
    # correlation_pos = np.absolute(signal_momentum(g_pos_t, 1.0, signal_time, dt, b_t))
    # g_neg_t = transforms.q_chirplet(time, mu_t, mu_f, -c_abs, np.sqrt(2) * sigma_t)
    # b_t = np.conj(g_neg_t) * g_neg_t / np.sum((np.conj(g_neg_t) * g_neg_t), axis=0, keepdims=True)
    # correlation_neg = np.absolute(signal_momentum(g_neg_t, 1.0, signal_time, dt, b_t))
    # c = np.where(correlation_pos > correlation_neg, c_abs, -c_abs)

    result.append((mu_t, mu_f, sigma_t, sigma_f, c))
    print("Estimation: ", mu_t, mu_f, sigma_t, sigma_f, c)

  return result


import sim_radar_data

data_name = "Running"
data = sim_radar_data.data[data_name]

fs = 1e3
duration = 1.0  # 5 second
time = np.arange(duration * fs) / float(fs)
signal_time = data["signal"]
freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=1 / fs))
signal_freq = np.fft.fftshift(np.fft.fft(signal_time))

ncenter = data["ncenter"]

result = lem(time, signal_time, 400, ncenter=ncenter)

mu_t, mu_f, sigma_t, sigma_f, c = result[-1]
print("Final: ", mu_t, mu_f, sigma_t, sigma_f, c)

# Create animation
nrow = 3
ncoln = 3
fig, axs = plt.subplots(nrow, ncoln)
fig.suptitle("LEM", fontsize=BIGGER_SIZE)
fig.set_size_inches(ncoln * 5, nrow * 5)
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.85, wspace=0.4, hspace=0.4)

# Plot Spectrogram
plot.plot_spectrogram(axs[0][0], signal_time, fs)
plot.plot_signal_real(axs[1][0], time, signal_time, fs)
plot.plot_signal_imag(axs[2][0], time, signal_time, fs)

# Plot Spectrogram
mu_t, mu_f, sigma_t, sigma_f, c = result[0]
fake_signal = np.sum(transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t), axis=0)
plot.plot_spectrogram(axs[0][1], fake_signal, fs)
plot.plot_signal_real(axs[1][1], time, fake_signal, fs)
plot.plot_signal_imag(axs[2][1], time, fake_signal, fs)

# Plot Spectrogram
mu_t, mu_f, sigma_t, sigma_f, c = result[-1]
fake_signal = np.sum(transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t), axis=0)
plot.plot_spectrogram(axs[0][2], fake_signal, fs)
plot.plot_signal_real(axs[1][2], time, fake_signal, fs)
plot.plot_signal_imag(axs[2][2], time, fake_signal, fs)

# Plot true data

# Plot approximated data
plt.show()
plt.savefig(data_name + ".pdf", format="pdf")

exit()


def animate(frames, time, freq, true_signal):

  i, (mu_t, mu_f, sigma_t, sigma_f, c) = frames
  fake_signal = np.sum(transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t), axis=0)

  # Plot Spectrogram
  ax = axs[0][0]
  ax.clear()
  f, t, Sxx = signal.spectrogram(
      true_signal,
      return_onesided=False,
      fs=fs,
      window=("tukey", 0.25),
      nperseg=128,
      noverlap=112,
  )
  ax.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0))
  ax.set_ylabel("Frequency [Hz]")
  ax.set_xlabel("Time [sec]")

  # Plot Spectrogram
  ax = axs[0][1]
  ax.clear()
  f, t, Sxx = signal.spectrogram(
      fake_signal,
      return_onesided=False,
      fs=fs,
      window=("tukey", 0.25),
      nperseg=128,
      noverlap=112,
  )
  ax.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0))
  ax.set_ylabel("Frequency [Hz]")
  ax.set_xlabel("Time [sec]")

  # Plot true data
  ax = axs[1][0]
  ax.clear()
  ax.plot(time, true_signal.real, color="g")
  ax.set_xlim([time[0], time[-1]])
  ax.set_ylim([-2, 2])
  ax.set_xlabel("Time [Second]")
  ax.set_ylabel("Amplitude")
  ax.set_title("True Signal (Real)")

  # Plot approximated data
  ax = axs[1][1]
  ax.clear()
  ax.plot(time, fake_signal.real, color="r")
  ax.set_xlim([time[0], time[-1]])
  ax.set_ylim([-2, 2])
  ax.set_xlabel("Time [Second]")
  ax.set_ylabel("Amplitude")
  ax.set_title("Approximated Signal (Real)")


ani = animation.FuncAnimation(
    fig,
    animate,
    frames=enumerate(result),
    fargs=(time, freq, signal_time),
    interval=10,
    save_count=20,
    blit=False,
)
plt.show()
ani.save("LEM.mp4")
