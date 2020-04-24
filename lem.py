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
  return np.sum(weight * np.conj(a) * b * c * dtf, axis=-1)


def signal_mean(tf, signal, dtf, weight=1.0):
  m1 = signal_momentum(signal, tf, signal, dtf, weight)
  m0 = signal_momentum(signal, 1.0, signal, dtf, weight)
  return m1 / m0


def signal_stddev(tf, signal, dtf, weight=1.0):
  m2 = signal_momentum(signal, tf**2, signal, dtf, weight)
  m1 = signal_momentum(signal, tf, signal, dtf, weight)
  m0 = signal_momentum(signal, 1.0, signal, dtf, weight)
  return np.sqrt((tf.shape[-1] - 1) / tf.shape[-1] * (m2 / m0 - (m1 / m0)**2))


def mplem(time, signal):
  fs = (len(time) - 1) / (time[-1] - time[0])
  dt = 1 / fs

  results = []

  # R^0f
  error = np.inf
  curr_signal = signal  # approximated results

  for _ in range(4):

    # single chirplet estimation with 1 more chirplet
    result = matching_pursuit(time, curr_signal)
    results.append(result)

    # construct new approximated signal
    approx_signal = np.sum([
        a * transforms.q_chirplet(time, mu_t, mu_f, c,
                                  np.sqrt(2) * sigma_t) for a, mu_t, mu_f, sigma_t, _, c in results
    ],
                           axis=0)
    curr_signal = signal - approx_signal
    new_mp_error = np.sum(np.absolute(signal_momentum(curr_signal, 1.0, curr_signal, dt)))
    print("After MP current error: ", new_mp_error)

    if error - new_mp_error < 0.0001:
      break

    error = new_mp_error

    # multiple chirplets refinement
    while True:
      # choose a lem algorithm
      results = lem3(time, signal, np.array(results).T)
      results = lem2(time, signal, results)

      # construct new approximated signal
      approx_signal = np.sum([
          a * transforms.q_chirplet(time, mu_t, mu_f, c,
                                    np.sqrt(2) * sigma_t) for a, mu_t, mu_f, sigma_t, _, c in results
      ],
                             axis=0)
      curr_signal = signal - approx_signal
      lem_new_error = np.sum(np.absolute(signal_momentum(curr_signal, 1.0, curr_signal, dt)))
      if lem_new_error >= error:
        break

      error = lem_new_error
      print("After LEM current error: ", lem_new_error)

  results = np.array(results).T

  return results


def lem3(time, signal, guesses):
  # infer sample frequency from time
  fs = (len(time) - 1) / (time[-1] - time[0])
  signal_time = signal
  dt = 1 / fs
  # fft to convert to freq domain
  freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=dt), axes=-1)
  signal_freq = np.fft.fftshift(np.fft.fft(signal_time), axes=-1)
  df = 1 / ((len(freq) - 1) / (freq[-1] - freq[0]))

  # set initial guess
  a, mu_t, mu_f, sigma_t, sigma_f, c = guesses

  # compute weight in time
  c_t = a[..., np.newaxis] * transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t)
  b_t = np.conj(c_t) * c_t / np.sum((np.conj(c_t) * c_t), axis=0, keepdims=True)
  # update mu, sigma in time
  new_mu_t = signal_mean(time, signal_time, dt, b_t).real
  new_sigma_t = signal_stddev(time, signal_time, dt, b_t).real

  # compute weight in freq
  c_f = a[..., np.newaxis] * np.fft.fftshift(np.fft.fft(c_t), axes=-1)
  b_f = np.conj(c_f) * c_f / np.sum((np.conj(c_f) * c_f), axis=0, keepdims=True)
  # update mu, sigma in freq
  new_mu_f = signal_mean(freq, signal_freq, df, b_f).real
  new_sigma_f = signal_stddev(freq, signal_freq, df, b_f).real

  # compute correlation between uncertainty in time and frequency
  sigma_tf_pow4 = new_sigma_f**2 * new_sigma_t**2 - 1 / (4 * np.pi)**2
  sigma_tf = np.sqrt(np.sqrt(np.where(sigma_tf_pow4 > 0.0, sigma_tf_pow4, np.zeros_like(sigma_tf_pow4))))
  S = np.array([[new_sigma_t**2, sigma_tf**2], [sigma_tf**2, new_sigma_f**2]])
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
  new_c = best_perm * c_abs

  new_chirplets = transforms.q_chirplet(time, new_mu_t, new_mu_f, new_c, np.sqrt(2) * new_sigma_t)
  new_a = signal_momentum(new_chirplets, 1.0, signal_time, dt)

  # compare
  new_signal = np.sum(new_a[..., np.newaxis] * transforms.q_chirplet(time, new_mu_t, new_mu_f, new_c,
                                                                     np.sqrt(2) * new_sigma_t),
                      axis=0)
  new_error = signal_time - new_signal
  new_error = np.sum(np.absolute(signal_momentum(new_error, 1.0, new_error, dt)))

  old_signal = np.sum(a[..., np.newaxis] * transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t), axis=0)
  old_error = signal_time - old_signal
  old_error = np.sum(np.absolute(signal_momentum(old_error, 1.0, old_error, dt)))

  print("old error: ", old_error)
  print("new error: ", new_error)

  if old_error > new_error:
    results = np.array([new_a, new_mu_t, new_mu_f, new_sigma_t, new_sigma_f, new_c]).T
  else:
    results = np.array([a, mu_t, mu_f, sigma_t, sigma_f, c]).T

  return list(results)


def lem2(time, signal, guesses):

  fs = (len(time) - 1) / (time[-1] - time[0])
  dt = 1 / fs
  # fft to convert to freq domain
  freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=dt), axes=-1)
  df = 1 / ((len(freq) - 1) / (freq[-1] - freq[0]))

  approx_signal = np.sum([
      a * transforms.q_chirplet(time, mu_t, mu_f, c,
                                np.sqrt(2) * sigma_t) for a, mu_t, mu_f, sigma_t, _, c in guesses
  ],
                         axis=0)
  total_error = signal - approx_signal
  error = total_error / len(guesses)

  results = []
  for a, mu_t, mu_f, sigma_t, sigma_f, c in guesses:
    signal_time = a * transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t) + error
    signal_freq = np.fft.fftshift(np.fft.fft(signal_time), axes=-1)

    # update mu, sigma in time
    new_mu_t = signal_mean(time, signal_time, dt).real
    new_sigma_t = signal_stddev(time, signal_time, dt).real

    # update mu, sigma in freq
    new_mu_f = signal_mean(freq, signal_freq, df).real
    new_sigma_f = signal_stddev(freq, signal_freq, df).real

    # compute correlation between uncertainty in time and frequency
    sigma_tf_pow4 = new_sigma_f**2 * new_sigma_t**2 - 1 / (4 * np.pi)**2
    sigma_tf = np.sqrt(np.sqrt(np.where(sigma_tf_pow4 > 0.0, sigma_tf_pow4, np.zeros_like(sigma_tf_pow4))))
    S = np.array([[new_sigma_t**2, sigma_tf**2], [sigma_tf**2, new_sigma_f**2]])
    S = np.swapaxes(S, -1, 0)
    w, v = np.linalg.eig(S)
    c_abs = np.absolute(v[0, 0] / v[1, 0]) / 2 if not v[1, 0] == 0.0 else np.inf
    c_abs = np.where(np.isposinf(c_abs), np.zeros_like(c_abs), c_abs)

    # reconstruct signal and estimate chirprate
    perms = [-1.0, 1.0]
    best_correlation = -np.inf
    for perm in perms:
      g_t = np.sum(transforms.q_chirplet(time, new_mu_t, new_mu_f, perm * c_abs, np.sqrt(2) * new_sigma_t), axis=0)
      correlation = np.absolute(signal_momentum(g_t, 1.0, signal_time, dt))
      if correlation > best_correlation:
        best_correlation = correlation
        best_perm = perm
    new_c = best_perm * c_abs

    new_chirplet = transforms.q_chirplet(time, new_mu_t, new_mu_f, new_c, np.sqrt(2) * new_sigma_t)
    new_a = signal_momentum(new_chirplet, 1.0, signal, dt)

    new_signal = new_chirplet * new_a

    new_error = signal_time - new_signal

    error = np.sum(np.absolute(error))
    new_error = np.sum(np.absolute(new_error))

    if new_error < error:
      print("Lem is refining the signal!!!")
      results.append((new_a, new_mu_t, new_mu_f, new_sigma_t, new_sigma_f, new_c))
    else:
      results.append((a, mu_t, mu_f, sigma_t, sigma_f, c))

  return results


def matching_pursuit(time, signal):
  """
  """
  fs = (len(time) - 1) / (time[-1] - time[0])
  dt = 1 / fs

  c = np.linspace(-fs / 2, fs / 2, int(fs / 4))
  # print("c\n", c)
  sigma_t = 2**np.arange(3, 8) / 100.0
  # print("sigma_t\n", sigma_t)
  tc = np.linspace(time[0], time[-1], int(fs / 100))
  fc = np.linspace(-fs / 2, fs / 2, int(fs / 12))

  tc = [0.25, 0.5, 0.75]
  # fc = [-80, -180]
  # c = [0.0, -160]

  mp_params = np.array([[i, j] for i in c for j in sigma_t])

  mag_best = -np.inf
  for i in tc:
    for j in fc:
      chirp_dict = transforms.q_chirplet(time, i, j, mp_params[:, 0], np.sqrt(2) * mp_params[:, 1])

      mag = signal_momentum(chirp_dict, 1.0, signal, dt)
      loc = np.argmax(np.absolute(mag))
      if mag_best < mag[loc]:
        mag_best = mag[loc]
        result = (mag[loc], i, j, mp_params[loc, 1], 0.0, mp_params[loc, 0])

  return result

  # # mp_params -> this only contains tc and c
  # mp_params = np.empty((0, 2))  # the 4 parameters to be estimated

  # N = len(signal)  # signal size
  # i_0 = 0  # the first level to chirp/rotate logons
  # a = 2  # radix of scales
  # gamma = np.arange(0, N, 1)  # signal range
  # T = N  # normalized time range
  # F = 2 * np.pi  # normalized frequency range
  # D = np.ceil(0.5 * np.log(N) / np.log(a))  # number of levels of decomposition

  # k = np.arange(0, D - i_0, 1)  # scale index
  # M = N**2 * (i_0 + np.sum(4 * (a**(2 * k)) - 1))

  # for k_i in k:
  #   m_k = 4 * (a**(2 * k_i)) - 1  # number of chirplets at each scale
  #   m = np.arange(0, m_k + 1, 1)
  #   alpha_m = np.arctan(m / (a**(2 * k_i)))
  #   dt = a**(2 * k_i) / fs
  #   c = m / dt
  #   mp_params = np.concatenate([mp_params, np.array([c, dt * np.ones_like(c)]).T], axis=0)
  #   # print(mp_params)
  # time = time * 1000
  # chirp_dict = transforms.q_chirplet(time, 0.5 * fs, -120.0 / fs, mp_params[:, 0], mp_params[:, 1])


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

    result.append((mu_t, mu_f, sigma_t, sigma_f, c))
    print("Estimation: ", mu_t, mu_f, sigma_t, sigma_f, c)

  return result


if __name__ == "__main__":

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

  result = mplem(time, signal_time)
  # result = [matching_pursuit(time, signal_time)]
  # result = lem(time, signal_time, 400, ncenter=ncenter)

  # mu_t, mu_f, sigma_t, sigma_f, c = result[-1]
  # print("Final: ", mu_t, mu_f, sigma_t, sigma_f, c)

  # Create animation
  nrow = 3
  ncoln = 3
  fig, axs = plt.subplots(nrow, ncoln)
  fig.suptitle("LEM")
  fig.set_size_inches(ncoln * 5, nrow * 5)
  fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.85, wspace=0.4, hspace=0.4)

  # Plot Spectrogram
  plot.plot_spectrogram(axs[0][0], signal_time, fs)
  plot.plot_signal_real(axs[1][0], time, signal_time, fs)
  plot.plot_signal_imag(axs[2][0], time, signal_time, fs)

  # Plot Spectrogram
  a, mu_t, mu_f, sigma_t, sigma_f, c = result

  fake_signal = np.sum(a[..., np.newaxis] * transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t), axis=0)
  # fake_signal = a * transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t)
  plot.plot_spectrogram(axs[0][1], fake_signal, fs)
  plot.plot_signal_real(axs[1][1], time, fake_signal, fs)
  plot.plot_signal_imag(axs[2][1], time, fake_signal, fs)

  # # Plot Spectrogram
  # mu_t, mu_f, sigma_t, sigma_f, c = result[-1]
  # fake_signal = np.sum(transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t), axis=0)
  # plot.plot_spectrogram(axs[0][2], fake_signal, fs)
  # plot.plot_signal_real(axs[1][2], time, fake_signal, fs)
  # plot.plot_signal_imag(axs[2][2], time, fake_signal, fs)

  # Generate Frequency Frequency Plot

  plt.show()
  # plt.savefig(data_name + ".pdf", format="pdf")

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
