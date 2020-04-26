import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy import signal, stats
from scipy.fft import fftshift

import plot
import plot_utils
import transforms


def signal_momentum(a, b, c, dtf, weight=1.0):
  """compute <weight|a*|b|c>"""
  return np.sum(weight * np.conj(a) * b * c * dtf, axis=-1)


def signal_mean(tf, signal, dtf, weight=1.0):
  """compute mu_t and mu_f"""
  m1 = signal_momentum(signal, tf, signal, dtf, weight)
  m0 = signal_momentum(signal, 1.0, signal, dtf, weight)
  return m1 / m0


def signal_stddev(tf, signal, dtf, weight=1.0):
  """compute sigma_t and sigma_f"""
  m2 = signal_momentum(signal, tf**2, signal, dtf, weight)
  m1 = signal_momentum(signal, tf, signal, dtf, weight)
  m0 = signal_momentum(signal, 1.0, signal, dtf, weight)
  return np.sqrt((tf.shape[-1] - 1) / tf.shape[-1] * (m2 / m0 - (m1 / m0)**2))


def mplem(time, signal):
  """the MPLEM algorithm introduced by Cui and Wong
  """
  # sampling frequency and time interval
  fs = (len(time) - 1) / (time[-1] - time[0])
  dt = 1 / fs

  # initialization
  error = np.inf
  curr_signal = signal  # residual
  M = 4  # maximum MP iterations
  N = 100  # maximum LEM iteratons
  results = []  # chirplet parameters to be returned

  for _ in range(M):

    # single chirplet estimation with 1 more chirplet (MP)
    result = mp(time, curr_signal)
    results.append(result)

    # construct new approximated signal
    approx_signal = np.sum([
        a * transforms.q_chirplet(time, mu_t, mu_f, c,
                                  np.sqrt(2) * sigma_t) for a, mu_t, mu_f, sigma_t, _, c in results
    ],
                           axis=0)

    # calculate residual
    curr_signal = signal - approx_signal
    new_mp_error = np.sum(np.absolute(signal_momentum(curr_signal, 1.0, curr_signal, dt)))
    print("After MP current error: ", new_mp_error)

    if error - new_mp_error < 0.01:
      break  # if error change is too small, stop
    error = new_mp_error

    # multiple chirplets refinement (LEM)
    for _ in range(N):
      # choose a lem algorithm (either the one from Cui or from Mann)
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
      print("After LEM current error: ", lem_new_error)

      if lem_new_error >= error:
        break  # if error change is too small, stop
      error = lem_new_error

  return np.array(results).T


def mp(time, signal):
  """1 iteration of MP
  """
  fs = (len(time) - 1) / (time[-1] - time[0])
  dt = 1 / fs

  # construct a family of chirplets
  c = np.linspace(-fs / 2, fs / 2, int(fs / 4) + 1)
  sigma_t = 2**np.arange(3, 8) / 100.0
  tc = np.linspace(time[0], time[-1], int(fs / 200) + 1)
  fc = np.linspace(-fs / 2, fs / 2, int(fs / 10) + 1)
  mp_params = np.array([[i, j] for i in c for j in sigma_t])

  # find the one that best approximates the signal
  mag_best = -np.inf
  for i in tc:
    for j in fc:
      chirp_dict = transforms.q_chirplet(time, i, j, mp_params[:, 0], np.sqrt(2) * mp_params[:, 1])
      mag = signal_momentum(chirp_dict, 1.0, signal, dt)
      loc = np.argmax(np.absolute(mag))
      if mag_best < np.absolute(mag)[loc]:
        mag_best = np.absolute(mag)[loc]
        result = (mag[loc], i, j, mp_params[loc, 1], 0.0, mp_params[loc, 0])

  return result


def lem(time, signal, niter=5, ncenter=2):
  """LEM algorithm proposed by Mann and Haykin
  """
  fs = (len(time) - 1) / (time[-1] - time[0])
  signal_time = signal
  dt = 1 / fs
  # fft to convert to freq domain
  freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=dt), axes=-1)
  signal_freq = np.fft.fftshift(np.fft.fft(signal_time), axes=-1)
  df = 1 / ((len(freq) - 1) / (freq[-1] - freq[0]))

  # set initial guesses (we only support up to 3 centers) -> TUNE THIS!
  if ncenter == 1:
    mu_t = np.array([0.5])
    mu_f = np.array([0.0])
    sigma_t = np.array([np.sqrt(2) * 1 / 3])
    sigma_f = np.array([10.0])
    c = np.array([0.0])
  if ncenter == 2:
    mu_t = np.array([0.4, 0.6])
    mu_f = np.array([-10, 10])
    sigma_t = np.array([0.1 * np.sqrt(2) * 1 / 3, 0.1 * np.sqrt(2) * 1 / 3])
    sigma_f = np.array([20.0, 20.0])
    c = np.array([0.0, 0.0])
  elif ncenter == 3:
    mu_t = np.array([0.1, 0.5, 0.9])
    mu_f = np.array([-50.0, -120.0, -200.0])
    sigma_t = np.array([0.1 * np.sqrt(2) * 1 / 3, 0.3 * np.sqrt(2) * 1 / 3, 0.1 * np.sqrt(2) * 1 / 3])
    sigma_f = np.array([20.0, 20.0, 20.0])
    c = np.array([0.0, 0.0, 0.0])

  results = [(mu_t, mu_f, sigma_t, sigma_f, c)]  # list of estimated parameters from all iterations

  for i in range(niter):

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

    results.append((mu_t, mu_f, sigma_t, sigma_f, c))

  return results


def lem2(time, signal, guesses):
  """LEM algorithm proposed by Mann and Haykin, modified so that it can be used in MPLEM
  """
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
      print("LEM refines the approximation")
      results.append((new_a, new_mu_t, new_mu_f, new_sigma_t, new_sigma_f, new_c))
    else:
      # LEM cannot refine the approximation, return initial guesses
      results.append((a, mu_t, mu_f, sigma_t, sigma_f, c))

  return results


def lem3(time, signal, guesses):
  """LEM algorithm used by Cui and Wong in MPLEM
  """
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

  # compare the approximation before and after refinement, return with better one
  new_signal = np.sum(new_a[..., np.newaxis] * transforms.q_chirplet(time, new_mu_t, new_mu_f, new_c,
                                                                     np.sqrt(2) * new_sigma_t),
                      axis=0)
  new_error = signal_time - new_signal
  new_error = np.sum(np.absolute(signal_momentum(new_error, 1.0, new_error, dt)))

  old_signal = np.sum(a[..., np.newaxis] * transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t), axis=0)
  old_error = signal_time - old_signal
  old_error = np.sum(np.absolute(signal_momentum(old_error, 1.0, old_error, dt)))
  if old_error > new_error:
    results = np.array([new_a, new_mu_t, new_mu_f, new_sigma_t, new_sigma_f, new_c]).T
  else:
    results = np.array([a, mu_t, mu_f, sigma_t, sigma_f, c]).T

  return list(results)
