import numpy as np

import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt


def convert_to_nparray(v):
  v = np.asarray(v)
  if v.shape == ():
    v = v.reshape(1)
    return v
  return v


def is_sequence(obj):
  return isinstance(obj, (collections.Sequence, np.ndarray))


def gaussian_env(t, tc, dt):
  is_seq = is_sequence(tc) | is_sequence(dt)
  tc = convert_to_nparray(tc)[..., np.newaxis]
  dt = convert_to_nparray(dt)[..., np.newaxis]
  t = t[np.newaxis, ...]
  ret = 1 / (np.sqrt(np.sqrt(np.pi) * dt)) * np.exp(-1 / 2 * ((t - tc) / dt)**2)
  if not is_seq:
    ret = ret[0, ...]
  return ret


def q_chirp(t, tc, fc, c):
  is_seq = is_sequence(tc) | is_sequence(fc) | is_sequence(c)
  tc = convert_to_nparray(tc)[..., np.newaxis]
  fc = convert_to_nparray(fc)[..., np.newaxis]
  c = convert_to_nparray(c)[..., np.newaxis]
  t = t[np.newaxis, ...]
  assert len(t.shape) == 2
  ret = np.exp(1j * 2 * np.pi * (c * ((t - tc)**2) + fc * (t - tc)))
  if not is_seq:
    ret = ret[0, ...]
  return ret


def wave(t, tc, fc):
  return np.exp(1j * 2 * np.pi * (fc * (t - tc)))


def warble(t, tc, fc, fm, bm, pm):
  is_seq = is_sequence(tc) | is_sequence(fc) | is_sequence(fm) | is_sequence(bm) | is_sequence(pm)
  tc = convert_to_nparray(tc)[..., np.newaxis]
  fc = convert_to_nparray(fc)[..., np.newaxis]
  fm = convert_to_nparray(fm)[..., np.newaxis]
  bm = convert_to_nparray(bm)[..., np.newaxis]
  pm = convert_to_nparray(pm)[..., np.newaxis]
  t = t[np.newaxis, ...]
  assert len(t.shape) == 2
  ret = np.exp(1j * (2 * np.pi * fc * (t - tc) + bm / fm * np.sin(2 * np.pi * fm * (t - tc) + pm)))
  if not is_seq:
    ret = ret[0, ...]
  return ret


def q_chirplet(t, tc, fc, c, dt):
  gs_env = gaussian_env(t, tc, dt)
  cp = q_chirp(t, tc, fc, c)
  return gs_env * cp


def q_chirplet_fbe(t, tc, fb, fe, dt):
  c = (fe - fb) / 2 / (3 * dt / np.sqrt(2))
  fc = (fe + fb) / 2
  return q_chirplet(t, tc, fc, c, dt)


def wavelet(t, tc, fc, dt):
  gs_env = gaussian_env(t, tc, dt)
  wv = wave(t, tc, fc)
  return gs_env * wv


def warblet(t, tc, fc, fm, bm, pm, dt):
  gs_env = gaussian_env(t, tc, dt)
  wb = warble(t, tc, fc, fm, bm, pm)
  return gs_env * wb
