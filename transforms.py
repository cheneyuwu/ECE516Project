import numpy as np

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


def gaussian_env(t, tc, dt):
    return 1 / (np.sqrt(np.sqrt(np.pi) * dt)) * np.exp(-1 / 2 * ((t - tc) / dt) ** 2)


def q_chirp(t, tc, fc, c):
    tc = convert_to_nparray(tc)[..., np.newaxis]
    fc = convert_to_nparray(fc)[..., np.newaxis]
    c = convert_to_nparray(c)[..., np.newaxis]
    t = t[np.newaxis, ...]
    assert len(t.shape) == 2
    ret = np.exp(1j * 2 * np.pi * (c * ((t - tc) ** 2) + fc * (t - tc)))
    if ret.shape[0] == 1:
        ret = ret[0, ...]
    return ret


def wave(t, tc, fc):
    return np.exp(1j * 2 * np.pi * (fc * (t - tc)))


def warble(t, tc, fc, fm, bm, pm):
    tc = convert_to_nparray(tc)[..., np.newaxis]
    fc = convert_to_nparray(fc)[..., np.newaxis]
    fm = convert_to_nparray(fm)[..., np.newaxis]
    bm = convert_to_nparray(bm)[..., np.newaxis]
    pm = convert_to_nparray(pm)[..., np.newaxis]
    t = t[np.newaxis, ...]
    assert len(t.shape) == 2
    ret = np.exp(1j * (2 * np.pi * fc * (t - tc) + bm / fm * np.sin(2 * np.pi * fm * (t - tc) + pm)))
    if ret.shape[0] == 1:
        ret = ret[0, ...]
    return ret


def q_chirplet(t, tc, fc, c, dt):
    """
        t - 1d array containing time steps
        dt - delta t = sqrt(2) * sigma
    """
    gs_env = gaussian_env(t, tc, dt)
    cp = q_chirp(t, tc, fc, c)
    return gs_env * cp


def q_chirplet_fbe(t, tc, fb, fe, dt):
    # the boundary of a chirplet is defined to be 3 * sigma
    # dt = sqrt(2) * sigma, so boundary is 3 / sqrt(2) * dt
    #
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


if __name__ == "__main__":

    fs = 1e3
    t = np.linspace(-1, 1, 2000)  # sampling frequency 1e3
    warble_base = warble(t, 0, 0, 10.0, 250, 0)

    sample = 200
    bm = np.linspace(-500, 500, sample)
    fm = np.linspace(0.01, 20, sample)
    fm, bm = np.meshgrid(fm, bm)
    window = warblet(t, 0, 0, fm, bm, 0, np.sqrt(2) / 3)
    res = np.inner(np.conj(window), warble_base)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.contour(fm, bm, np.absolute(res))
    ax.set_xlabel("Modulated Frequency")
    ax.set_ylabel("Amplitude")

    plt.show()

    # Chirplet transform test
    # chirplet_base = q_chirplet(t, 0, 0, 250, np.sqrt(2) / 3)

    # plt.plot(t, chirplet_base.real)
    # plt.show()

    # sample = 200
    # fb = np.linspace(-500, 500, sample)
    # fe = np.linspace(-500, 500, sample)
    # fb, fe = np.meshgrid(fb, fe)
    # window = q_chirplet_fbe(t, 0, fb, fe, np.sqrt(2) / 3)
    # res = np.inner(np.conj(window), chirplet_base)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # ax.contour(fb, fe, np.absolute(res))
    # ax.set_xlabel("base freq / f begin")
    # ax.set_ylabel("slope / f end")

    # plt.show()
    exit()
