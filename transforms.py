import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def q_chirp_fbe(t, tc, fb, fe):
    c = (fe - fb) / 2
    fc = (fe + fb) / 2
    return q_chirp(t, tc, fc, c)


def wave(t, tc, fc):
    return np.exp(1j * 2 * np.pi * (fc * (t - tc)))


def q_chirplet(t, tc, fc, c, dt):
    """
        t - 1d array containing time steps
        dt - delta t = sqrt(2) * sigma
    """
    gs_env = gaussian_env(t, tc, dt)
    cp = q_chirp(t, tc, fc, c)
    return gs_env * cp


def q_chirplet_fbe(t, tc, fb, fe, dt):
    c = (fe - fb) / 2
    fc = (fe + fb) / 2
    return q_chirplet(t, tc, fc, c, dt)


def wavelet(t, tc, fc, dt):
    gs_env = gaussian_env(t, tc, dt)
    wv = wave(t, tc, fc)
    return gs_env * wv


if __name__ == "__main__":

    t = np.linspace(-10, 10, 1000)
    chirplet_base = q_chirplet(t, 0, 0, 0, 3)

    sample = 200
    fc = np.linspace(-100, 100, sample)
    c = np.linspace(-100, 100, sample)
    fc, c = np.meshgrid(fc, c)
    window = q_chirplet(t, 0, fc, c, 1)
    res = np.inner(np.conj(window), chirplet_base)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.contour(fc, c, np.absolute(res))
    ax.set_xlabel("base freq / f begin")
    ax.set_ylabel("slope / f end")

    plt.show()
