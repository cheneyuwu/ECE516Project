import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gaussian_env(t, tc, dt):
    return 1 / (np.sqrt(np.sqrt(np.pi) * dt)) * np.exp(-1 / 2 * ((t - tc) / dt) ** 2)


def q_chirp(t, tc, fc, c):
    return np.exp(1j * 2 * np.pi * (c * ((t - tc) ** 2) + fc * (t - tc)))


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
    fc = (fb + fe) / 2
    c = (fe - fb) / (t[-1] - t[0])
    return q_chirplet(t, tc, fc, c, dt)


def wavelet(t, tc, fc, dt):
    gs_env = gaussian_env(t, tc, dt)
    wv = wave(t, tc, fc)
    return gs_env * wv


if __name__ == "__main__":

    t = np.linspace(-10, 10, 1000)
    chirplet_base = q_chirplet(t, 0, 0, 0, 3)

    fcs = []
    cs = []
    ress = []
    sample = 200
    # for fc in np.linspace(-10, 10, sample):
    #     for c in np.linspace(-10, 10, sample):
    #         window = q_chirplet(t, 0, fc, c, 1)
    #         res = np.vdot(window, chirplet_base)
    #         fcs.append(fc)
    #         cs.append(c)
    #         ress.append(np.absolute(res))


    for fb in np.linspace(-10, 10, sample):
        for fe in np.linspace(-10, 10, sample):
            window = q_chirplet_fbe(t, 0, fb, fe, 3)
            res = np.vdot(window, chirplet_base)
            fcs.append(fb)
            cs.append(fe)
            ress.append(np.absolute(res))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # ax.plot(fcs, cs, ress)

    fcs = np.array(fcs).reshape(sample, sample)
    cs = np.array(cs).reshape(sample, sample)
    ress = np.array(ress).reshape(sample, sample)

    ax.contour(fcs, cs, ress)
    ax.set_xlabel("base freq / f begin")
    ax.set_ylabel("slope / f end")

    plt.show()
