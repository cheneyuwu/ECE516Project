import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import animation
from scipy import signal, stats
from scipy.fft import fftshift

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


def signal_momentum(a, b, c, dtf, weight=1.0):
    """compute <weight|a*|b|c> """
    return np.sum(weight * np.conj(a) * b * c, axis=-1)


def signal_mean(tf, signal, dtf, weight=1.0):
    m1 = signal_momentum(signal, tf, signal, dtf, weight)
    m0 = signal_momentum(signal, 1.0, signal, dtf, weight)
    return m1 / m0


def signal_stddev(tf, signal, dtf, weight=1.0):
    m2 = signal_momentum(signal, tf ** 2, signal, dtf, weight)
    m1 = signal_momentum(signal, tf, signal, dtf, weight)
    m0 = signal_momentum(signal, 1.0, signal, dtf, weight)
    return np.sqrt((tf.shape[-1] - 1) / tf.shape[-1] * (m2 / m0 - (m1 / m0) ** 2))


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
    mu_t = np.array([2.0, 3.0, 3.5])[:ncenter]
    mu_f = np.array([-20.0, 0.0, 20.0])[:ncenter]
    sigma_t = np.array([1.0, 1.0, 1.0])[:ncenter]
    sigma_f = np.array([10.0, 10.0, 10.0])[:ncenter]
    c = np.array([0.0, 0.0, 0.0])[:ncenter]

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
        sigma_tf = np.sqrt(np.sqrt(sigma_f ** 2 * sigma_t ** 2 - 1 / (4 * np.pi) ** 2))  # note: delta
        S = np.array([[sigma_t ** 2, sigma_tf ** 2], [sigma_tf ** 2, sigma_f ** 2]])
        S = np.swapaxes(S, -1, 0)
        w, v = np.linalg.eig(S)
        c_abs = np.absolute(v[:, 0, 0] / v[:, 1, 0]) / 2

        # reconstruct signal and estimate chirprate
        g_pos_t = transforms.q_chirplet(time, mu_t, mu_f, c_abs, np.sqrt(2) * sigma_t)
        correlation_pos = np.absolute(signal_momentum(g_pos_t, 1.0, signal_time, dt))
        g_neg_t = transforms.q_chirplet(time, mu_t, mu_f, -c_abs, np.sqrt(2) * sigma_t)
        correlation_neg = np.absolute(signal_momentum(g_neg_t, 1.0, signal_time, dt))
        c = np.where(correlation_pos > correlation_neg, c_abs, -c_abs)

        result.append((mu_t, mu_f, sigma_t, sigma_f, c))
        print("Estimation: ", mu_t, mu_f, sigma_t, sigma_f, c)

    return result


fs = 1e3
duration = 5.0  # 5 second
time = np.arange(duration * fs) / float(fs)

true_magnitude = np.array([1.0, 1.0, 2.5])[:2]
true_mu_t = np.array([1.0, 1.0, 2.5])[:2]
true_mu_f = np.array([-30, 350.0, 10.0])[:2]
true_c = np.array([-40, 60.0, 100.0])[:2]
true_sigma_t = np.array([0.5 * np.sqrt(2) / 3, 0.6 * np.sqrt(2) / 3, np.sqrt(2) / 3])[:2]
signal_time = np.sum(
    true_magnitude[..., np.newaxis] * transforms.q_chirplet(time, true_mu_t, true_mu_f, true_c, true_sigma_t), axis=0
)

freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=1 / fs))
signal_freq = np.fft.fftshift(np.fft.fft(signal_time))

result = lem(time, signal_time, 25, ncenter=2)
mu_t, mu_f, sigma_t, sigma_f, c = result[-1]
print("Final: ", mu_t, mu_f, sigma_t, sigma_f, c)


# Create animation
nrow = 1
ncoln = 3
fig, axs = plt.subplots(nrow, ncoln)
fig.suptitle("LEM", fontsize=BIGGER_SIZE)
fig.set_size_inches(ncoln * 5, nrow * 5)
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.85, wspace=0.4, hspace=0.4)


def animate(frames, time, freq, signal_time, true_mu_t, true_mu_f):

    i, (mu_t, mu_f, sigma_t, sigma_f, c) = frames

    # Plot center
    ax = axs[0]
    ax.clear()
    ax.scatter(true_mu_t, true_mu_f, color="g")
    ax.scatter(mu_t, mu_f, color="r")

    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim([freq[0], freq[-1]])
    ax.set_xlabel("Time [Second]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Signal Center (Iteration: {})".format(i))

    # Plot true data
    ax = axs[1]
    ax.clear()
    ax.plot(time, signal_time.real, color="g")
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim([-2, 2])
    ax.set_xlabel("Time [Second]")
    ax.set_ylabel("Amplitude")
    ax.set_title("True Signal (Real)")

    # Plot approximated data
    ax = axs[2]
    ax.clear()
    # generate approximated signal
    signal_time = np.sum(transforms.q_chirplet(time, mu_t, mu_f, c, np.sqrt(2) * sigma_t), axis=0)
    ax.plot(time, signal_time.real, color="r")
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim([-2, 2])
    ax.set_xlabel("Time [Second]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Approximated Signal (Real)")


ani = animation.FuncAnimation(
    fig,
    animate,
    frames=enumerate(result),
    fargs=(time, freq, signal_time, true_mu_t, true_mu_f),
    interval=500,
    save_count=20,
    blit=False,
)
ani.save("LEM_example_3center.mp4")
# plt.show()


# f, t, Sxx = signal.spectrogram(
#     signal_time,
#     return_onesided=False,
#     fs=fs,
#     window=("tukey", 0.25),
#     nperseg=128,
#     noverlap=112,
#     # window=("tukey", 1.0),
#     # nperseg=64,
#     # noverlap=48,
# )
# plt.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0))
# plt.ylabel("Frequency [Hz]")
# plt.xlabel("Time [sec]")
# plt.show()
