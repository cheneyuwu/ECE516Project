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


def signal_momentum(a, b, c, dt):
    return np.sum(np.conj(a) * b * c * dt, axis=-1)


def signal_mean(tf, signal, dt):
    m1 = signal_momentum(signal, tf, signal, dt)
    m0 = signal_momentum(signal, 1.0, signal, dt)
    return m1 / m0


def signal_stddev(tf, signal, dt):
    m2 = signal_momentum(signal, tf ** 2, signal, dt)
    m1 = signal_momentum(signal, tf, signal, dt)
    m0 = signal_momentum(signal, 1.0, signal, dt)
    return np.sqrt((tf.shape[-1] - 1) / tf.shape[-1] * (m2 / m0 - (m1 / m0) ** 2))


fs = 1e3
duration = 5.0  # 5 second
time = np.arange(duration * fs) / float(fs)
dt = 1 / fs
signal_time = transforms.q_chirplet(time, duration / 2, 100, 2.0, np.sqrt(2) * 1.0 / 3.0)
# fft to convert to freq domain
freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=dt))
signal_freq = np.fft.fftshift(np.fft.fft(signal_time), axes=-1)
df = 1 / ((len(freq) - 1) / (freq[-1] - freq[0]))

mu_t = signal_mean(time, signal_time, dt)
mu_f = signal_mean(freq, signal_freq, df)
sigma_t = signal_stddev(time, signal_time, dt)
sigma_f = signal_stddev(freq, signal_freq, df)
print(mu_t, mu_f, sigma_t, sigma_f)

sigma_t = signal_stddev(time, signal_time, dt).real
sigma_f = signal_stddev(freq, signal_freq, df).real

print(sigma_f * sigma_t)
diag_term = np.sqrt((sigma_f ** 2) * (sigma_t ** 2) - (1 / (4 * np.pi)) ** 2)

S = np.array([[sigma_t ** 2, diag_term], [diag_term, sigma_f ** 2]])
w, v = np.linalg.eig(S)

print("Diag Term", diag_term)
print(S)
print(v @ np.diag(w) @ v.T)
# print(w, v)
print(v)
print(v[0][0] / v[1][0])
print(v[1][0] / v[1][1])
exit(0)


def lem(time, signal, niter=5, ncenter=2):
    # infer sample frequency from time
    fs = (len(time) - 1) / (time[-1] - time[0])
    dt = 1 / fs
    # fft to convert to freq domain
    freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=dt))
    df = 1 / ((len(freq) - 1) / (freq[-1] - freq[0]))

    # multiple center case
    # initial guess (there should be multiple parameters)
    mu_t = np.array([1.0, 4.5, 2.5])[:ncenter]
    mu_f = np.array([-400, -390.0, 10.0])[:ncenter]
    sigma_t = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])[:ncenter]
    magnitude = np.array([1.0, 1.0, 1.0])[:ncenter]

    assert mu_t.shape[0] == ncenter
    assert mu_f.shape[0] == ncenter

    result = [
        (mu_t, mu_f, sigma_t, magnitude),
        (mu_t, mu_f, sigma_t, magnitude),
        (mu_t, mu_f, sigma_t, magnitude),
        (mu_t, mu_f, sigma_t, magnitude),
    ]

    for i in range(niter):

        print("Iteration: ", i)

        # E step
        # compute error
        approx_signal = transforms.q_chirplet(time, mu_t, mu_f, 0, np.sqrt(2) * sigma_t)
        magnitude = signal_momentum(approx_signal, 1.0, signal, dt)

        approx_signal = transforms.q_chirplet(time, mu_t, mu_f, 0, np.sqrt(2) * sigma_t)
        approx_signal = magnitude[..., np.newaxis] * approx_signal
        approx_signal = np.sum(approx_signal, axis=0)
        e = signal - approx_signal
        print(np.absolute(e))

        # compute weighted signal:  y.shape == (ncenter, samples)
        y_t = (
            magnitude[..., np.newaxis] * transforms.q_chirplet(time, mu_t, mu_f, 0, np.sqrt(2) * sigma_t) + e / ncenter
        )
        y_f = np.fft.fftshift(np.fft.fft(y_t), axes=-1)

        # M step
        # update duration frequency and variance
        mu_t = signal_mean(time, y_t, dt).real
        mu_f = signal_mean(freq, y_f, df).real
        sigma_t = signal_stddev(time, y_t, dt).real

        result.append((mu_t, mu_f, sigma_t, magnitude))

    # return center and sigma
    return result


fs = 1e3
duration = 5.0  # 5 second
time = np.arange(duration * fs) / float(fs)

true_mu_t = np.array([1.0, 4.0, 2.5])
true_mu_f = np.array([-30, 350.0, 10.0])
signal_time = 0.0
for i in range(2):
    # for i in range(true_mu_t.shape[0]):
    signal_time = signal_time + transforms.q_chirplet(time, true_mu_t[i], true_mu_f[i], 0, 1e-0 * np.sqrt(2) / 3)

freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=1 / fs))
signal_freq = np.fft.fftshift(np.fft.fft(signal_time))

result = lem(time, signal_time, 20, ncenter=2)
mu_t, mu_f, sigma_t, sigma_f = result[-1]
print("Final: ", mu_t, mu_f, sigma_t, sigma_f)


# Create animation
nrow = 1
ncoln = 3
fig, axs = plt.subplots(nrow, ncoln)
fig.suptitle("LEM", fontsize=BIGGER_SIZE)
fig.set_size_inches(ncoln * 5, nrow * 5)
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.85, wspace=0.4, hspace=0.4)


def animate(frames, time, freq, signal_time, true_mu_t, true_mu_f):

    i, (mu_t, mu_f, sigma_t, sigma_f) = frames

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
    signal_time = 0.0
    for i in range(mu_t.shape[0]):
        signal_time = signal_time + transforms.q_chirplet(time, mu_t[i], mu_f[i], 100, np.sqrt(2) * sigma_t[i])
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
    save_count=21,
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
