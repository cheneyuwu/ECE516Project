"""Simulated data used in this project
"""
import numpy as np
import transforms

fs = 1e3
duration = 1.0
time = np.arange(duration * fs) / float(fs)

# Stabbing
mu_t = [duration / 2, duration / 4, duration / 4 * 3]
mu_f = [-100, 0.0, 150.0]
c = [0.0, 100.0, 200.0]
sigma_t = [np.sqrt(2) * 10 / 3, np.sqrt(2) * 10 / 3, np.sqrt(2) * 10 / 3]
a = [1.0, 1.0, 1.5]
signal = np.array(a)[..., np.newaxis] * transforms.q_chirplet(time, mu_t, mu_f, c, sigma_t)
signal[1, int(1 * len(time) / 2):] = 0.0
signal[2, :int(1 * len(time) / 2)] = 0.0
noise1 = np.random.normal(scale=np.sqrt(0.5), size=time.shape)
noise2 = np.random.normal(scale=np.sqrt(0.5), size=time.shape)
stabbing_signal = np.sum(signal, axis=0) + noise1 + 1j * noise2

# Pickpocket
mu_t = [duration / 2, duration / 4, duration / 4 * 3]
mu_f = [-100, 0.0, -50.0]
c = [0.0, 100.0, -200.0]
sigma_t = [np.sqrt(2) * 10 / 3, np.sqrt(2) * 10 / 3, np.sqrt(2) * 10 / 3]
a = [1.0, 1.5, 1.5]
signal = np.array(a)[..., np.newaxis] * transforms.q_chirplet(time, mu_t, mu_f, c, sigma_t)
signal[1, int(1 * len(time) / 2):] = 0.0
signal[2, :int(1 * len(time) / 2)] = 0.0
noise1 = np.random.normal(scale=np.sqrt(0.5), size=time.shape)
noise2 = np.random.normal(scale=np.sqrt(0.5), size=time.shape)
pickpocket_signal = np.sum(signal, axis=0) + noise1 + 1j * noise2

# Start Walking
mu_t = [duration / 4, duration / 4 * 3]
mu_f = [-80.0, -180.0]
c = [-160.0, -0.0]
sigma_t = [np.sqrt(2) * 2 / 3, np.sqrt(2) * 2 / 3]
a = [1.0, 1.0]
signal = np.array(a)[..., np.newaxis] * transforms.q_chirplet(time, mu_t, mu_f, c, sigma_t)
signal[0, int(1 * len(time) / 2):] = 0.0
signal[1, :int(1 * len(time) / 2)] = 0.0
noise1 = np.random.normal(scale=np.sqrt(1.0), size=time.shape)
noise2 = np.random.normal(scale=np.sqrt(1.0), size=time.shape)
running_signal = np.sum(signal, axis=0) + noise1 + 1j * noise2

# example chirplet
mu_t = [duration / 2]
mu_f = [70]
c = [100.0]
sigma_t = [0.3 * np.sqrt(2) / 3]
signal = transforms.q_chirplet(time, mu_t, mu_f, c, sigma_t)
chirplet_signal = np.sum(signal, axis=0)

# example warblet
mu_t = [duration / 2]
mu_f = [70]
sigma_t = [0.3 * np.sqrt(2) / 3]
signal = transforms.warblet(time, mu_t, mu_f, 2.0, 40.0, np.pi / 2, sigma_t)
warblet_signal = np.sum(signal, axis=0)

# example chirp
mu_t = [duration / 2]
mu_f = [100]
c = [100.0]
sigma_t = [0.3 * np.sqrt(2) / 3]
signal = transforms.q_chirp(time, mu_t, mu_f, c)
chirp_signal = np.sum(signal, axis=0)

# example warbling signal
mu_t = [duration / 2]
mu_f = [100]
sigma_t = [0.3 * np.sqrt(2) / 3]
signal = transforms.warble(time, mu_t, mu_f, 2.0, 70.0, np.pi / 2)
warble_signal = np.sum(signal, axis=0)

# lem test - 2 center
mu_t = [duration * 3 / 8, duration * 4 / 5]
mu_f = [-30, 80]
c = [-70.0, 150.0]
sigma_t = [0.2 * np.sqrt(2) / 3, 0.1 * np.sqrt(2) / 3]
signal = transforms.q_chirplet(time, mu_t, mu_f, c, sigma_t)
test_2center_signal = np.sum(signal, axis=0)

# lem test - 3 center
mu_t = [duration / 5, duration * 3 / 8, duration * 4 / 5]
mu_f = [-40.0, 20.0, 50.0]
c = [30.0, 0.0, -20]
sigma_t = [0.1 * np.sqrt(2) / 3, 0.1 * np.sqrt(2) / 3, 0.2 * np.sqrt(2) / 3]
signal = transforms.q_chirplet(time, mu_t, mu_f, c, sigma_t)
test3_signal = np.sum(signal, axis=0)

# warblet transform simulated data
mu_t = [duration / 2, duration / 2]
mu_f = [0, 0]
fm = [2.0, 3.0]
bm = [200.0, 300]
pm = [0.0, 0.0]
sigma_t = [0.3 * np.sqrt(2) / 3]
signal = transforms.warble(time, mu_t, mu_f, fm, bm, pm)
noise1 = np.random.normal(scale=np.sqrt(0.5), size=time.shape)
noise2 = np.random.normal(scale=np.sqrt(0.5), size=time.shape)
waving_signal = np.sum(signal, axis=0) + noise1 + 1j * noise2

data = {}
data["Start Walking"] = {"signal": running_signal, "ncenter": 3}
data["PickPocket"] = {"signal": pickpocket_signal, "ncenter": 2}
data["Stabbing"] = {"signal": stabbing_signal, "ncenter": 2}
data["Chirplet"] = {"signal": chirplet_signal, "ncenter": 1}
data["Warblet"] = {"signal": warblet_signal, "ncenter": 1}
data["Chirp"] = {"signal": chirp_signal, "ncenter": 1}
data["Warble"] = {"signal": warble_signal, "ncenter": 1}
data["Visualize LEM"] = {"signal": test_2center_signal, "ncenter": 2}
data["Waving"] = {"signal": waving_signal}