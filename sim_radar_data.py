import numpy as np
import transforms

fs = 1e3
duration = 1.0  # 5 second
time = np.arange(duration * fs) / float(fs)

# generate some fakedata
data = {}

# Stabbing
mu_t = [duration / 6, duration * 2 / 3]
mu_f = [-50.0, 50.0]
c = [40.0, 150.0]
sigma_t = [np.sqrt(2) * 100 / 3, np.sqrt(2) * 100 / 3]
signal = transforms.q_chirplet(time, mu_t, mu_f, c, sigma_t)
signal[0, int(1 * len(time) / 3):] = 0.0
signal[1, :int(1 * len(time) / 3)] = 0.0
stabbing_signal = np.sum(signal, axis=0)

# Pick Pocket
mu_t = [duration / 4, duration * 3 / 4]
mu_f = [50.0, -120.0]
c = [200.0, -200.0]
sigma_t = [np.sqrt(2) * 100 / 3, np.sqrt(2) * 100 / 3]
signal = transforms.q_chirplet(time, mu_t, mu_f, c, sigma_t)
signal[0, int(1 * len(time) / 3):] = 0.0
signal[1, :int(1 * len(time) / 3)] = 0.0
pickpocket_signal = np.sum(signal, axis=0)

# Start Running
mu_t = [duration / 6, duration / 2, duration * 5 / 6]
mu_f = [-50.0, -120.0, -200.0]
c = [0.0, -280.0, -0.0]
sigma_t = [np.sqrt(2) * 100 / 3, np.sqrt(2) * 100 / 3, np.sqrt(2) * 100 / 3]
signal = transforms.q_chirplet(time, mu_t, mu_f, c, sigma_t)
signal[0, int(1 * len(time) / 3):] = 0.0
signal[1, :int(1 * len(time) / 3)] = 0.0
signal[1, int(2 * len(time) / 3):] = 0.0
signal[2, :int(2 * len(time) / 3)] = 0.0
running_signal = np.sum(signal, axis=0)

mu_t = [duration / 2]
mu_f = [100]
c = [100.0]
sigma_t = [0.3 * np.sqrt(2) / 3]
signal = transforms.q_chirplet(time, mu_t, mu_f, c, sigma_t)
chirplet_signal = np.sum(signal, axis=0)

mu_t = [duration / 2]
mu_f = [100]
sigma_t = [0.3 * np.sqrt(2) / 3]
signal = transforms.warblet(time, mu_t, mu_f, 2.0, 50.0, np.pi / 2, sigma_t)
warblet_signal = np.sum(signal, axis=0)

mu_t = [duration / 2]
mu_f = [100]
c = [100.0]
sigma_t = [0.3 * np.sqrt(2) / 3]
signal = transforms.q_chirp(time, mu_t, mu_f, c)
chirp_signal = np.sum(signal, axis=0)

mu_t = [duration / 2]
mu_f = [100]
sigma_t = [0.3 * np.sqrt(2) / 3]
signal = transforms.warble(time, mu_t, mu_f, 2.0, 70.0, np.pi / 2)
warble_signal = np.sum(signal, axis=0)

# Test 1
mu_t = [duration * 2 / 8, duration * 6 / 8]
mu_f = [-30, 80]
c = [-70.0, -80.0]
sigma_t = [0.2 * np.sqrt(2) / 3, 0.2 * np.sqrt(2) / 3]
signal = transforms.q_chirplet(time, mu_t, mu_f, c, sigma_t)
test2_signal = np.sum(signal, axis=0)

# Test 1
mu_t = [duration / 5, duration * 3 / 8, duration * 4 / 5]
mu_f = [-40.0, 20.0, 50.0]
c = [30.0, 0.0, -20]
sigma_t = [0.1 * np.sqrt(2) / 3, 0.1 * np.sqrt(2) / 3, 0.2 * np.sqrt(2) / 3]
signal = transforms.q_chirplet(time, mu_t, mu_f, c, sigma_t)
test3_signal = np.sum(signal, axis=0)

data["Running"] = {"signal": running_signal, "ncenter": 3}
data["PickPocket"] = {"signal": pickpocket_signal, "ncenter": 2}
data["Stabbing"] = {"signal": stabbing_signal, "ncenter": 2}

data["Test2"] = {"signal": test2_signal, "ncenter": 2}
data["Test3"] = {"signal": test3_signal, "ncenter": 3}

data["Chirplet"] = {"signal": chirplet_signal, "ncenter": 1}
data["Warblet"] = {"signal": warblet_signal, "ncenter": 1}

data["Chirp"] = {"signal": chirp_signal, "ncenter": 1}
data["Warble"] = {"signal": warble_signal, "ncenter": 1}
