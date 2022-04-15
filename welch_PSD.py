from scipy.signal import welch
import numpy as np
import matplotlib as plt
SAMPLING_FREQUENCY = 400
PSD_FREQ = np.array([[0.1, 4], [4, 8], [8, 15], [15, 30], [30, 90], [90, 170]])
def compute_power_spectral_density(self, windowed_signal):
    # Windowed signal of shape [channel][sample] [16 x 12000]
    ret = []

    # Welch parameters
    sliding_window = 512
    overlap = 0.25
    n_overlap = int(sliding_window * overlap)

    # compute psd using Welch method
    freqs, power = welch(windowed_signal, fs=SAMPLING_FREQUENCY,
                                nperseg=sliding_window, noverlap=n_overlap)

    for psd_freq in PSD_FREQ:
        tmp = (freqs >= psd_freq[0]) & (freqs < psd_freq[1])
        ret.append(power[:, tmp].mean(1))

    return (np.log(np.array(ret) / np.sum(ret, axis=0)))

def plot_psd(freqs,psd):
    freqs = freqs[freqs > 0]
    psd = psd[freqs > 0]
    plt.plot(np.log10(freqs), 10 * np.log10(psd.ravel()))#, label=label,color=color