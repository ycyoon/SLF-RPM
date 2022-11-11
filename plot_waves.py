import numpy as np
import scipy.io
import os
import sys
import argparse
sys.path.append('../')
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy.sparse import spdiags

def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal

def read_wave(file):
    with np.load(file) as f:
        wave = f["wave"].astype(np.float32)
    return wave

def predict_vitals(args):
    fs = args.sampling_rate
    pulse = read_wave(args.wave_path)
    pulse = detrend(np.cumsum(pulse), 100)
    print('pulse', pulse)
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse))
    print('after filtfilt', pulse)

   
    ########## Plot ##################
    plt.subplot(211)
    plt.plot(pulse)
    plt.title('Pulse')   
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wave_path', type=str, help='gts wave path')
    parser.add_argument('--sampling_rate', type=int, default = 30, help='sampling rate of your video')
    args = parser.parse_args()

    predict_vitals(args)

