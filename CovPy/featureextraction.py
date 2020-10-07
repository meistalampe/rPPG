import h5py
import math
import numpy as np
import scipy
from scipy import signal

from scipy import fftpack

import matplotlib.pyplot as plt


def main():
    # open data file
    with h5py.File('RoiValues.h5', 'r') as f:
        dataset = f['ROIVALUES'].value
        timeline = f['TIMELINE'].value

    with h5py.File('Results_ThermalData_18_06_2020_13_19_36.h5', 'r') as f:
        print(list(f.keys()))
        dataseta = f['PERIORBITAL'].value
        datasetb = f['MAXILLARY'].value
        datasetc = f['NOSETIP'].value
        datasetd = f['TIMESTAMPS'].value

    plt.plot(dataset)
    plt.show()

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y

    f_min = 0.1
    f_max = 0.45
    fs = int(1/(timeline[1]-timeline[0]))
    # avg = np.mean(dataset)
    # temp_avg_filt = [a - avg for a in dataset]
    # filtered = butter_bandpass_filter(dataset, f_min, f_max, fs, order=2)
    b, a = butter_bandpass(f_min, f_max, fs, order=2)
    filtered = signal.filtfilt(b, a, dataset)
    plt.plot(timeline, filtered)
    plt.show()

    N = 150
    ma = np.convolve(filtered, np.ones((N,)) / N, mode='valid')

    # peaks, _ = signal.find_peaks(ma, prominence=0.85)
    # plt.plot(ma)
    # plt.plot(peaks, ma[peaks], 'x')
    # plt.show()
    #
    # bi = np.diff(peaks) / fs
    # br = 60 / bi
    #
    # plt.plot(br)
    # plt.show()

    # y = fftpack.fft(filtered)
    # freqs = fftpack.fftfreq(len(filtered)) * fs
    #
    # fr = freqs[0:int(fs/2)]
    # br = fr[np.where(y == np.amax(y[0:int(fs/2)]))[0]]
    # print(br)


if __name__ == '__main__':
    main()
