# 2023 BCI Competition MI Dataset Preprocessing

# Authors: Corey Lin <coreylin2023@outlook.com>
# Date: 2023/06/25 
# License: MIT License

import os, pickle
import numpy as np
from scipy.signal import  iircomb, butter, filtfilt, resample

def read_blocks(folder):
    data = []
    for filename in os.listdir(folder):
        if filename in ["block1.pkl", "block2.pkl", "block3.pkl"]:
            filepath = os.path.join(folder, filename)
            with open(filepath, "rb") as f:
                block = pickle.load(f)
            blockdata = block['data']
            srate = block['srate']
            ch_names = block['ch_names']
        data.append(blockdata)
    return data, srate, ch_names

def extract_samples_and_labels(EEG, fs):
    samples, labels = [], []
    labels = []
    label_signal = EEG[64, :]
    for i, label in enumerate(label_signal):
        if label in [11, 21, 31]:
            start = i
            end = i + 4 * fs
            if end < len(label_signal):
                data = EEG[:64, start:end]
                samples.append(data)
                labels.append(int((label//10) %10))
    samples = np.array(samples)
    labels = np.array(labels)
    return samples, labels

def downsample_and_extract(EEG, fs_old, fs_new, window):
    all_EEG = []
    for sample in EEG:
        length_new = int(len(sample[0]) * fs_new / fs_old)
        EEG_new = np.array([resample(signal, length_new) for signal in sample])
        start = int(window[0] * fs_new)
        end = int(window[1] * fs_new)
        EEG_window = EEG_new[:, start:end]
        all_EEG.append(EEG_window)
    all_EEG = np.array(all_EEG)
    return all_EEG

def get_pre_filter(data, fs=250):    
    f0 = 50
    q = 35
    b, a = iircomb(f0, q, ftype='notch', fs=fs)
    filter_data = filtfilt(b, a, data)
    return filter_data

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):   
    data = get_pre_filter(data)
    nyq = 0.5 * fs 
    low = lowcut / nyq 
    high = highcut / nyq 
    b, a = butter(order, [low, high], btype='band') 
    data_filtered = filtfilt(b, a, data) 
    return data_filtered

def split_eeg(eeg, tags, fs=250, window_width=2, window_step=0.1):
    width = int(window_width * fs)
    step = int(window_step * fs)
    n_samples, n_channels, n_timepoints = eeg.shape
    samples, labels = [], []
    for i in range(n_samples):
        sample, label = eeg[i], tags[i]
        start = 0
        end = width
        while end <= n_timepoints:
            window = sample[:, start:end]
            samples.append(window)
            labels.append(label)
            start += step
            end += step
    return np.float32(samples), np.array(labels)
