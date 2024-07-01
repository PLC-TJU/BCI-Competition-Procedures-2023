"""
TL_Classifier: Transfer Learning Classifier
Author: Pan.LC <coreylin2023@outlook.com>
Date: 2023/6/21
License: MIT License
"""

import os, pickle
import numpy as np
from scipy.signal import resample

# 定义一个函数，用于读取一个文件夹中的block1.pkl，block2.pkl，block3.pkl数据文件
def read_blocks(folder):
    """Read data from block1.pkl, block2.pkl, block3.pkl files in a folder.

    Parameters
    ----------
    folder : str
        The path of the folder that contains the data files.

    Returns
    -------
    data : list
        A list of arrays that contain the data from each file.
    srate : float
        The sampling rate of the data.
    ch_names : list
        A list of strings that contain the channel names of the data.

    Example
    -------
    >>> folder = '/path/to/folder'
    >>> data, srate, ch_names = read_blocks(folder)
    """
    
    # Create an empty list to store the read data
    data = []
    # Traverse the file names in the folder
    for filename in os.listdir(folder):
        # If the file name is one of block1.pkl, block2.pkl, block3.pkl
        if filename in ["block1.pkl", "block2.pkl", "block3.pkl"]:
            # Concatenate the full path of the file
            filepath = os.path.join(folder, filename)
            # Open the file and read the data
            with open(filepath, "rb") as f:
                block = pickle.load(f)
            # Add the data to the list
            blockdata = block['data']
            srate = block['srate']
            ch_names = block['ch_names']
        data.append(blockdata)
    # Return the list
    return data, srate, ch_names


# 定义一个函数，用于从EEG信号中提取样本数据和标签数据
def extract_samples_and_labels(EEG, fs):
    """Extract sample data and label data from EEG signals.

    This function takes an EEG signal array and a sampling frequency as inputs,
    and returns two arrays of sample data and label data as outputs.
    The sample data are segments of 4 seconds from the first 64 channels of the EEG signal,
    corresponding to the labels 11, 21 and 31 in the 65th channel of the EEG signal.
    The label data are integers of 1, 2 and 3, representing the classes of the samples.

    Parameters
    ----------
    EEG : array
        The EEG signal array, with shape (65, n), where n is the number of samples.
        The first 64 channels are the EEG data, and the 65th channel is the label signal.
    fs : int
        The sampling frequency of the EEG signal, in Hz.

    Returns
    -------
    samples : array
        The sample data array, with shape (m, 64, 4*fs), where m is the number of samples.
        Each sample is a segment of 4 seconds from the first 64 channels of the EEG signal.
    labels : array
        The label data array, with shape (m,), where m is the number of samples.
        Each label is an integer of 1, 2 or 3, representing the class of the sample.

    Example
    -------
    >>> EEG = np.random.rand(65,1000)
    >>> fs = 250
    >>> samples, labels = extract_samples_and_labels(EEG, fs)
    """
    
    samples = []
    labels = []
    label_signal = EEG[-1, :]
    
    # check if the label signal is valid
    count1 = np.sum(label_signal == 11)
    count2 = np.sum(label_signal == 21)
    count3 = np.sum(label_signal == 31)
    if count1 != count2 or count2 != count3:
        raise ValueError("Invalid label signal!")
    # check the interval between each label
    interval = np.diff(np.where(np.isin(label_signal, [11, 21, 31, 12, 22, 32, 13, 23, 33]))[0])
    if not np.all(interval >= 950):
        raise ValueError("Invalid label signal!")
    interval = np.diff(np.where(label_signal == 11)[0])
    if not np.all(interval >= 7950):
        raise ValueError("Invalid label signal!")
    
    for i, label in enumerate(label_signal):
        if label in [11, 21, 31]:
            start = i
            end = i + 4 * fs
            if end < len(label_signal):
                # Cut out data, i.e. data from first 64 leads
                data = EEG[:59, start:end]
                samples.append(data)
                labels.append(int((label//10) %10))
    samples = np.array(samples)
    labels = np.array(labels)
    return samples, labels

class LoadData:
    def __init__(self, data_path, fs=250):
        self.data_path = data_path  
        self.fs = fs
        
        subfolders = os.listdir(self.data_path)
        self.subject_list = [int(subfolder.split('S')[-1]) for subfolder in subfolders]
        self.subject_list.sort()

    def get_data(self, subjects: list[int]):
        if not set(subjects).issubset(set(self.subject_list)):
            print(f"dataset: {self.dataset.code}, valid subjects: {self.subject_list}, entered subjects: {subjects}")
            raise ValueError("Invalid subject numbers were entered!")
        
        all_data = []
        all_labels = []
        for subject in subjects:
            subfolder = f"S{subject}"
            subfolder_path = os.path.join(self.data_path, subfolder)
            data, srate, ch_names = read_blocks(subfolder_path)
            EEG = np.concatenate(data, axis=1)
            samples, labels = extract_samples_and_labels(EEG, srate)
            timepoints = samples.shape[-1]
            samples = resample(samples, int(self.fs/srate*timepoints), axis=-1)
            all_data.append(samples)
            all_labels.append(labels)
        return all_data, all_labels


if __name__ == '__main__':
    dataA = LoadData('TrainData/A')
    subjects = dataA.subject_list
    data, label = dataA.get_data(subjects)