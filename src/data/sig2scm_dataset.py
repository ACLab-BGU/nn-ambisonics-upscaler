import glob
import os

import numpy as np
import torch
import torch.utils.data as data

from src.data.matlab_loaders import load_mat_file


def get_narrowband_signal(x, nfft, freq_bin):
    raise NotImplemented


def load_single_freq(file, freq):
    d = load_mat_file(file)
    freq_bin = np.argmin(np.abs(d['freq'] - freq))
    x = get_narrowband_signal(d['anm'], d['nfft'], freq_bin)
    y = d['R'][freq_bin]
    return x, y


class Dataset(data.Dataset):
    def __init__(self, root, frequency, transform=None, preload=True, dtype=torch.float32, train=True):

        self.dtype = dtype
        self.transform = transform
        self.preload_flag = preload
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'test')
        self.filenames = glob.glob(os.path.join(root, '*.mat'))  # get list of all mat files in the root folder
        assert len(self.filenames) > 0, 'data folder is empty'
        self.len = len(self.filenames)

        self.frequency = frequency
        # self.frequencies = load_free_field_frequencies()  # frequencies of each SCM in the dataset

        if preload:
            self._preload()

    def _preload(self):
        """
        load all dataset to memory
        """
        self.samples = []

        # load all files to memory and form the database
        for fn in self.filenames:
            self.samples.append(load_single_freq(fn, self.frequency))

    def __getitem__(self, item: int):
        """
        Get a sample from the dataset
        """

        # load sample
        if self.preload_flag:
            x, y = self.samples[item]
        else:
            x, y = load_single_freq(self.filenames[item], self.frequency)

        # perform some transformation
        if self.transform:
            x, y = self.transform(x, y)

        norm_x = torch.norm(x)
        x /= norm_x
        y /= norm_x ** 2

        # return sample
        return x, y

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
