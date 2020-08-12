import glob
import os

import numpy as np
import torch
import torch.utils.data as data

from src.data.matlab_loaders import load_mat_file
from src.utils.audio import torch_stft_nd


def get_narrowband_signal(x, nfft, freq_bin):
    # x is of shape (time, channels)
    x = torch_stft_nd(torch.from_numpy(x), time_dim=0,
                      n_fft=nfft, hop_length=nfft//4, onesided=True, window=torch.hann_window(nfft))
    # x_stft is of shape (channels, freq, time, real/imag)

    # slice to a single freq
    x = x[:, freq_bin, :, :]

    # transform to shape (real/imag, channels, time)
    return x.permute((2, 0, 1))


def load_single_freq(file, freq, cache=None):
    d, cache = load_mat_file(file, cache)
    freq_bin = np.argmin(np.abs(d['freq'] - freq))
    x = get_narrowband_signal(d['anm'], d['nfft'], freq_bin)
    # x is of shape (real/imag, channels, time)

    y = d['R'][freq_bin]
    # y is now a  complex numpy array. convert to a real torch.tensor
    y = torch.stack([torch.from_numpy(np.real(y)), torch.from_numpy(np.imag(y))])

    # y is now of show (2, channels_out, channels_out)
    return x, y, cache


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
        self.vec2mat_cache = None
        if preload:
            self._preload()

    def _preload(self):
        """
        load all dataset to memory
        """
        self.samples = []

        # load all files to memory and form the database
        for i, fn in enumerate(self.filenames):
            print(i/len(self.filenames))
            x, y, self.vec2mat_cache = load_single_freq(fn, self.frequency, self.vec2mat_cache)
            self.samples.append((x, y))

    def __getitem__(self, item: int):
        """
        Get a sample from the dataset
        """

        # load sample
        if self.preload_flag:
            x, y = self.samples[item]
        else:
            x, y, self.vec2mat_cache = load_single_freq(self.filenames[item], self.frequency, self.vec2mat_cache)

        # perform some transformation
        if self.transform:
            x, y = self.transform(x, y)

        # norm_x = torch.norm(x)
        # x /= norm_x
        # y /= norm_x ** 2

        norm_y = torch.norm(y)
        y /= norm_y
        x /= torch.sqrt(norm_y)

        # return sample
        return x.type(self.dtype), y.type(self.dtype)

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
