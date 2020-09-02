import glob
import os

import numpy as np
import torch
import torch.utils.data as data

from src.data.matlab_loaders import load_mat_file
from src.utils.audio import torch_stft_nd


def get_sliced_stft(x, nfft, freq_bins_range):
    # x is of shape (time, channels)
    x = torch_stft_nd(torch.from_numpy(x), time_dim=0,
                      n_fft=nfft, hop_length=nfft // 4, onesided=True, window=torch.hann_window(nfft))
    # x_stft is of shape (channels, freq, time, real/imag)

    # slice to a single freq
    x = x[:, freq_bins_range, :, :]

    # transform to shape (real/imag, channels, freq, time)
    return x.permute((3, 0, 1, 2))


def get_narrowband_signal(x, nfft, freq_bin):
    # x is of shape (time, channels)
    x = torch_stft_nd(torch.from_numpy(x), time_dim=0,
                      n_fft=nfft, hop_length=nfft // 4, onesided=True, window=torch.hann_window(nfft))
    # x_stft is of shape (channels, freq, time, real/imag)

    # slice to a single freq
    x = x[:, freq_bin, :, :]

    # transform to shape (real/imag, channels, time)
    return x.permute((2, 0, 1))


def load_single_freq(file, freq, cache=None, sh_order_sig=float("inf"), sh_order_scm=float("inf"),
                     time_len_sig=float("inf")):
    d, cache = load_mat_file(file, cache)
    d = select_orders_and_time(d, sh_order_sig=sh_order_sig, sh_order_scm=sh_order_scm, time_len_sig=time_len_sig)
    freq_bin = np.argmin(np.abs(d['freq'] - freq))
    x = get_narrowband_signal(d['anm'], d['nfft'], freq_bin)
    # x is of shape (real/imag, channels, time)

    y = d['R'][freq_bin]
    # y is now a  complex numpy array. convert to a real torch.tensor
    y = torch.stack([torch.from_numpy(np.real(y)), torch.from_numpy(np.imag(y))])

    # y is now of show (2, channels_out, channels_out)
    return x, y, cache


def load_stft_slice(file, center_freq_hz, bandwidth_hz, cache=None, sh_order_sig=float("inf"),
                    sh_order_scm=float("inf"),
                    time_len_sig=float("inf")):
    d, cache = load_mat_file(file, cache)
    d = select_orders_and_time(d, sh_order_sig=sh_order_sig, sh_order_scm=sh_order_scm, time_len_sig=time_len_sig)
    freq_to_bin = lambda f: int(np.argmin(np.abs(d['freq'] - f)))
    center_bin = freq_to_bin(center_freq_hz)
    bin_low = freq_to_bin(center_freq_hz - bandwidth_hz / 2)
    bin_high = 2 * center_bin - bin_low
    bin_range = range(bin_low, bin_high + 1)
    x = get_sliced_stft(d['anm'], d['nfft'], bin_range)
    # x is of shape (real/imag, channels, frequency, time)

    y = d['R'][center_bin]
    # y is now a  complex numpy array. convert to a real torch.tensor
    y = torch.stack([torch.from_numpy(np.real(y)), torch.from_numpy(np.imag(y))])
    # y is now of shape (2, channels_out, channels_out)

    return x, y, cache


def select_orders_and_time(d, sh_order_sig, sh_order_scm, time_len_sig):
    ''' takes the loaded data in a dictionary d, and process it so only selected
    SH orders of signal and SCM, and desired time length of the signals, are saved'''

    L_sig = (sh_order_sig + 1) ** 2
    L_scm = (sh_order_scm + 1) ** 2
    samples = (time_len_sig * d['fs'])

    if L_sig == np.inf:
        L_sig = None
    else:
        L_sig = int(L_sig)
    if L_scm == np.inf:
        L_scm = None
    else:
        L_scm = int(L_scm)
    if samples == np.inf:
        samples = None
    else:
        samples = int(samples)

    d['anm'] = d['anm'][:samples, :L_sig]
    d['R'] = d['R'][:, :L_scm, :L_scm]

    return d


class Dataset(data.Dataset):
    def __init__(self, root, center_frequency, bandwidth, transform=None, preload=True, dtype=torch.float32, train=True,
                 sh_order_sig=float("inf"), sh_order_scm=float("inf"), time_len_sig=float("inf")):

        self.dtype = dtype
        self.transform = transform
        self.preload_flag = preload
        self.sh_order_sig = sh_order_sig
        self.sh_order_scm = sh_order_scm
        self.time_len_sig = time_len_sig

        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'test')
        self.filenames = glob.glob(os.path.join(root, '*.mat'))  # get list of all mat files in the root folder
        assert len(self.filenames) > 0, 'data folder is empty'
        self.len = len(self.filenames)

        self.center_frequency = center_frequency
        self.bandwidth = bandwidth
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
            print(i / len(self.filenames))
            x, y = self.load_single_file(fn)
            self.samples.append((x, y))

    def load_single_file(self, filename):
        x, y, self.vec2mat_cache = load_stft_slice(filename, self.center_frequency, self.bandwidth, self.vec2mat_cache,
                                                   sh_order_sig=self.sh_order_sig,
                                                   sh_order_scm=self.sh_order_scm, time_len_sig=self.time_len_sig)
        return x, y

    def __getitem__(self, item: int):
        """
        Get a sample from the dataset
        """

        # load sample
        if self.preload_flag:
            x, y = self.samples[item]
        else:
            x, y = self.load_single_file(self.filenames[item])

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
