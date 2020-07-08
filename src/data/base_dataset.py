import glob
import os

import numpy as np
import torch
import torch.utils.data as data

from src.data.make_dataset import open_and_convert_raw_file, load_free_field_frequencies


def downgrade_scm(scm, sh_order, axis=(0, 1)):
    """
    downgrade the SCM to a given SH order
    :param scm: an nd-array containing SCMs
    :param sh_order: the required SH-order for downgrading
    :param axis: a list with the axis to downgrade
    :return: a downgraded SCM
    """

    ndim = scm.ndimension() if type(scm) == torch.Tensor else scm.ndim
    tuple_slice = [slice(None)] * ndim
    tuple_slice[axis[0]] = slice((sh_order + 1) ** 2)
    tuple_slice[axis[1]] = slice((sh_order + 1) ** 2)
    scm_out = scm[tuple(tuple_slice)]

    return scm_out


class BasicDataset(data.Dataset):
    def __init__(self, root, transform=None, preload=False, input_sh_order=3):
        """
        initialize database
        :param root: root directory of the dataset
        :param transform: a custom transform function
        :param preload: a flag to preload all dataset to memory
        """

        self.transform = transform
        self.preload_flag = preload
        self.sh_order = input_sh_order
        self.filenames = glob.glob(os.path.join(root, '*.bin'))  # get list of all bin files in the root folder
        assert len(self.filenames) > 0, 'data folder is empty'
        self.len = len(self.filenames)
        self.frequencies = load_free_field_frequencies()  # frequencies of each SCM in the dataset

        if preload:
            self._preload()

    def _preload(self):
        """
        load all dataset to memory
        """
        self.samples = []

        # load all files to memory and form the database
        for fn in self.filenames:
            self.samples.append(open_and_convert_raw_file(fn))

    def __getitem__(self, item: int):
        """
        Get a sample from the dataset
        """

        # load sample
        if self.preload_flag:
            sample = self.samples[item]
        else:
            sample = open_and_convert_raw_file(self.filenames[item])

        # generate target/label: decouple real and imaginary parts of sample
        target = np.concatenate((np.real(sample[np.newaxis]), np.imag(sample[np.newaxis])), axis=0)
        target = target.transpose([1, 0, 2, 3])

        # convert to torch tensor
        target = torch.from_numpy(target)

        # generate input data
        inputs = downgrade_scm(target, self.sh_order, axis=(2, 3))

        # perform some transformation
        if self.transform:
            inputs = self.transform(inputs)

        # return sample
        return inputs, target

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


class BasicDatasetLT(data.Dataset):
    def __init__(self, root, transform=None, preload=True, input_sh_order=3, dtype=torch.float32, train=True):

        self.dtype = dtype
        self.transform = transform
        self.preload_flag = preload
        self.sh_order = input_sh_order
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'test')
        self.filenames = glob.glob(os.path.join(root, '*.bin'))  # get list of all bin files in the root folder
        assert len(self.filenames) > 0, 'data folder is empty'
        self.len = len(self.filenames)
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
            self.samples.append(open_and_convert_raw_file(fn))

    def __getitem__(self, item: int):
        """
        Get a sample from the dataset
        """

        # load sample
        if self.preload_flag:
            sample = self.samples[item]
        else:
            sample = open_and_convert_raw_file(self.filenames[item])

        # generate target/label: decouple real and imaginary parts of sample
        targets = np.concatenate((np.real(sample[np.newaxis]), np.imag(sample[np.newaxis])), axis=0)
        targets = targets.transpose([1, 0, 2, 3])

        # convert to torch tensor
        targets = torch.from_numpy(targets).type(self.dtype)

        # take only a single frequency
        targets = targets[0, :]

        # generate input data
        inputs = downgrade_scm(targets, self.sh_order, axis=(1, 2))

        # normalize
        inputs = inputs / inputs.norm()
        targets = targets / targets.norm()

        # perform some transformation
        if self.transform:
            inputs = self.transform(inputs)

        # return sample
        return inputs, targets

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
