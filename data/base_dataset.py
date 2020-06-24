import numpy as np
import torch.utils.data as data
import glob
import os

from src.data.make_dataset import open_convert_raw_file, load_free_field_frequencies


def downgrade_scm(scm, sh_order, axis=(0, 1)):
    """
    downgrade the SCM to a given SH order
    :param scm: an nd-array containing SCMs
    :param sh_order: the required SH-order for downgrading
    :param axis: a list with the axis to downgrade
    :return: a downgraded SCM
    """

    tuple_slice = [slice(None)] * scm.ndim
    tuple_slice[axis[0]] = slice((sh_order+1)**2)
    tuple_slice[axis[1]] = slice((sh_order+1)**2)
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
            self.samples.append(open_convert_raw_file(fn))

    def __getitem__(self, item):
        """
        Get a sample from the dataset
        """

        # load sample
        if self.preload_flag:
            sample = self.samples[item]
        else:
            sample = open_convert_raw_file(self.filenames[item])

        # generate target/label: decouple real and imaginary parts of sample
        target = np.concatenate((np.real(sample[np.newaxis]), np.imag(sample[np.newaxis])), axis=0)

        # generate input data
        input_data = downgrade_scm(target, self.sh_order, axis=(2, 3))

        # perform some transformation
        if self.transform:
            input_data = self.transform(input_data)

        # return sample
        return input_data, target

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
