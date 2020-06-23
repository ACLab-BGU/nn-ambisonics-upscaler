import numpy as np
import os

FREE_FIELD_RAW_DATA_FOLDER = os.path.join('..', '..', 'data', 'raw', 'free-field')
NUM_OF_CHANNELS = (6 + 1) ** 2


def load_free_field_raw_file(file_index):
    file_name = '%04d.bin' % file_index
    file_path = os.path.join(FREE_FIELD_RAW_DATA_FOLDER, file_name)

    # read from file
    R = np.fromfile(file_path, dtype=np.double)

    # to complex
    R = R[::2] + 1j * R[1::2]

    # reshape to matrices
    R = R.reshape((-1, NUM_OF_CHANNELS, NUM_OF_CHANNELS))
    R = np.transpose(R, axes=(0, 2, 1))
    return R
