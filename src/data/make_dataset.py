import numpy as np
import os

dir_cur = os.path.dirname(__file__)
FREE_FIELD_RAW_DATA_FOLDER = os.path.join(dir_cur, '..', '..', 'data', 'raw', 'free-field')
NUM_OF_CHANNELS = (6 + 1) ** 2


def load_free_field_raw_file(file_index):
    file_name = '%04d.bin' % file_index
    file_path = os.path.join(FREE_FIELD_RAW_DATA_FOLDER, file_name)
    data = open_convert_raw_file(file_path)
    return data


def open_convert_raw_file(file_path):
    # read from file
    data = np.fromfile(file_path, dtype=np.double)

    # to complex
    data = data[::2] + 1j * data[1::2]

    # reshape to matrices
    data = data.reshape((-1, NUM_OF_CHANNELS, NUM_OF_CHANNELS))
    data = np.transpose(data, axes=(0, 2, 1))

    return data

def load_free_field_frequencies():
    file_name = 'frequencies.txt'
    file_path = os.path.join(FREE_FIELD_RAW_DATA_FOLDER, file_name)
    return np.loadtxt(file_path)
