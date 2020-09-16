import glob
import os
import pickle
from pathlib import Path

import matlab.engine
import numpy as np
import scipy.io as sio

# default settings
from scipy import io as sio

num_of_reflections = [10]
source_type = ["whitenoise"]  # whitenoise/speech
num_of_files = 50000
sig_length = 2
target_sh_order = 4
compact_cov = False
path_data = os.path.join("..", "..", "data")
output_format = "pickle"  # pickle/mat/npz


def gen_data(num_of_reflections, source_type, compact_cov, num_of_files, path_data, output_format):
    '''generate data using matlab'''
    eng = matlab.engine.start_matlab()
    eng.cd('MATLAB')
    eng.restoredefaultpath(nargout=0)
    eng.addpath('spherical harmonics/')
    for refs in num_of_reflections:
        for sig in source_type:
            # folders stuff
            folder_name = f"{sig}_{refs}_reflections_python"
            folder_path = os.path.abspath(os.path.join(path_data, folder_name))
            os.makedirs(folder_path, exist_ok=True)

            # generate file with matlab and save to mat file
            eng.make_image_method_data(num_of_files, 1, "folder_path", folder_path, "number_of_reflections", refs,
                                       "source_type", sig, "duration", sig_length, "target_sh_order", target_sh_order,"compact_cov", compact_cov)

            # convert files to a better format
            # TODO: check more efficient alternatives, instead of saving mat files and then converting
            convert_mat_files(folder_path, output_format)


def convert_mat_files(folder_path, output_format):
    '''convert all the files in folder_path to the desired output_format (pickle/mat/npz)'''

    path_list = glob.glob(os.path.join(folder_path, "*.mat"))
    for path_old in path_list:
        # load file
        data = sio.loadmat(path_old)

        # change memory order to python convention
        for key in data:
            if type(data[key]) == np.ndarray:
                data[key] = np.ascontiguousarray(data[key])
        path_new = os.path.join(os.path.dirname(path_old), Path(path_old).stem)

        if output_format == "mat":
            sio.savemat(path_old, data)
        elif output_format == "pickle":
            path_new = path_new + ".pickle"
            file_new = open(path_new, 'wb')
            pickle.dump(data, file_new)
            file_new.close()
        elif output_format == "npz":
            path_new = path_new + ".npz"
            np.savez(path_new, **data)
        else:
            raise NotImplementedError

        if output_format is not "mat":
            os.remove(path_old)


def load_data_file(file_name, cache=None):
    '''load a single data file'''

    # perform loading according to data type
    file_format = Path(file_name).suffix
    if file_format == ".mat":
        d = sio.loadmat(file_name)
    elif file_format == ".pickle":
        pickle_file = open(file_name, 'rb')
        d = pickle.load(pickle_file)
        pickle_file.close()
    elif file_format == ".npz":
        d = dict(np.load(file_name, allow_pickle=True))
    else:
        raise NotImplementedError

    # some post-processing
    if d['R'].ndim == 2:
        is_cov_compact = True
    else:
        is_cov_compact = False

    if is_cov_compact:
        R_vecs = d['R'].transpose()
        Q = np.sqrt(R_vecs.shape[1]).astype(int)
        F = d['freq'].shape[0]
        d['R'] = np.zeros((F, Q, Q), dtype=np.cdouble)
        for f, R_vec in enumerate(R_vecs):
            d['R'][f, :, :], cache = vectorized_cov_to_mat(R_vec, cache)
    else:
        cache = None
    d['R'] = d['R'] / d['R_scaling']
    d['anm'] = d['anm'].astype(np.double) / d['anm_scaling']
    d['nfft'] = int(d['nfft'])
    return d, cache


def vectorized_cov_to_mat(v, cache=None):
    K = v.shape[0]
    Q = np.sqrt(K).astype(int)
    mat = np.zeros((Q, Q))

    if cache is None:
        n = np.arange(Q)
        [n, m] = np.meshgrid(n, n)
        diag = n == m
        lower = n > m
        L = np.count_nonzero(lower)
        cache = {"diag": diag, "lower": lower, "L": L}
    else:
        diag = cache["diag"]
        lower = cache["lower"]
        L = cache["L"]

    mat[diag] = v[:Q]
    mat[lower] = v[Q:Q + L]
    mat = mat.astype(np.cdouble)
    mat[lower] += 1j * v[Q + L:]
    mat_h = np.conj(mat.T)
    mat_h[diag] = 0
    mat = mat + mat_h

    return mat, cache


if __name__ == '__main__':
    gen_data(num_of_reflections=num_of_reflections, source_type=source_type, num_of_files=num_of_files,
             path_data=path_data, output_format=output_format, compact_cov=compact_cov)
