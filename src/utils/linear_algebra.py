import numpy as np


def find_nearest_PSD_mat(mat: np.ndarray):
    """ finds the nearest semi-positive-definite matrix, using the SVD"""
    # preliminaries
    assert mat.ndim == 2, "input must be a matrix"
    assert mat.shape[0] == mat.shape[1], "input must be a square matrix"

    # make hermitian
    mat = 0.5 * (mat + mat.T.conj())

    # make semi-positive
    eigs, v = np.linalg.eig(mat)
    eigs = np.real(eigs)
    eigs[eigs < 0] = 0
    # u, s, vh = np.linalg.svd(mat, full_matrices=True)
    mat = v * eigs @ v.T.conj()

    return mat
