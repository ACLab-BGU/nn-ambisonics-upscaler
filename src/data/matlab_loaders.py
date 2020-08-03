import numpy as np
import scipy.io as sio


def load_mat_file(file_name):
    d = sio.loadmat(file_name)
    R_vecs = d['R'].transpose()
    Q = np.sqrt(R_vecs.shape[1]).astype(int)
    F = d['freq'].shape[0]
    d['R'] = np.zeros((F, Q, Q), dtype=np.cdouble)
    for f, R_vec in enumerate(R_vecs):
        d['R'][f, :, :] = vectorized_cov_to_mat(R_vec)
    d['R'] = d['R'] / d['R_scaling']
    d['anm'] = d['anm'].astype(np.cdouble) / d['anm_scaling']
    return d


def vectorized_cov_to_mat(v):
    K = v.shape[0]
    Q = np.sqrt(K).astype(int)
    mat = np.zeros((Q, Q))

    n = np.arange(Q)
    [n, m] = np.meshgrid(n, n)
    diag = n == m
    lower = n > m
    L = np.count_nonzero(lower)
    mat[diag] = v[:Q]
    mat[lower] = v[Q:Q+L]
    mat = mat.astype(np.cdouble)
    mat[lower] += 1j * v[Q+L:]
    mat_h = np.conj(mat.T)
    mat_h[n == m] = 0
    mat = mat + mat_h
    return mat
