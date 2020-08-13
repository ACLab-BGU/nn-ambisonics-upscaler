from typing import Tuple

import numpy as np


def i2nm(i: np.ndarray, is_complex=True) -> Tuple[np.ndarray, np.ndarray]:
    if not is_complex:
        raise NotImplementedError('Only complex spherical harmonics are supported')

    n = np.floor(np.sqrt(i))
    m = i - n * (n + 1)
    return n.astype(int), m.astype(int)


def nm2i(n: np.ndarray, m: np.ndarray, is_complex=True) -> np.ndarray:
    if not is_complex:
        raise NotImplementedError('Only complex spherical harmonics are supported')
    return n * (n + 1) + m


def mat(max_order, omega, is_transposed=False, is_complex=True) -> np.ndarray:
    if not is_complex:
        raise NotImplementedError('Only complex spherical harmonics are supported')

    th = omega[0, :]
    ph = omega[1, :]
    n, m = i2nm(np.arange(int(max_order + 1) ** 2))
    if is_transposed:
        n = n[:, np.newaxis]
        m = m[:, np.newaxis]
    else:
        th = th[:, np.newaxis]
        ph = ph[:, np.newaxis]

    return sample(n, m, th, ph)


def sample(n, m, th, ph):
    from scipy.special import sph_harm
    return sph_harm(m, n, ph, th)


def Q2N(Q):
    return (np.sqrt(Q)-1).astype(int).tolist()


def power_map(cov, omega, approx_nearest_PSD=False) -> np.ndarray:
    Q, Q1 = cov.shape
    assert Q == Q1
    order = Q2N(Q)

    if approx_nearest_PSD:
        from src.utils.linear_algebra import find_nearest_PSD_mat
        cov = find_nearest_PSD_mat(cov)

    Y = mat(order, omega.reshape((2, -1)))  # (XY, Q)
    power = np.sum((Y @ cov) * Y.conj(), axis=1)

    power = power.reshape(omega.shape[1:])
    return np.real(power)