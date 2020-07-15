from typing import Union

import numpy as np


def s2c(doa: np.ndarray, r: Union[float, np.ndarray] = 1) -> np.ndarray:
    if doa.shape[0] != 2:
        raise ValueError('doa must be of shape (2,n)')
    n = doa.shape[1]
    x = np.zeros((3, n))

    th = doa[0, :]
    ph = doa[1, :]
    sin_th = np.sin(th)

    x[0, :] = r * sin_th * np.cos(ph)
    x[1, :] = r * sin_th * np.sin(ph)
    x[2, :] = r * np.cos(th)
    return x
