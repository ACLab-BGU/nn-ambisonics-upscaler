import numpy as np


def uniform(length) -> np.ndarray:
    ph = np.random.rand(length) * 2 * np.pi
    v = np.random.rand(length)
    th = np.arccos(2 * v - 1)
    return np.vstack([th, ph])
