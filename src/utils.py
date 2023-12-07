import numpy as np
from numpy.typing import NDArray

def squared_norm(q: NDArray) -> float:
    return np.average(np.inner(q, q))

def norm(q: NDArray) -> float:
    return squared_norm(q)**.5

def normal_space_base(q: NDArray) -> NDArray:
    # Naively
    n = q.shape[1]
    T = q.shape[0]
    ret = np.zeros(n, T, n)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        ei = np.tile(ei, (T, 1))
        ret[i] = (q[:,i]/np.linalg.norm(q, axis=-1))[:,np.newaxis]*q + (np.linalg.norm(q, axis=-1))[:,np.newaxis] * ei
    return ret