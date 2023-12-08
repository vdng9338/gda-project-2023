import numpy as np
from numpy.typing import NDArray

def inner(x: NDArray, y: NDArray):
    """Inner product of paths or vector fields x and y, both of which have shape (T+1, n).

    The integral is computed using the points at the left of the discretization
    intervals, hence the last points of the input paths are ignored."""
    return np.average(np.inner(x[:-1], y[:-1]))

def squared_norm(q: NDArray) -> float:
    return inner(q, q)

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

def SRV(beta: NDArray) -> NDArray:
    k = beta.shape[0]-1
    der_beta = k*(beta[1:] - beta[:-1])
    sqrt_norm_der_beta = np.linalg.norm(der_beta, axis=-1)**.5
    return der_beta/sqrt_norm_der_beta
