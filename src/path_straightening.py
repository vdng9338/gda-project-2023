import numpy as np
from numpy.typing import NDArray
import utils

def proj_Co(q: NDArray):
    return q/utils.norm(q)

def G(q: NDArray):
    norm_q_t = np.linalg.norm(q, axis=-1)
    return np.average(q * norm_q_t[:,np.newaxis], axis=0)

def jacobian_G(q: NDArray):
    n = q.shape[1]
    mat = np.average(3*q[:,:,np.newaxis]*q[:,np.newaxis,:], axis=0) + np.eye(n)
    return mat

def proj_Co_to_Cc(q: NDArray, delta: float = 1e-2, eps: float = 1e-6):
    """Project an SRV representation of a curve in C^o (open curve) to C^c (closed curves).

    Parameters:
    q: NDArray of shape (T, n)
        SRV representation of the curve.
    
    delta: float
        Gradient descent step.

    eps: float
        Stopping threshold.
    """
    while True:
        J = jacobian_G(q)
        r = G(q)
        beta = np.linalg.solve(J, -r)
        Nq_base = utils.normal_space_base(q)
        q = q + delta*np.sum(beta[:,np.newaxis,np.newaxis]*Nq_base, axis=0)
        q /= utils.norm(q)
        newres = G(q)
        if np.inner(newres, newres) < eps**2:
            return q/utils.norm(q)
