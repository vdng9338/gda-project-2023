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

    Parameters
    ----------
    q: NDArray of shape (T+1, n)
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

def proj_Tq_Cc(w: NDArray, q: NDArray) -> NDArray:
    """
    Parameters
    ----------
    w: NDArray of shape (T+1, n)
        The element to project to T_q(C^c).
        
    q: NDArray of shape (T+1, n)
        The point at which the tangent space is taken.
        
    Returns
    -------
    proj: NDArray of shape (T+1, n)
        The projection of w into T_q(C^c)."""
    n = w.shape[1]
    Nq_base = utils.normal_space_base(q)
    # TODO Do we want to consider q in the base of the normal space? (Eqn. 1 in the paper)
    for i in range(n):
        w -= utils.inner(Nq_base[i], w) * Nq_base[i]
    return w

def differentiate_path(alpha: NDArray) -> NDArray:
    k = alpha.shape[0]-1
    T = alpha.shape[1]-1
    n = alpha.shape[2]
    ret = np.zeros(alpha.shape)
    for tau in range(1, k+1):
        ret[tau] = proj_Tq_Cc(k*(alpha[tau] - alpha[tau-1]), alpha[tau])
    return ret

def covariant_integral(der_alpha: NDArray, alpha: NDArray) -> NDArray:
    k = der_alpha.shape[0]
    u = np.zeros_like(der_alpha)
    # u[0] is already zero
    for tau in range(1, k+1):
        u_proj = proj_Tq_Cc(u[tau-1], alpha[tau])
        u_parallel = u_proj * utils.norm(u[tau-1]) / utils.norm(u_proj)
        
