import numpy as np
from numpy.typing import NDArray
import utils
from math import isnan

def proj_Co(q: NDArray):
    return q/utils.norm(q)

def G(q: NDArray):
    norm_q_t = np.linalg.norm(q, axis=-1)
    return np.average(q * norm_q_t[:,np.newaxis], axis=0)

def jacobian_G(q: NDArray):
    n = q.shape[1]
    mat = np.average(3*q[:,:,np.newaxis]*q[:,np.newaxis,:], axis=0) + np.eye(n)
    return mat

def proj_Cc(q: NDArray, delta: float = 1e-1, eps: float = 1e-4):
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
    #utils.plot(q, title="Before projection onto $\mathcal{C}^o$")
    q = proj_Co(q)
    #utils.plot(q, title="After projection onto $\mathcal{C}^o$")
    while True:
        J = jacobian_G(q)
        r = G(q)
        beta = np.linalg.solve(J, -r)
        """print(f"Jacobian: {J}")
        print(f"Residual G(q): {r}")
        print(f"Solution to Jbeta = -r: {beta}")"""
        Nq_base = utils.normal_space_base(q)
        q = q + delta*np.sum(beta[:,np.newaxis,np.newaxis]*Nq_base, axis=0)
        q /= utils.norm(q)
        newres = G(q)
        if isnan(np.inner(newres, newres)):
            raise SystemExit
        #print(f'New residual squared norm: {np.inner(newres, newres)}')
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
    #print(f'Projecting w to tangent space')
    n = w.shape[1]
    #print(f'Norm of w: {utils.norm(w)}')
    Nq_base = utils.normal_space_base(q)
    #utils.plot_path_animation(Nq_base, interval=500, title="Normal space base")
    # TODO BIG PROBLEM HERE!!
    # TODO Do we want to consider q in the base of the normal space? (Eqn. 1 in the paper)
    for i in range(n):
        w = w - utils.inner(Nq_base[i], w) * Nq_base[i]
    #print(f'Norm of projected w: {utils.norm(w)}')
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
    k = der_alpha.shape[0]-1
    u = np.zeros_like(der_alpha)
    #utils.plot_path_animation(der_alpha, title="$\\frac{d\\alpha}{d\\tau}$")
    # u[0] is already zero
    for tau in range(1, k+1):
        #print(f"Before projection to Tq(Cc) tau={tau}")
        u_proj = proj_Tq_Cc(u[tau-1], alpha[tau])
        #print("After projection to Tq(Cc).")
        #print(f'Norm of u_proj: {utils.norm(u_proj)}, norm of u[tau-1]: {utils.norm(u[tau-1])}')
        if utils.norm(u_proj) > 1e-50:
            u_parallel = u_proj * utils.norm(u[tau-1]) / utils.norm(u_proj)
        else:
            u_parallel = u_proj
        u[tau] = 1/k*der_alpha[tau] + u_parallel
    return u

def backward_parallel_transport(u: NDArray, alpha: NDArray) -> NDArray:
    tilde_u = np.zeros_like(u)
    tilde_u[-1] = u[-1]
    l = utils.norm(u[-1])
    k = u.shape[0]-1
    for tau in range(k-1, -1, -1):
        c = proj_Tq_Cc(tilde_u[tau+1], alpha[tau])
        tilde_u[tau] = l*c/utils.norm(c)
    return tilde_u

def grad_E_H0(u, tilde_u):
    w = np.zeros_like(u)
    k = u.shape[0]-1
    for tau in range(k+1):
        w[tau] = u[tau] - tau/k*tilde_u[tau]
    return w

def gradient_descent(alpha: NDArray, w: NDArray, eps: float = 1e-1) -> NDArray:
    k = alpha.shape[0]-1
    for tau in range(k+1):
        alpha_prime = alpha[tau] - eps*w[tau]
        alpha[tau] = proj_Cc(alpha_prime)

def path_straightening(beta_0, beta_1, k: int, eps_2: float = 1e-1):
    q_0 = utils.SRV(beta_0)
    q_1 = utils.SRV(beta_1)
    q_0 /= utils.norm(q_0)
    q_1 /= utils.norm(q_1)
    """print(f"Norm of q_0: {utils.norm(q_0)}")
    print(f"Norm of q_1: {utils.norm(q_1)}")
    print(f"Inner product of q_0 and q_1: {utils.inner(q_0, q_1)}")"""
    theta = np.arccos(utils.inner(q_0, q_1))
    taus = np.linspace(0.0, 1.0, k+1)[:,np.newaxis,np.newaxis]
    alpha = 1/np.sin(theta)*(np.sin(theta*(1-taus)) * q_0 + np.sin(theta*taus) * q_1)
    utils.plot_path_animation(alpha, True, title="Path in $\mathcal{C}^o$")
    for tau in range(k+1):
        alpha[tau] = proj_Cc(alpha[tau])
    utils.plot_path_animation(alpha, True, title="Path in $\mathcal{C}^c$")
    num_iterations = 0
    while True:
        der_alpha = differentiate_path(alpha)
        #print("Alpha differentiated")
        u = covariant_integral(der_alpha, alpha)
        #print("u computed")
        #utils.plot_path_animation(u[:30])
        tilde_u = backward_parallel_transport(u, alpha)
        #print("tilde_u computed")
        w = grad_E_H0(u, tilde_u)
        #print("Gradient computed")
        gradient_descent(alpha, w)
        if num_iterations % 10 == 1:
            utils.plot_path_animation(alpha, True, title=f"Iteration #{num_iterations}")
        num_iterations+=1
        if np.sum([utils.norm(w[tau]) for tau in range(k+1)]) <= eps_2:
            return alpha

