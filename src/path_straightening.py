import numpy as np
from numpy.typing import NDArray
import utils
from matplotlib import pyplot as plt

def proj_Co(q: NDArray):
    return q/utils.norm(q)

def G(q: NDArray):
    norm_q_t = np.linalg.norm(q, axis=-1)
    return np.einsum('tn,t->n', q, norm_q_t) / q.shape[0]

def jacobian_G(q: NDArray):
    n = q.shape[1]
    integ = np.einsum("ti, tj -> ij", q, q) / q.shape[0]
    mat = 3*integ + np.eye(n)
    return mat

def proj_Cc(q: NDArray, delta: float = 1e-1, eps: float = 1e-8):
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
    q = proj_Co(q)
    while True:
        J = jacobian_G(q)
        r = G(q)
        beta = np.linalg.solve(J, -r)
        Nq_base = utils.normal_space_base(q, False)
        q += delta*np.sum(beta[:,np.newaxis,np.newaxis]*Nq_base, axis=0)
        q /= utils.norm(q)
        newres = G(q)
        if np.inner(newres, newres) < eps:
            return q

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
    Nq_base = utils.normal_space_base(q, True)
    # TODO Do we want to consider q in the base of the normal space? (Eqn. 1 in the paper)
    for i in range(n):
        w = w - utils.inner(Nq_base[i], w) * Nq_base[i]
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
    # u[0] is already zero
    for tau in range(1, k+1):
        u_proj = proj_Tq_Cc(u[tau-1], alpha[tau])
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
    w = u.copy()
    k = u.shape[0]-1
    for tau in range(k+1):
        w[tau] -= (tau/k)*tilde_u[tau]
    return w

def gradient_descent(alpha: NDArray, w: NDArray, eps: float = 1e-1, iter_num: int = 0) -> NDArray:
    # scale = pow(0.5, iter_num/10)
    scale = 1
    scaled_eps = eps * scale
    k = alpha.shape[0]-1
    for tau in range(k+1):
        alpha_prime = alpha[tau] - scaled_eps*w[tau]
        alpha[tau] = proj_Cc(alpha_prime)

def energy(der_alpha: NDArray) -> float:
    return np.tensordot(der_alpha, der_alpha, 3) / der_alpha.shape[0] / der_alpha.shape[1] / 2

def path_straightening(beta_0, beta_1, k: int, eps_2: float = 1e-4):
    q_0 = utils.SRV(beta_0)
    q_1 = utils.SRV(beta_1)
    q_0 /= utils.norm(q_0)
    q_1 /= utils.norm(q_1)
    theta = np.arccos(utils.inner(q_0, q_1))
    taus = np.linspace(0.0, 1.0, k+1)[:,np.newaxis,np.newaxis]
    # Start with a wildly suboptimal path to show the effects of path straightening
    #taus = 4*taus**2 - 3*taus # a polynomial P of degree two such that P(0)=0, P(1)=1, P(1/4)=-1/2
    alpha = 1/np.sin(theta)*(np.sin(theta*(1-taus)) * q_0 + np.sin(theta*taus) * q_1)
    utils.plot_path_animation(alpha, True, title="Path in $\mathcal{C}^o$")
    utils.plot_save_path(alpha, "path_Co.pdf", True)
    for tau in range(k+1):
        alpha[tau] = proj_Cc(alpha[tau])
    utils.plot_path_animation(alpha, True, title="Path in $\mathcal{C}^c$")
    utils.plot_save_path(alpha, "path_Cc_init.pdf", True)
    num_iterations = 0
    energies = []
    criteria = []
    while True:
        der_alpha = differentiate_path(alpha)
        u = covariant_integral(der_alpha, alpha)
        tilde_u = backward_parallel_transport(u, alpha)
        w = grad_E_H0(u, tilde_u)
        gradient_descent(alpha, w, iter_num=num_iterations)
        num_iterations+=1
        criterion = np.tensordot(w, w, 3)/w.shape[0]/w.shape[1]
        energ = energy(der_alpha)
        criteria.append(criterion)
        energies.append(energ)
        if num_iterations % 100 == 0:
            utils.plot_save_path(alpha, f"iter_{num_iterations}.pdf", True)
            utils.plot_path_animation(alpha, True, title=f"Iteration #{num_iterations}")
            plt.plot(energies, label='Energy')
            #plt.plot(criteria, label='Stopping criterion')
            plt.xlabel('Iteration')
            plt.ylabel('Energy')
            plt.show()
        if num_iterations == 1 or not(num_iterations % 10):
            print(f"Iter: {num_iterations}")
            print(f"Energy: {energ:.5f}")
            print(f"Sum of <w(tau), w(tau)>: {criterion:.5f}")
        #if np.sum([utils.norm(w[tau]) for tau in range(k+1)]) <= eps_2:
        if criterion <= eps_2:
            return alpha

