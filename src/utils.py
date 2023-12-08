import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import animation
from functools import partial

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
    ret = np.zeros((n, T, n))
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
    return der_beta/sqrt_norm_der_beta[:,np.newaxis]

def SRV_to_orig(q: NDArray, beta0: NDArray = None) -> NDArray:
    T = q.shape[0]
    n = q.shape[1]
    if beta0 is None:
        beta0 = np.zeros(n)
    beta = np.zeros((T+1, n))
    beta[0] = beta0
    for t in range(T):
        beta[t+1] = beta[t] + 1/T*np.linalg.norm(q[t])*q[t]
    return beta

def plot(q):
    plt.plot(q[:, 0], q[:, 1])
    plt.show()

def animation_update(frame, line, path):
    line.set_xdata(path[frame, :, 0])
    line.set_ydata(path[frame, :, 1])
    return line

def plot_path_animation(path, convert_from_SRV=False, interval=50):
    if convert_from_SRV:
        path = np.array([SRV_to_orig(curve) for curve in path])
    fig, ax = plt.subplots()
    line = ax.plot(path[0, :, 0], path[0, :, 1])[0]
    ax.set(xlim=[np.min(path[:,:,0]), np.max(path[:,:,0])], ylim=[np.min(path[:,:,1]), np.max(path[:,:,1])])
    ani = animation.FuncAnimation(fig=fig, func=partial(animation_update, line=line, path=path), frames=len(path), interval=interval)
    plt.show()