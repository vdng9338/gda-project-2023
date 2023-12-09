import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import animation
from functools import partial
import json

def inner(x: NDArray, y: NDArray):
    """Inner product of paths or vector fields x and y, both of which have shape (T+1, n).

    The integral is computed using the points at the left of the discretization
    intervals, hence the last points of the input paths are ignored."""
    # Renormalize because indices are in [0,...T] instead of [0, ..., 1]
    return np.tensordot(x[:-1], y[:-1]) / (x.shape[0]**2)

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
    beta[1:] = np.linalg.norm(q, axis=-1, keepdims=True)*q
    beta = np.cumsum(beta, axis=0) / T
    return beta

def plot(q, title=None):
    plt.plot(q[:, 0], q[:, 1])
    if title is not None:
        plt.title(title)
    plt.show()

def animation_update(frame, line, path):
    line.set_xdata(path[frame, :, 0])
    line.set_ydata(path[frame, :, 1])
    return line

def plot_path_animation(path, convert_from_SRV=False, interval=50, title=None):
    if convert_from_SRV:
        path = np.array([SRV_to_orig(curve) for curve in path])
    fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    line = ax.plot(path[0, :, 0], path[0, :, 1])[0]
    ax.set(xlim=[np.min(path[:,:,0]), np.max(path[:,:,0])], ylim=[np.min(path[:,:,1]), np.max(path[:,:,1])])
    ani = animation.FuncAnimation(fig=fig, func=partial(animation_update, line=line, path=path), frames=len(path), interval=interval)
    plt.show()

def load_path(filename: str, discretization_steps: int = 100):
    f = open(filename, 'r')
    points = np.array(json.load(f))
    f.close()
    ret = np.zeros((discretization_steps, points.shape[1]))
    for i in range(discretization_steps):
        pos = i/(discretization_steps-1)*(len(points)-1)
        index = int(pos)
        frac = pos-index
        if index < len(points)-1:
            ret[i] = points[index] + frac*(points[index+1] - points[index])
        else:
            ret[i] = points[index]
    return ret