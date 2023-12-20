import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits import mplot3d
from functools import partial
import json

def inner(x: NDArray, y: NDArray):
    """Inner product of paths or vector fields x and y, both of which have shape (T+1, n).

    The integral is computed using the points at the left of the discretization
    intervals, hence the last points of the input paths are ignored."""
    # Renormalize because indices are in [0,...T] instead of [0, ..., 1]
    return np.tensordot(x[:-1], y[:-1]) / (x.shape[0])

def squared_norm(q: NDArray) -> float:
    return inner(q, q)

def norm(q: NDArray) -> float:
    return squared_norm(q)**.5

def normal_space_base(q: NDArray, orthonormalize: bool) -> NDArray:
    # Naively
    n = q.shape[1]
    T = q.shape[0]
    ret = np.zeros((n, T, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        ei = np.tile(ei, (T, 1))
        ret[i] = (q[:,i]/np.linalg.norm(q, axis=-1))[:,np.newaxis]*q + (np.linalg.norm(q, axis=-1))[:,np.newaxis] * ei
    # Orthonormalize it
    if orthonormalize:
        for i in range(n):
            for j in range(i-1):
                ret[i] -= inner(ret[j], ret[i])*ret[j]
            ret[i] /= norm(ret[i])
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
    dim = q.shape[1]
    if dim < 2 or dim > 3:
        return
    fig = plt.figure()
    if dim == 2:
        ax = fig.add_subplot()
        ax.scatter(q[:, 0], q[:, 1])
    elif dim == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(q[:, 0], q[:, 1], q[:, 2])
    if title is not None:
        ax.set_title(title)
    plt.show()

def animation_update(frame, scat, path):
    dim = path.shape[2]
    if dim == 2:
        """line.set_xdata(path[frame, :, 0])
        line.set_ydata(path[frame, :, 1])"""
        scat.set_offsets(path[frame])
    else:
        #line.set_data_3d(path[frame, :, 0], path[frame, :, 1], path[frame, :, 2])
        scat.set_offsets(path[frame,:,0:2])
        scat.set_3d_properties(path[frame,:,2], 'z')
    return scat

def plot_path_animation(path, convert_from_SRV=False, interval=50, num_images=-1, title=None):
    dim = path.shape[2]
    if dim < 2:
        return
    if convert_from_SRV:
        path = np.array([SRV_to_orig(curve) for curve in path])
    if num_images > 1:
        newpath = np.zeros((num_images, path.shape[1], dim))
        for i in range(num_images):
            newpath[i] = path[int(i*(len(path)-1)/(num_images-1))]
        path = newpath
    fig = plt.figure()
    if dim == 2:
        ax = fig.add_subplot()
        scat = ax.scatter(path[0, :, 0], path[0, :, 1], marker='+')
    else:
        ax = fig.add_subplot(projection='3d')
        scat = ax.scatter(path[0, :, 0], path[0, :, 1], path[0, :, 2], marker='+')
    if title is not None:
        ax.set_title(title)
    ax.set(xlim=[np.min(path[:,:,0]), np.max(path[:,:,0])], ylim=[np.min(path[:,:,1]), np.max(path[:,:,1])])
    if dim >= 3:
        ax.set_zlim(np.min(path[:,:,2]), np.max(path[:,:,2]))
    ani = animation.FuncAnimation(fig=fig, func=partial(animation_update, scat=scat, path=path), frames=len(path), interval=interval)
    plt.show()

def plot_save_path(path, filename, convert_from_SRV=False, num_images=6, figsize=(1.5, 1)):
    dim = path.shape[2]
    if dim < 2 or num_images < 2:
        return
    if convert_from_SRV:
        path = np.array([SRV_to_orig(curve) for curve in path])
    newpath = np.zeros((num_images, path.shape[1], dim))
    for i in range(num_images):
        newpath[i] = path[int(i*(len(path)-1)/(num_images-1))]
    path = newpath
    lims = [[np.min(path[:, :, d]), np.max(path[:, :, d])] for d in range(min(3, dim))]
    fig = plt.figure(figsize=(num_images*figsize[0], figsize[1]))
    for i in range(num_images):
        if dim >= 3:
            ax = fig.add_subplot(1, num_images, i+1, projection='3d')
            ax.scatter(path[i, :, 0], path[i, :, 1], path[i, :, 2], marker='+')
            ax.dist = 7
        else:
            ax = fig.add_subplot(1, num_images, i+1)
            ax.scatter(path[i, :, 0], path[i, :, 1], marker='+')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set(xlim=lims[0], ylim=lims[1])
        if dim >= 3:
            ax.set_zticks([])
            ax.set_zlim(lims[2])
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
    plt.close('all')

def load_path(filename: str, discretization_steps: int = 100):
    f = open(filename, 'r')
    points = json.load(f)
    f.close()
    if points[-1] != points[0]:
        print(f"Warning: Input path in file {filename} is not closed, closing")
        points.append(points[0])
    points = np.array(points)
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