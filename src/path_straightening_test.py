from path_straightening import *
import utils
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from numpy.typing import ArrayLike, NDArray
import sys

def gen_ellipse(a: float, b: float, T: int) -> NDArray:
    """Generates the discretization of an ellipse as a path.
    
    Parameters
    ----------
    a, b: float
        Parameters of the ellipse, such that the equation of the ellipse is
        x^2/a^2 + y^2/b^2 = 1.
    
    T: int
        Number of discretization steps."""
    thetas = np.linspace(0.0, 2*np.pi, T+1)
    return np.stack((a*np.cos(thetas), b*np.sin(thetas)), axis=-1)

def gen_3d_oscillating_ellipse(a: float, b: float, c: float, k: int, dims: ArrayLike, T: int) -> NDArray:
    """Generates the discretization of an ellipse that oscillates in a certain dimension.
    
    Parameters
    ----------
    a, b: float
        Parameters of the 2D projection of the ellipse.

    c: float
        Amplitude of the oscillations of the ellipse.

    k: int
        Number of oscillations of the ellipse.
        
    dims: list (or tuple or array) of 3 integers
        The orientation of the ellipse: the two first dimensions will be used for the ellipse part, the third dimension
        will be used for the oscillations."""
    thetas = np.linspace(0.0, 2*np.pi, T+1)
    return np.stack((a*np.cos(thetas), b*np.sin(thetas), c*np.sin(k*thetas)), axis=-1)[:,dims]
    
def main():
    T = 100
    k = 100
    if len(sys.argv) == 2:
        if sys.argv[1] == '2dellipses':
            shape1 = gen_ellipse(a = 2.0, b = 0.5, T = T)
            shape2 = gen_ellipse(a = 1.0, b = 1.0, T = T)
        elif sys.argv[1] == '3dellipses':
            shape1 = gen_3d_oscillating_ellipse(a = 2.0, b = 0.5, c = 1.0, k = 1, dims=(0, 1, 2), T = T)
            shape2 = gen_3d_oscillating_ellipse(a = 1.0, b = 1.0, c = 2.0, k = 3, dims=(2, 0, 1), T = T)
    elif len(sys.argv) == 3:
        shape1 = utils.load_path(sys.argv[1], discretization_steps=T)
        shape2 = utils.load_path(sys.argv[2], discretization_steps=T)
    else:
        print(f"Usage for 2D ellipses: python3 {sys.argv[0]} 2dellipses")
        print(f"Usage for 3D oscillating ellipses: python3 {sys.argv[0]} 3dellipses")
        print(f"Usage for arbitrary shapes: python3 {sys.argv[0]} <shape1.json> <shape2.json>")
        sys.exit(1)
    dim = shape1.shape[1]
    fig = plt.figure()
    if dim == 2:
        ax = fig.add_subplot()
        ax.plot(shape1[:,0], shape1[:,1])
        ax.plot(shape2[:,0], shape2[:,1])
    else:
        ax = fig.add_subplot(projection='3d')
        ax.plot3D(shape1[:,0], shape1[:,1], shape1[:,2])
        ax.plot3D(shape2[:,0], shape2[:,1], shape2[:,2])
    ax.set_aspect('equal', adjustable='datalim')
    plt.show()
    path_srv = path_straightening(shape1, shape2, k)
    path = np.array([utils.SRV_to_orig(path_srv[i]) for i in range(path_srv.shape[0])])
    utils.plot_path_animation(path, title="Straightened path")

if __name__ == "__main__":
    main()