from path_straightening import *
import utils
import matplotlib.pyplot as plt
import numpy as np
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
    
def main():
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <shape1.json> <shape2.json>")
        sys.exit(1)
    T = 100
    k = 100
    """ellipse = gen_ellipse(a = 2.0, b = 0.5, T = T)
    circle = gen_ellipse(a = 1.0, b = 1.0, T = T)"""
    shape1 = utils.load_path(sys.argv[1], discretization_steps=T)
    shape2 = utils.load_path(sys.argv[2], discretization_steps=T)
    plt.plot(shape1[:,0], shape1[:,1])
    plt.plot(shape2[:,0], shape2[:,1])
    plt.axis('equal')
    plt.show()
    path_srv = path_straightening(shape1, shape2, k)
    path = np.array([utils.SRV_to_orig(path_srv[i]) for i in range(path_srv.shape[0])])
    utils.plot_path_animation(path, title="Straightened path")

if __name__ == "__main__":
    main()