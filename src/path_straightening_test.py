from path_straightening import *
import utils
import matplotlib.pyplot as plt
import numpy as np

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
    T = 100
    k = 100
    ellipse = gen_ellipse(a = 2.0, b = 0.5, T = T)
    circle = gen_ellipse(a = 1.0, b = 1.0, T = T)
    plt.plot(ellipse[:,0], ellipse[:,1])
    plt.plot(circle[:,0], circle[:,1])
    plt.axis('equal')
    plt.show()
    path_srv = path_straightening(ellipse, circle, k)
    path = np.array([utils.SRV_to_orig(path_srv[i]) for i in range(path_srv.shape[0])])
    utils.plot_path_animation(path, title="Straightened path")

if __name__ == "__main__":
    main()