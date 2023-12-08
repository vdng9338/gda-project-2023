Data types
==========

We use `numpy.ndarray`s to represent curves, vector fields, geodesics, etc.

We will use the following notations in this page:
- `k` will denote the number of discretization steps for paths of curves (same notation as in the paper).
- `T` will denote the number of discretization steps for curves.
- `n` will denote the dimension of the ambient space (same notation as in the paper).

The types that we use are as follows:

- Curves, SRV representations, tangent vectors at curves (functions from $[0, 1]$ to $\mathbb{R}^n$, discretized): `numpy.ndarray` of shape `(T+1, n)`.
- Paths, vector fields along paths (functions from $[0, 1]$ to functions from $[0, 1]$ to $\mathbb{R}^n$): `numpy.ndarray` of shape `(k+1, T+1, n)`.
- Orthonormal basis of $N_q(\mathcal{C}^c)$: `numpy.ndarray` of shape `(n, T+1, n)`.