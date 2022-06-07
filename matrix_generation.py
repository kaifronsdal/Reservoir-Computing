import numpy as np
from numpy.random import Generator
from scipy import sparse, stats


def _random_sparse(n: int, m: int,
                   connectivity: float = 0.1,
                   dtype: np.dtype = np.float64,
                   sparsity_type: str = "csr",
                   random_generator: np.random.Generator = np.random.default_rng(),
                   dist=stats.uniform(loc=-0.5, scale=1.).rvs
                   ):
    """
    computes a random sparse matrix
    :param n: matrix width
    :param m: matrix height
    :param connectivity: proportion of non-zero entries
    :param dtype: data type to fill matrix with
    :param sparsity_type: sparse matrix format (see scipy docs)
    :param random_generator: random number generator to use
    :param dist: distribution to sample from to fill non-zero entries with
    :return: a sparse random matrix
    """
    matrix = sparse.random(
        n, m,
        density=connectivity,
        format=sparsity_type,
        random_state=random_generator,
        data_rvs=dist,
        dtype=dtype,
    )
    # sparse.random may return np.matrix if format="dense".
    if type(matrix) is np.matrix:
        matrix = np.asarray(matrix)

    return matrix


def _set_spectral_radius(matrix: np.ndarray, radius: float):
    """
    Rescales matrix to have given spectral radius
    :param matrix: matrix to scale
    :param radius: the new spectral radius
    :return: matrix with given spectral radius
    """
    max_eigenvalue = np.max(np.abs(sparse.linalg.eigs(matrix)[0]))
    matrix *= radius / max_eigenvalue
    return matrix
