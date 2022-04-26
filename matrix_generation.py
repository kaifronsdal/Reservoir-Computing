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


def _set_spectal_radius(matrix: np.ndarray, radius: float):
    max_eigenvalue = np.max(np.abs(sparse.linalg.eigs(matrix)[0]))
    matrix *= radius / max_eigenvalue
    return matrix
