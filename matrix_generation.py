import numpy as np
from numpy.random import Generator
from scipy import sparse, stats


def _random_sparse(n: int, m: int,
                   connectivity: float = 1.0,
                   dtype: np.dtype = np.float64,
                   sparsity_type: str = "csr",
                   random_generator: np.random.Generator = np.random.default_rng,
                   dist=stats.norm.rvs
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
