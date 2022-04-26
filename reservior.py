import numpy as np
from matrix_generation import _random_sparse
from scipy import stats
from functools import partial


class Reservoir:
    def __init__(self, n_input: int,
                 n_reservoir: int,
                 # n_output: int,
                 alpha: int = 0.1,
                 dtype: np.dtype = np.float64,
                 spectral_radius: int = 1.0,
                 reservoir_density: int = 0.2,
                 random_seed=None,
                 activation=np.sigmoid,
                 input_density: int = 1.0):
        """
            n_input : Size of the input.
            n_reservoir : Number of units in the reservoir.
            n_output : Size of the output.
            spectral_radius : Spectral radius of the reservoir matrix.
            reservoir_density : Proportion of non-zero weight connections in the reservoir.
            random_seed : Seed for random weight initialization.
            input_density : Proportion of non-zero weights between the input and reservoir matrix.
            activation : Activation function.
        """
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        # self.n_output = n_output

        self._spectral_radius = spectral_radius
        self._reservoir_density = reservoir_density
        self.input_density = input_density
        self._activation = activation
        self._random_seed = random_seed

        self.alpha = alpha
        # self.state = None
        # self.W = None
        # self.W_in = None
        # self.W_out = None
        self.dtype = dtype

        self.W = _random_sparse(self.n_reservoir, self.n_reservoir, self._reservoir_density,
                                dist=partial(stats.uniform.rvs, loc=-0.5, scale=1))
        
        self.W_in = _random_sparse(self.n_reservoir, self.n_input, self._reservoir_density,
                                   dist=partial(stats.uniform.rvs, loc=-0.5, scale=1))

    def forward_internal(self, x: np.ndarray) -> np.ndarray:
        next_state = (
                (1 - self.alpha) * self.state
                + self.alpha * self._activation(self.W @ self.state + self.W_in @ x)
        )
        return next_state

    def reset(self, to_state: np.ndarray = None):
        if to_state is None:
            self.state = self.zero_state()
        else:
            self.state = to_state

    def zero_state(self):
        return np.zeros((1, self.n_reservoir), dtype=self.dtype)



