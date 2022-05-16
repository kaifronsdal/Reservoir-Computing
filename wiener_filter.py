import numpy as np

class WienerFilter:
    # enter 'input_states' with each state as a row vector and 'output_states' as a column vector, with each
    # entry corresponding to corresponding row of 'input_states'
    def __init__(self, input_states, output_states, num_timesteps):
        assert input_states.shape[0] == output_states.shape[0]
        self.X = input_states
        self.Y = output_states
        self.timesteps = num_timesteps
        self.L = np.zeros((self.Y.shape[1], self.X.shape[0] - self.timesteps))

    def optimize(self):
        X_large = self.get_large_X_matrix()
        Y_out = self.Y.T[:, self.timesteps:]
        self.L = Y_out @ np.linalg.pinv(X_large)
        return self.L

    def get_large_X_matrix(self):
        input_states = self.X.T
        N = input_states.shape[1]

        X_large = np.ones((1, N - self.timesteps))

        for i in range(self.timesteps):
            curr_matrix = input_states[:, i: i + N - self.timesteps]
            X_large = np.vstack((curr_matrix, X_large))

        return X_large



