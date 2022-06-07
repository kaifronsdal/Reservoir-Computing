import numpy as np
import matplotlib.pyplot as plt
from reservoirpy_model import create_data

# contains code to create a Wiener Filter to make predictions on squid data,
# incorporating some number of previous timesteps of data
class WienerFilter:
    # Enter 'input_states' with each state as a row vector and 'output_states' as a column vector, with each
    # entry corresponding to corresponding row of 'input_states'. 'num_timesteps' corresponds to amount of history
    # used in making predictions.
    def __init__(self, input_states, output_states, num_timesteps):
        assert input_states.shape[0] == output_states.shape[0]
        self.X = input_states
        self.Y = output_states
        self.timesteps = num_timesteps
        self.L = np.zeros((self.Y.shape[1], self.X.shape[0] - self.timesteps))

    # create large Wiener filter matrix and use it to obtain optimal filtering matrix
    def optimize(self):
        X_large = self.get_large_X_matrix()
        Y_out = self.Y.T[:, self.timesteps:]
        self.L = Y_out @ np.linalg.pinv(X_large)
        return self.L

    # creates large Wiener filter matrix
    def get_large_X_matrix(self):
        input_states = self.X.T
        N = input_states.shape[1]

        X_large = np.ones((1, N - self.timesteps))

        for i in range(self.timesteps):
            curr_matrix = input_states[:, i: i + N - self.timesteps]
            X_large = np.vstack((curr_matrix, X_large))

        return X_large

# train a Wiener filter to make predictions for timesteps 10,000-20,000 of axon data for Axon 'axon_num'
def wiener_filter_predict(axon_num, train_time=10000, num_timesteps=1000):
    data = create_data(50, 0.05, axon_num, 2 * train_time)
    train_data = data[:train_time].reshape(-1, 1)
    test_data = data[train_time+1:2 * train_time].reshape(-1, 1)

    # create and train Wiener filter
    wf = WienerFilter(train_data, train_data, num_timesteps)
    L = wf.optimize()

    # form predictions for timesteps 10,000-20,000, using earlier predictions as inputs for making
    # later predictions
    preds = np.array([])
    overall_states = test_data[0:num_timesteps].reshape(-1, 1)
    for i in range(num_timesteps, len(test_data)):
        curr_input = np.vstack((overall_states[i - num_timesteps: i].reshape(-1, 1), 1))
        next_pred = L @ curr_input

        overall_states = np.append(overall_states, next_pred)
        preds = np.append(preds, next_pred)

    preds = preds.T
    print(preds.shape)

    # plot predictions made by Wiener filter against actual test data
    plt.figure(figsize=(10, 3))
    plt.title("Wiener Filter Predictions on SGAMP Axon 1")
    plt.xlabel("Time $(t)$")
    plt.ylabel("Membrane Potential ($V$)")
    plt.plot(test_data[num_timesteps:train_time], label="Test Data", color="blue")
    plt.plot(preds[:train_time-num_timesteps], label="Wiener Filter Prediction", color="red")
    plt.legend()
    plt.show()
    plt.savefig("wiener_filter_preds.png")