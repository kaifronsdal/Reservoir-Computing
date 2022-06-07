import numpy as np
import matplotlib.pyplot as plt
from reservoir import Reservoir
from reservoirpy.nodes import Reservoir as ReservoirPy, Ridge
from reservoirpy.observables import rmse
from sklearn.linear_model import SGDRegressor


# In most of the data, there's a time period with no impulse being applied to the axon
# This function finds the start of the impulse (and therefore the start of the interesting period of data)
def create_data(window, min_buffer, axon_num, num_points):
    # Load all data
    raw_data = np.loadtxt(f"raw_data/axon_{axon_num}_t01_data.csv", delimiter=",")
    vmembrane_data = raw_data[:, 0]
    impulse_data = raw_data[:, 1]

    # Convolve data to find index of first spike
    moving_average = np.convolve(impulse_data, np.ones(window), 'valid') / window
    pulse_index = list(map(lambda avg: avg > min_buffer, moving_average)).index(1)

    # Return processed data
    return vmembrane_data[pulse_index:pulse_index + num_points]

# Downsample data by a preset temporal resolution to make comparisons with LSTM predictions
def downsample_data(axon_num, train_time=10000, temporal_resolution=50):
    data = create_data(50, 0.05, axon_num, 2 * train_time)
    train_data = data[:train_time:temporal_resolution]
    test_data = data[:2 * train_time:temporal_resolution]
    return train_data, test_data

# Implement and test our own reservoir computing network, built from scratch
def squid_predict(axon_num, train_time=400000):
    # Load data
    data = create_data(50, 0.05, axon_num, 2 * train_time)
    train_data = data[:train_time]
    test_data = data[:2 * train_time]

    # Create a reservoir with 200 nodes and train
    res = Reservoir(1, 200)
    states_training = res.train(train_data)
    # Initialize regularization and fit output layer using ridge regression
    model = SGDRegressor(penalty='l2', alpha=0.00001, verbose=1)
    model.fit(states_training, train_data)
    # Make predictions on the test data
    prediction = res.run(2 * train_time, model)

    # Save the resulting predictions, if desired
    # np.savetxt(f"sample_reservoir_predictions/axon_{axon_num}_test_data.csv", test_data, delimiter=",")
    # np.savetxt(f"sample_reservoir_predictions/axon_{axon_num}_prediction.csv", prediction, delimiter=",")

    # Plot predictions against the test data
    plt.figure(figsize=(10, 3))
    plt.title("SQUID Training Data")
    plt.xlabel("$t$")
    plt.plot(test_data, label="Test Data", color="blue")
    plt.plot(prediction, label="Prediction", color="red")
    plt.legend()
    plt.show()

# Implement ReservoirPy's reservoir computing framework
def test_reservoir_py(axon_num, reservoir_size, lr, sr, reg, train_time, predict_n, warmup, num_points, save_data=False, plot_data=False):
    # Initialize reservoir with the input hyperparameters
    reservoir = ReservoirPy(units=reservoir_size, lr=lr, sr=sr, iss = 0.9)

    # Initialize ridge regression on output states
    ridge = Ridge(output_dim=1, ridge=reg)

    # Create the echo state networ by connecting the reservoir to output ridge regression
    esn = reservoir >> ridge

    # Load training and test data
    data = np.array(create_data(50, 0.05, axon_num, num_points))
    data = data.reshape(len(data), 1)

    # Catch invalid prediction lengths
    if (predict_n + train_time >= len(data)):
        raise Exception(f"Cannot predict more than {len(data) - train_time} time-steps!")

    # Fit the echo state network to the training data
    esn = esn.fit(data[:train_time], data[1:train_time + 1], warmup=warmup)

    # On the warmup period, the reservoir does not make predictions
    test_warmup = esn.run(data[train_time - warmup:train_time], reset=True)

    # Make predictions, feeding each prediction back into the reservoir to make the next
    predictions = np.empty((predict_n, 1))
    init = test_warmup[-1].reshape(1, -1)

    for i in range(predict_n):
        init = esn(init)
        predictions[i] = init

    # Output evaluation metric, RMSE
    print(
        "Root mean-squared error:", rmse(data[train_time + 1:train_time + predict_n + 1], predictions)
    )

    # If specified, save and plot the predictions against test data
    if save_data:
        np.savetxt(f"sample_reservoir_predictions/axon_{axon_num}_test_data.csv", data[train_time + 1:train_time + predict_n], delimiter=",")
        np.savetxt(f"sample_reservoir_predictions/axon_{axon_num}_prediction.csv", predictions, delimiter=",")

    if plot_data:
        plt.figure(figsize=(8, 6))
        plt.title("Reservoir Computing Network Predictions on SGAMP Axon 1")
        plt.xlabel("Timesteps $(t)$")
        plt.ylabel("Membrane Potential ($V$)")
        plt.plot(data[train_time + 1:train_time + predict_n], label="Test Data", color="blue")
        plt.plot(predictions, label="Prediction", color="red")
        plt.legend()
        plt.show()
        plt.savefig(f"sample_reservoir_predictions/axon_{axon_num}_timeseries.png")


# Example usage of our reservoir computing model, built from scratch, on Axon 1 of SGAMP
squid_predict(1)

# Example usage of ReservoirPy model on Axon 1 of SGAMP, with optimal hyperparameters
test_reservoir_py(1, reservoir_size=1338, lr=0.000653, sr=0.158, reg=1.53e-07, train_time=10000, warmup=148, predict_n=10000, num_points=100000, plot_data=False)