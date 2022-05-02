import numpy as np
import matplotlib.pyplot as plt
from reservoir import Reservoir
from reservoirpy.nodes import Reservoir as ReservoirPy, Ridge
from reservoirpy.observables import rmse, rsquare
from reservoirpy.datasets import mackey_glass
from sklearn.linear_model import LinearRegression, SGDRegressor


# In most of the data, there's a time period with no impulse being applied to the axon
# This function finds the start of the impulse (and therefore the start of the interesting period of data)
def create_data(window, min_buffer, axon_num, num_points):
    raw_data = np.loadtxt(f"raw_data/axon_{axon_num}_t01_data.csv", delimiter=",")
    vmembrane_data = raw_data[:, 0]
    impulse_data = raw_data[:, 1]
    moving_average = np.convolve(impulse_data, np.ones(window), 'valid') / window
    pulse_index = list(map(lambda avg: avg > min_buffer, moving_average)).index(1)

    return vmembrane_data[pulse_index:pulse_index + num_points]


# Preliminary predictions on axon data
def squid_predict(axon_num, train_time=400000):
    data = create_data(50, 0.05, axon_num, 2 * train_time)
    train_data = data[:train_time]
    test_data = data[:2 * train_time]

    res = Reservoir(1, 200)
    states_training = res.train(train_data)
    model = SGDRegressor(penalty='l2', alpha=0.00001, verbose=1)
    model.fit(states_training, train_data)
    prediction = res.run(2 * train_time, model)

    np.savetxt(f"predictions/axon_{axon_num}_test_data.csv", test_data, delimiter=",")
    np.savetxt(f"predictions/axon_{axon_num}_prediction.csv", prediction, delimiter=",")

    plt.figure(figsize=(10, 3))
    plt.title("SQUID Training Data")
    plt.xlabel("$t$")
    plt.plot(test_data, label="Test Data", color="blue")
    plt.plot(prediction, label="Prediction", color="red")
    plt.legend()
    plt.show()

def test_reservoir_py(axon_num, train_time, warmup, num_points):
    reservoir = ReservoirPy(units=20, lr=0.3, sr=1.25)
    readout = Ridge(output_dim=1, ridge=1e-5)
    esn = reservoir >> readout

    data = np.array(create_data(50, 0.05, axon_num, num_points))
    data = data.reshape(len(data), 1)

    esn.fit(data[:train_time], data[1:train_time + 1], warmup=warmup)
    predictions = esn.run(data[train_time + 1:-1])

    print(
        "Root mean-squared error:", rmse(data[train_time + 2:], predictions),
        "R^2 score:", rsquare(data[train_time + 2:], predictions)
    )

    np.savetxt(f"predictions/axon_{axon_num}_test_data.csv", data[train_time + 2:], delimiter=",")
    np.savetxt(f"predictions/axon_{axon_num}_prediction.csv", predictions, delimiter=",")

    plt.figure(figsize=(10, 3))
    plt.title("SQUID Training Data: ReservoirPy")
    plt.xlabel("$t$")
    plt.plot(np.arange(0, len(data), 1), data, label="Test Data", color="blue")
    plt.plot(np.arange(train_time + 2, len(data), 1), predictions, label="Prediction", color="red")
    plt.legend()
    plt.show()


# squid_predict(1)
test_reservoir_py(1, train_time=1000, warmup=100, num_points=100000)

test_data = np.loadtxt(f"predictions/axon_1_test_data.csv", delimiter=",")
predictions = np.loadtxt(f"predictions/axon_1_prediction.csv", delimiter=",")

# plt.figure(figsize=(10, 3))
# plt.title("SQUID Training Data: ReservoirPy")
# plt.xlabel("$t$")
# plt.plot(test_data, label="Test Data", color="blue")
# plt.plot(predictions, label="Prediction", color="red")
# plt.legend()
# plt.show()

# Mackey Glass toy example

# X = mackey_glass(n_timesteps=2000)
# reservoir = ReservoirPy(units=100, lr=0.3, sr=1.25)
# readout = Ridge(output_dim=1, ridge=1e-5)
# esn = reservoir >> readout
# esn.fit(X[:500], X[1:501], warmup=100)
# predictions = esn.fit(X[:500], X[1:501]).run(X[501:-1])
# print("RMSE:", rmse(X[502:], predictions), "R^2 score:", rsquare(X[502:], predictions))
