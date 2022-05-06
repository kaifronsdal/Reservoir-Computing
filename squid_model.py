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

def test_reservoir_py(axon_num, reservoir_size, lr, sr, reg, train_time, predict_n, warmup, num_points, save_data=False, plot_data=False):
    reservoir = ReservoirPy(units=reservoir_size, lr=lr, sr=sr)
    ridge = Ridge(output_dim=1, ridge=reg)

    # reservoir <<= ridge
    esn = reservoir >> ridge

    data = np.array(create_data(50, 0.05, axon_num, num_points))
    data = data.reshape(len(data), 1)

    if (predict_n + train_time >= len(data)):
        raise Exception(f"Cannot predict more than {len(data) - train_time} time-steps!")

    esn = esn.fit(data[:train_time], data[1:train_time + 1], warmup=warmup)

    test_warmup = esn.run(data[train_time - warmup:train_time], reset=True)

    predictions = np.empty((predict_n, 1))
    init = test_warmup[-1].reshape(1, -1)

    for i in range(predict_n):
        init = esn(init)
        predictions[i] = init

    print(
        "Root mean-squared error:", rmse(data[train_time + 1:train_time + predict_n + 1], predictions),
        "R^2 score:", rsquare(data[train_time + 1:train_time + predict_n + 1], predictions)
    )

    if save_data:
        np.savetxt(f"predictions/axon_{axon_num}_test_data.csv", data[train_time + 1:train_time + predict_n], delimiter=",")
        np.savetxt(f"predictions/axon_{axon_num}_prediction.csv", predictions, delimiter=",")

    if plot_data:
        plt.figure(figsize=(10, 3))
        plt.title("SQUID Training Data: ReservoirPy")
        plt.xlabel("$t$")
        plt.plot(data[train_time + 1:train_time + predict_n], label="Test Data", color="blue")
        plt.plot(predictions, label="Prediction", color="red")
        plt.legend()
        plt.show()
        plt.savefig(f"predictions/axon_{axon_num}_timeseries.png")


# squid_predict(1)
# {"iss": 0.9, "lr": 0.002001591748531257, "ridge": 1e-07, "seed": 1234, "size": 871.3675133429208, "sr": 0.08189176846577592}
# test_reservoir_py(1, reservoir_size=871, lr=0.002001591748531257, sr=0.08189176846577592, reg = 1e-7, train_time=5000, warmup=100, predict_n=5000, num_points=100000, plot_data=True)

# Hyperparameter tuning

# 1. Tune reservoir size


#for size in range(100, 2200, 100):
    #test_reservoir_py(1, reservoir_size=size, lr=0.5, sr=0.9, reg = 1e-7, train_time=10000, warmup=100, predict_n=10000, num_points=100000)

# test_data = np.loadtxt(f"predictions/axon_1_test_data.csv", delimiter=",")
# predictions = np.loadtxt(f"predictions/axon_1_prediction.csv", delimiter=",")

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
