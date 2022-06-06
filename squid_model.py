import numpy as np
import matplotlib.pyplot as plt
import json
from reservoir import Reservoir
from reservoirpy.nodes import Reservoir as ReservoirPy, Ridge
from reservoirpy.observables import rmse, rsquare
from reservoirpy.datasets import mackey_glass
from sklearn.linear_model import LinearRegression, SGDRegressor
from wiener_filter import WienerFilter


# In most of the data, there's a time period with no impulse being applied to the axon
# This function finds the start of the impulse (and therefore the start of the interesting period of data)
def create_data(window, min_buffer, axon_num, num_points):
    raw_data = np.loadtxt(f"raw_data/axon_{axon_num}_t01_data.csv", delimiter=",")
    vmembrane_data = raw_data[:, 0]
    impulse_data = raw_data[:, 1]
    # plt.figure(figsize=(10, 3))
    # plt.plot(vmembrane_data, label="Unprocessed Data", color="blue")
    # plt.xlabel("$t$")
    # plt.ylabel("Unprocessed Data")
    # plt.show()
    moving_average = np.convolve(impulse_data, np.ones(window), 'valid') / window
    pulse_index = list(map(lambda avg: avg > min_buffer, moving_average)).index(1)
    # plt.figure(figsize=(10, 3))
    # plt.plot(vmembrane_data[pulse_index:pulse_index + num_points], label="Processed Data", color="blue")
    # plt.xlabel("$t$")
    # plt.ylabel("Processed Data")
    # plt.show()
    return vmembrane_data[pulse_index:pulse_index + num_points]

def downsample_data(axon_num, train_time=10000, temporal_resolution=50):
    data = create_data(50, 0.05, axon_num, 2 * train_time)
    train_data = data[:train_time:temporal_resolution]
    test_data = data[:2 * train_time:temporal_resolution]
    return train_data, test_data

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

def wiener_filter_predict(axon_num, train_time=10000, num_timesteps=1000):
    data = create_data(50, 0.05, axon_num, 2 * train_time)
    train_data = data[:train_time].reshape(-1, 1)
    test_data = data[:2 * train_time].reshape(-1, 1)

    wf = WienerFilter(train_data, train_data, num_timesteps)
    L = wf.optimize()

    preds = np.array([])
    overall_states = test_data[0:num_timesteps].reshape(-1, 1)
    for i in range(num_timesteps, len(test_data)):
        curr_input = np.vstack((overall_states[i - num_timesteps: i].reshape(-1, 1), 1))
        next_pred = L @ curr_input

        overall_states = np.append(overall_states, next_pred)
        preds = np.append(preds, next_pred)

    preds = preds.T
    print(preds.shape)

    plt.figure(figsize=(10, 3))
    plt.title("SQUID Training Data: Wiener Filter, No Reservoir")
    plt.xlabel("$t$")
    plt.plot(test_data[num_timesteps:3000], label="squid data", color="blue")
    plt.plot(preds[:3000-num_timesteps], label="prediction", color="red")
    plt.legend()
    #plt.show()
    plt.savefig("wiener_squid_preds3.png")


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
# test_reservoir_py(1, reservoir_size=1571, lr=0.002001591748531257, sr=0.08189176846577592, reg=1e-07, train_time=5000, warmup=150, predict_n=5000, num_points=100000, plot_data=True)
# reservoir = ReservoirPy(units=1338, lr=0.0006532585731535753, sr=0.1579788705467242)
reservoir = ReservoirPy(units=1571, lr=0.002001591748531257, sr=0.08189176846577592)
ridge = Ridge(output_dim=1, ridge=1e-07)

# reservoir <<= ridge
esn = reservoir >> ridge

axon_num = 4

train_data, data = downsample_data(axon_num, train_time=10000, temporal_resolution=1)
train_data = train_data.reshape(len(train_data), 1)
data = data.reshape(len(data), 1)

train_time = len(train_data)
warmup = 10

esn = esn.fit(data[:train_time], data[1:train_time + 1], warmup=10)

test_warmup = esn.run(data[train_time - 10:train_time], reset=True)

predictions = np.empty((train_time, 1))
init = test_warmup[-1].reshape(1, -1)

for i in range(len(train_data)):
    init = esn(init)
    predictions[i] = init

print(
    "Root mean-squared error:", rmse(data[train_time:2 * train_time + 1], predictions),
    "R^2 score:", rsquare(data[train_time:2 * train_time + 1], predictions)
)


np.savetxt(f"predictions/axon_{axon_num}_test_data.csv", data[train_time + 1:2 * train_time], delimiter=",")
np.savetxt(f"predictions/axon_{axon_num}_prediction.csv", predictions, delimiter=",")

plt.figure(figsize=(10, 3))
plt.title("SQUID Training Data: ReservoirPy")
plt.xlabel("$t$")
plt.plot(data[train_time + 1:2 * train_time], label="Test Data", color="blue")
plt.plot(predictions, label="Prediction", color="red")
plt.legend()
plt.show()
plt.savefig(f"predictions/axon_{axon_num}_timeseries.png")

# {"iss": 0.9, "lr": 0.002001591748531257, "ridge": 1e-07, "seed": 1234, "size": 871.3675133429208, "sr": 0.08189176846577592}
# {"returned_dict": {"loss": 0.1610963847649383, "normed_losses": 0.1634309265899884, "status": "ok", "start_time": 1651822136.6525052, "duration": 3.4486289024353027}, "current_params": {"iss": 1, "lr": 0.001456009573851054, "ridge": 4.364516104765995e-07, "seed": 1234, "size": 871, "sr": 0.11209653282836465}}
# {"returned_dict": {"loss": 0.14715294774045815, "normed_losses": 0.14928542707375153, "status": "ok", "start_time": 1651823203.407072, "duration": 11.450938701629639}, "current_params": {"iss": 1, "lr": 0.001456009573851054, "ridge": 0.009181821500049373, "seed": 1234, "size": 2039.8446221301288, "sr": 0.7187356743957771}}
# test_reservoir_py(4, reservoir_size=1616, lr=0, sr=0.20682435846372868, reg = 0.9699098551710091, train_time=10000, warmup=148, predict_n=10000, num_points=100000, plot_data=True)

# Hyperparameter tuning

# 1. Tune reservoir size

# size_rms = {}
# for size in range(100, 2200, 50):
#     rms = test_reservoir_py(1, reservoir_size=size, lr=0.0015, sr=0.72, reg = 1e-7, train_time=5000, warmup=100, predict_n=5000, num_points=100000)
#     size_rms[size] = rms
#
# json = json.dumps(size_rms)
# f = open("optimization/size_grid_search.json","w")
# f.write(json)
# f.close()

# 2. Tune regularization of Ridge regression

# ridge_rms = {}
# regs = np.logspace(1e-9, 1, 40)
# for reg in regs:
#     rms = test_reservoir_py(1, reservoir_size=870, lr=0.0015, sr=0.72, reg = reg, train_time=5000, warmup=100, predict_n=5000, num_points=100000)
#     ridge_rms[reg] = rms
#
# json = json.dumps(ridge_rms)
# f = open("optimization/ridge_grid_search.json","w")
# f.write(json)
# f.close()

# 3. Tune spectral radius of initial random matrices

# spectral_rms = {}
# srs = np.logspace(1e-3, 1, 40)
# for sr in srs:
#     rms = test_reservoir_py(1, reservoir_size=870, lr=0.0015, sr=sr, reg = 1e-7, train_time=5000, warmup=100, predict_n=5000, num_points=100000)
#     spectral_rms[sr] = rms
#
# json = json.dumps(spectral_rms)
# f = open("optimization/spectral_rad_grid_search.json","w")
# f.write(json)
# f.close()

# test_data = np.loadtxt(f"predictions/axon_1_test_data.csv", delimiter=",")
# predictions = np.loadtxt(f"predictions/axon_1_prediction.csv", delimiter=",")

# wiener_filter_predict(1)

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
