import numpy as np
import matplotlib.pyplot as plt
from reservoir import Reservoir
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor


# In most of the data, there's a time period with no impulse being applied to the axon
# This function finds the start of the impulse (and therefore the start of the interesting period of data)
def find_pulse_start_index(impulse_data, window, min_buffer):
    moving_average = np.convolve(impulse_data, np.ones(window), 'valid') / window
    return list(map(lambda avg: avg > min_buffer, moving_average)).index(1)


# Preliminary predictions on axon data
def squid_predict(axon_num, train_time=400000):
    raw_data = np.loadtxt(f"raw_data/axon_{axon_num}_t01_data.csv", delimiter=",")
    vmembrane_data = raw_data[:, 0]
    impulse_data = raw_data[:, 1]

    pulse_index = find_pulse_start_index(impulse_data, 50, 0.05)
    train_data = vmembrane_data[pulse_index:pulse_index + train_time]
    test_data = vmembrane_data[pulse_index:pulse_index + 2 * train_time]

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


squid_predict(1)
