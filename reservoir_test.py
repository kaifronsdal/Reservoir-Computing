import numpy as np
import matplotlib.pyplot as plt
from reservoir import Reservoir
from sklearn.linear_model import LinearRegression, Ridge
from wiener_filter_model import WienerFilter
from matrix_generation import _random_sparse

def main():
    # squid_model.squid_predict("neuron_1_t01_data", 400000)
    # print(_random_sparse(3, 3))
    r = Reservoir(1, 20)
    # X = r.run(np.arange(0, 10, 1))
    X = np.sin(np.linspace(0, 100 * np.pi, 40000)).reshape(-1, 1) + np.sin(np.linspace(0, 2320 * np.pi, 40000)).reshape(
        -1, 1)
    states = r.train(X)
    #model = Ridge(alpha=0.0000001)
    #model.fit(states, X)
    X = np.sin(np.linspace(0, 400 * np.pi, 4 * 40000)).reshape(-1, 1) + np.sin(
        np.linspace(0, 4 * 2320 * np.pi, 4 * 40000)).reshape(-1, 1)
    states = r.train(X)
    # pred = r.run(100, model)
    #pred = model.predict(states)

    # Now, let's predict using Wiener Filter
    #---------------------------------------
    # num_timesteps = 100
    # wf = WienerFilter(states, X, num_timesteps)
    # L = wf.optimize()
    # print(L.shape)
    # pred = L @ wf.get_large_X_matrix()
    # pred = pred.T
    # print("Processing complete")
    #---------------------------------------
    num_timesteps = 100
    start_ind = np.random.randint(0, len(X) - 5101)
    X_train = X[start_ind: start_ind + 5001]
    # next_states = X[start_ind + 1: start_ind + 5002].reshape(-1,1)
    X_train = X_train.reshape(-1, 1)
    wf = WienerFilter(X_train, X_train, num_timesteps)
    L = wf.optimize()
    print(L.shape)

    preds = np.array([])
    overall_states = X[0:num_timesteps].reshape(-1, 1)
    for i in range(num_timesteps, 10000):
        curr_input = np.vstack((overall_states[-num_timesteps:].reshape(-1, 1), 1))
        next_pred = L @ curr_input
        overall_states = np.append(overall_states, next_pred)
        #print(overall_states)
        #print(next_pred)
        preds = np.append(preds, next_pred)
    preds = preds.T
