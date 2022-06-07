import numpy as np
import json
from reservoirpy.hyper import research, plot_hyperopt_report
from reservoirpy.nodes import Reservoir as ReservoirPy, Ridge
from reservoirpy_model import create_data
from reservoirpy.observables import rmse, nrmse

# Global variables for hyperparameter tuning with hyperopt

axon_num = 1
num_points = 100000
train_time = 10000
warmup = 100
predict_n = 10000

# Create all training and test data using the framework provided in reservoirpy_model.py
data = np.array(create_data(50, 0.05, axon_num, num_points))
data = data.reshape(len(data), 1)

data_train_x = data[:train_time]
data_train_y = data[1:train_time + 1]

data_warmup = data[train_time - warmup:train_time]
data_test = data[train_time + 1:train_time + predict_n + 1]

dataset = ((data_train_x, data_train_y), (data_warmup, data_test))

def objective(dataset, config, *, iss, size, sr, lr, ridge, seed):

    # Load all datasets
    train_data, validation_data = dataset
    data_train_x, data_train_y = train_data
    data_warmup, data_test = validation_data

    # Number of instances per trial
    instances = config["instances_per_trial"]

    # Change initialization seed across tests to prevent bias
    variable_seed = seed

    # Save losses (categorized as RMSE and normed RMSE) in two vectors
    losses = []; normed_losses = [];

    # Perform random search for optimal hyperparameters using ReservoirPy's built-in compatibility with hyperopt
    for n in range(instances):
        print(f"Size = {int(size)}, sr = {sr}, lr = {lr}")

        # The reservoir and ESN are initialized exactly as in reservoirpy_model.py
        reservoir = ReservoirPy(int(size), sr=sr, lr=lr, input_scaling=iss, seed=variable_seed)

        ridge = Ridge(ridge=ridge)

        esn = reservoir >> ridge

        esn = esn.fit(data_train_x, data_train_y, warmup=warmup)

        test_warmup = esn.run(data_warmup, reset=True)

        predictions = np.empty((predict_n, 1))
        init = test_warmup[-1].reshape(1, -1)

        for i in range(predict_n):
            init = esn(init)
            predictions[i] = init

        # Save both metrics to vectors
        loss = rmse(data_test, predictions)
        normed_loss = nrmse(data_test, predictions, norm="minmax")

        # Change seed to sample a wider variety of hyperparameters
        variable_seed += 1
        losses.append(loss)
        normed_losses.append(normed_loss)
    return {'loss': float(np.mean(losses)), 'normed_losses': float(np.mean(normed_losses))}

# Example tuning configuration for hyperopt

hyperopt_config = {
    "exp": "hyperopt-squid-uniform", # the experimentation name
    "hp_max_evals": 100,             # the number of different sets of parameters hyperopt has to try
    "hp_method": "random",           # the method used by hyperopt to choose those sets (see below)
    "seed": 42,                      # the random state seed, to ensure reproducibility
    "instances_per_trial": 1,        # how many random ESN will be tried with each sets of parameters
    "hp_space": {                    # what are the ranges of parameters explored
        "size": ["uniform", 700, 2500],             # the number of neurons is fixed to 300
        "sr": ["uniform", 1e-3, 1],   # the spectral radius is log-uniformly distributed between 1e-6 and 10
        "lr": ["choice", 0.001456009573851054],  # idem with the leaking rate, from 1e-3 to 1
        "iss": ["choice", 1],           # the input scaling is fixed
        "ridge": ["uniform", 1e-7, 1e-2],        # and so is the regularization parameter.
        "seed": ["choice", 1234]          # another random seed for the ESN initialization
    }
}


# Save hyperopt configuration

with open(f"sample_reservoir_tuning_trials/{hyperopt_config['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config, f)

# Perform sample_reservoir_tuning_trials step via random search

best = research(objective, dataset, f"sample_reservoir_tuning_trials/{hyperopt_config['exp']}.config.json",
                f"sample_reservoir_tuning_trials")
print(best)

# Show plots for sample_reservoir_tuning_trials results

fig = plot_hyperopt_report(f"optimization/{hyperopt_config['exp']}", ["lr", "sr", "ridge"], metric="loss", loss_metric="loss")
fig.savefig(f"optimization/{hyperopt_config['exp']}/hyperopt_plot.png")