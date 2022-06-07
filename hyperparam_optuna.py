import numpy as np
import joblib
from reservoirpy.nodes import Reservoir as ReservoirPy, Ridge
from reservoirpy_model import create_data
from reservoirpy.observables import rmse
import optuna
import json

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_slice
from optuna.visualization import plot_param_importances

# Global variables for hyperparameter tuning

axon_num = 1
num_points = 100000
train_time = 10000
warmup = 148
predict_n = 10000
EXP = 1
SEED = 1234
TARGET = "RMSE"
PATH = "sample_reservoir_tuning_trials/"

# Load and process data

data = np.array(create_data(50, 0.05, axon_num, num_points))
data = data.reshape(len(data), 1)

data_train_x = data[:train_time]
data_train_y = data[1:train_time + 1]

data_warmup = data[train_time - warmup:train_time]
data_test = data[train_time + 1:train_time + predict_n + 1]

dataset = ((data_train_x, data_train_y), (data_warmup, data_test))

def objective(trial):
    # Load all datasets for each trial
    train_data, validation_data = dataset
    data_train_x, data_train_y = train_data
    data_warmup, data_test = validation_data

    # All possible hyperparameters that we tuned for the ReservoirPy model
    size = trial.suggest_int("Reservoir Size", 200, 2200)
    warmup = trial.suggest_int("Warmup", 5, 500)
    lr = trial.suggest_float("Leaking Rate", 1e-5, 1e-3)
    sr = trial.suggest_float("Spectral Radius", 1e-3, 10)
    ridge = trial.suggest_float("Ridge", 1e-9, 1)
    # input_scaling = trial.suggest_float("ridge", 1e-3, 10)

    # The model itself is initialized and trained exactly as in reservoirpy_model.py
    print(f"Size = {size}, Leaking Rate = {lr}, Spectral Radius = {sr}")
    reservoir = ReservoirPy(size, sr=sr, lr = lr, warmup=warmup, input_scaling=1)

    ridge = Ridge(ridge=ridge)

    esn = reservoir >> ridge

    esn = esn.fit(data_train_x, data_train_y, warmup=warmup)

    test_warmup = esn.run(data_warmup, reset=True)

    predictions = np.empty((predict_n, 1))
    init = test_warmup[-1].reshape(1, -1)

    for i in range(predict_n):
        init = esn(init)
        predictions[i] = init

    metric = rmse(data_test, predictions)

    return metric

study = optuna.create_study(
    # Minimize RMSE using TPE algorithm
    direction="minimize",
    # Can also perform a random search or a standard grid search
    #sampler=optuna.samplers.RandomSampler(seed=SEED),
    #sampler=optuna.samplers.GridSampler(),
    sampler=optuna.samplers.TPESampler(seed=SEED),
    # Prune trials that perform poorly
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
)

# Perform the
study.optimize(objective, n_trials=150, timeout=6000)

# Save the optimal parameters found by the study
with open(f'{PATH}/opt-params', 'w') as f:
    json.dump(study.best_params, f)


# Plot and save all visualizations of the tuning study
plot_optimization_history(study, target_name=TARGET).write_image(f"{PATH}/history.png", scale=4)
plot_parallel_coordinate(study, target_name=TARGET).write_image(f"{PATH}/parallel_coords.png", scale=4)
plot_parallel_coordinate(study, params=["Reservoir Size", "Leaking Rate"], target_name=TARGET).write_image(f"{PATH}/size_vs_lr.png", scale=4)
plot_parallel_coordinate(study, params=["Reservoir Size", "Spectral Radius"], target_name=TARGET).write_image(f"{PATH}/size_vs_sr.png", scale=4)
plot_parallel_coordinate(study, params=["Leaking Rate", "Spectral Radius"], target_name=TARGET).write_image(f"{PATH}/lr_vs_sr.png", scale=4)
plot_contour(study, target_name=TARGET).write_image(f"{PATH}/contour.png", scale=4)
plot_slice(study, target_name=TARGET).write_image(f"{PATH}/slice.png", scale=4)
plot_slice(study, target_name=TARGET).update_layout(yaxis_range=[0.1, 0.2], height=300).write_image("final_writeup_plot.png", scale=4)
plot_edf(study, target_name=TARGET).write_image(f"{PATH}/edf.png", scale=4)
plot_param_importances(study, target_name=TARGET).write_image(f"{PATH}/param_importances.png", scale=4)

# If desired, save tuning study for later usage

# joblib.dump(study, "final_plot_study.pkl")
