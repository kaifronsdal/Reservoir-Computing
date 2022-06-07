import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In most of the data, there's a time period with no impulse being applied to the axon
# This function finds the start of the impulse (and therefore the start of the interesting period of data)
def create_data(window, min_buffer, axon_num, num_points):
    # get time series data for Axon 'axon_num' and extract given 'window' of data
    raw_data = np.loadtxt(f"raw_data/axon_{axon_num}_t01_data.csv", delimiter=",")
    vmembrane_data = raw_data[:, 0]
    impulse_data = raw_data[:, 1]
    moving_average = np.convolve(impulse_data, np.ones(window), 'valid') / window
    pulse_index = list(map(lambda avg: avg > min_buffer, moving_average)).index(1)

    return vmembrane_data[pulse_index:pulse_index + num_points]

# gets '2*train_time' timesteps of data from Axon 'axon_num', with first half being training data
# and second half being testing data
def get_data(axon_num, train_time=10000, temporal_resolution=50):
    data = create_data(50, 0.05, axon_num, 2 * train_time)

    # downsample data by factor of 'temporal_resolution' if necessary
    train_data = data[:train_time:temporal_resolution]
    test_data = data[train_time+1:2 * train_time:temporal_resolution]
    return train_data, test_data


# forms a sliding window matrix, each of whose rows is a sequence of 'sequence_length' consecutive
# timesteps; also forms a matrix, each of whose rows is the next 'pred_length' timesteps of the data
# after the previous 'sequence_length' timesteps from the corresponding row of the first matrix
def getSequentialData(data, sequence_length, pred_length):
    sequential_data = np.empty((0, sequence_length))
    next_value = []
    prediction_length = pred_length
    for i in range(len(data) - sequence_length - 1 - prediction_length):
        # add row of 'sequence_length' timesteps to first matrix
        sequential_data = np.vstack((sequential_data, data[i:i + sequence_length].reshape(1, -1)))

        # add row of the following 'pred_length' timesteps of data to the second matrix
        next_value.append(data[i + sequence_length:i + sequence_length + prediction_length])
    return sequential_data, np.array(next_value)


# sequences the data, reshapes it, and converts it into Variable Tensor form
def preprocess_data(data, sequence_length, pred_length):
    sequential_x, next_y = getSequentialData(data, sequence_length, pred_length)

    x = Variable(torch.Tensor(sequential_x))
    x = torch.reshape(x, (x.size(0), x.size(1), 1))

    y = Variable(torch.Tensor(next_y))
    y = torch.reshape(y, (y.size(0), y.size(1)))

    return x, y

# code for our initializing and forward propagation in our one-hidden-layer LSTM, which inherits properties
# from 'torch.nn' library
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length, pred_length):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.pred_length = pred_length

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    # pass input through LSTM to obtain prediction for just the next timestep of data
    def step(self, x):
        # initialize the hidden and cell states of the LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # update hidden and cell states based on LSTM cell equations
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        h_n = h_n.view(-1, self.hidden_size)

        # pass hidden units through linear layer to get output
        out = self.linear(h_n)

        return out

    # perform 'step' method 'pred_length' number of times to obtain next 'pred_length' number of predictions;
    # note that earlier predictions are fed back into the model to help make later predictions
    def forward(self, x):
        out = self.step(x)
        out_3D = torch.reshape(out, (out.size(0), out.size(1), 1))

        # 'x' will store all of the predictions
        x = torch.cat((x, out_3D), 1)

        # keep obtaining next prediction and adding it to 'x'
        for i in range(self.pred_length - 1):
            next_out = self.step(x[:, i + 1:])
            out_3D = torch.reshape(next_out, (next_out.size(0), next_out.size(1), 1))
            x = torch.cat((x, out_3D), 1)
            out = torch.cat((out, next_out), 1)
        return out

# trains an LSTM model (whether from scratch or an existing model) based on a choice of hyperparameters
def train_model(x_train, y_train, input_size, hidden_size, num_layers, pred_length, n_epochs=10001, lr=1e-3, saved_lstm=None):
    sequence_length = x_train.size(1)

    if saved_lstm is None:
        lstm = LSTM(input_size, hidden_size, num_layers, sequence_length, pred_length)
    else:
        lstm = saved_lstm

    # train LSTM by minimizing mean squared error loss; also using 'Adam' optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr=lr)

    # train model (using backprop) for 'n_epochs' epochs
    for epoch in range(n_epochs):
        y_out = lstm.forward(x_train)
        optimizer.zero_grad()

        loss = criterion(y_out, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch: {epoch} | loss: {round(loss.item(), 4)}")

    return lstm

# obtains predicted timeseries for the test data
def predict(lstm, x_test, y_test, seq_len, savepath=None):
    # get actual test data
    y_test_np = y_test.detach().numpy()[:, 0]
    y_test_np = y_test_np.reshape(-1)

    # get predicted time series data
    y_preds = np.zeros_like(y_test_np)
    x_so_far = x_test[0, :, :].detach().numpy()
    x_so_far = x_so_far.reshape(-1)

    x_curr = x_test[0, :, :]
    x_curr = Variable(torch.reshape(x_curr, (1, seq_len, x_test.size(2))))

    for i in range(len(y_preds)):
        next_pred_tensor = lstm.step(x_curr)
        next_pred = next_pred_tensor.detach().numpy()
        next_pred = next_pred.reshape(-1)[0]
        y_preds[i] = next_pred

        x_so_far = np.append(x_so_far, next_pred)
        x_curr = Variable(torch.tensor(x_so_far[-seq_len:]))
        x_curr = torch.reshape(x_curr, (1, seq_len, x_test.size(2)))

    # make plot of LSTM predictions on test data versus actual test data
    plt.figure(figsize=(10, 3))
    plt.title("SQUID Prediction (Testing) with LSTM")
    plt.xlabel("$t$")
    plt.plot(y_test_np, label="squid data", color="blue")
    plt.plot(y_preds, label="prediction", color="red")
    plt.legend()
    if savepath is None:
        savepath = "lstm_preds.png"
    plt.savefig(savepath)

    # compute mean squared error between predicted and actual timeseries
    mse = (1.0 / len(y_preds)) * np.sum((y_preds - y_test_np) ** 2)
    return mse

# obtain training and testing data from, say, Axon 1
train_data, test_data = get_data(1)

# set hyperparameters for LSTM model
pred_length = 5
seq_len = 50
hidden_size = 16
input_size = 1
num_layers = 1

# obtain preprocessed training and testing data
x_train, y_train = preprocess_data(train_data, seq_len, pred_length)
x_test, y_test = preprocess_data(test_data, seq_len, pred_length)

# train LSTM models based on given hyperparameters
lstm = train_model(x_train, y_train, input_size, hidden_size, num_layers, pred_length, n_epochs=5000)
torch.save(lstm.state_dict(), "lstm_trained.pb")

# Below is code to load in a prexisting LSTM model and train it further:
# lstm = LSTM(input_size, hidden_size, num_layers, x_train.size(1), pred_length)
# lstm.load_state_dict(torch.load("lstm_trained.pb"))
# lstm = train_model(x_train, y_train, input_size, hidden_size, num_layers, pred_length, n_epochs=1000, saved_lstm=lstm)
# lstm.eval()
# torch.save(lstm.state_dict(), "lstm_further_trained.pb")

# finally, obtain predicted time series for test data!
predict(lstm, x_test, y_test, seq_len)


# Shown below is code to perform a crude grid search across different choices of hyperparameters to
# find the optimal hyperparameters for our LSTM model. For each choice of hyperparameters, the routine
# below saves the MSE between the model's prediction and the actual data, and it also saves a plot of the
# predicted data vs. actual data.

# prediction_lengths = [5, 10, 15, 20, 25]
# sequence_lengths = [10, 20, 30, 40, 50]
# hidden_sizes = [8, 16, 32, 64, 128]
#
# trial_mses = 100000. * np.ones((len(prediction_lengths), len(sequence_lengths), len(hidden_sizes)))
#
# print("Tuning Hyperparams...")
# for i in range(len(prediction_lengths)):
#     for j in range(len(sequence_lengths)):
#         for k in range(len(hidden_sizes)):
#             try:
#                 print("Trial Indices: ")
#                 print([i, j, k])
#                 pred_length = prediction_lengths[i]
#                 seq_len = sequence_lengths[j]
#                 hidden_size = hidden_sizes[k]
#
#                 x_train, y_train = preprocess_data(train_data, seq_len, pred_length)
#                 x_test, y_test = preprocess_data(test_data, seq_len, pred_length)
#
#                 input_size = 1
#                 num_layers = 1
#
#
#                 lstm = train_model(x_train, y_train, input_size, hidden_size, num_layers, pred_length, n_epochs=5000)

                  # convention for naming a model based on hyperparameters used is shown below

#                 save_pb = "lstm_" + str(pred_length) + "_" + str(seq_len) + "_" + str(hidden_size) + ".pb"
#                 torch.save(lstm.state_dict(), save_pb)
#
#                 savepath = "lstm_" + str(pred_length) + "_" + str(seq_len) + "_" + str(hidden_size) + ".png"
#                 mse = predict(lstm, x_test, y_test, seq_len, savepath=savepath)
#                 trial_mses[i, j, k] = mse
#                 print("Mean-Squared Error: " + str(mse))

#             except:
#                 print(f"These hyperparams caused a crash. {[i, j, k]}")
