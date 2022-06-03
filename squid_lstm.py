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
    raw_data = np.loadtxt(f"raw_data/axon_{axon_num}_t01_data.csv", delimiter=",")
    vmembrane_data = raw_data[:, 0]
    impulse_data = raw_data[:, 1]
    moving_average = np.convolve(impulse_data, np.ones(window), 'valid') / window
    pulse_index = list(map(lambda avg: avg > min_buffer, moving_average)).index(1)

    return vmembrane_data[pulse_index:pulse_index + num_points]


def get_data(axon_num, train_time=10000, temporal_resolution=50):
    data = create_data(50, 0.05, axon_num, 2 * train_time)
    train_data = data[:train_time:temporal_resolution]
    test_data = data[:2 * train_time:temporal_resolution]
    return train_data, test_data


def getSequentialData(data, sequence_length, pred_length=5):
    sequential_data = np.empty((0, sequence_length))
    next_value = []
    prediction_length = pred_length
    for i in range(len(data) - sequence_length - 1 - prediction_length):
        sequential_data = np.vstack((sequential_data, data[i:i + sequence_length].reshape(1, -1)))
        next_value.append(data[i + sequence_length:i + sequence_length + prediction_length])
    return sequential_data, np.array(next_value)


# NOTE: must reshape data to be compatible with LSTM
def preprocess_data(data, sequence_length, pred_length=5):
    sequential_x, next_y = getSequentialData(data, sequence_length, pred_length)

    x = Variable(torch.Tensor(sequential_x))
    x = torch.reshape(x, (x.size(0), x.size(1), 1))
    # x = Variable(torch.Tensor(data[:-1].reshape(-1,1)))
    # x = torch.reshape(x, (x.size(0), 1, x.size(1)))

    y = Variable(torch.Tensor(next_y))
    y = torch.reshape(y, (y.size(0), y.size(1)))

    return x, y


# Maybe need to have larger sequence length... i.e. have sliding window...

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward1(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # out, _ = self.lstm(x, (h0, c0))
        # out = out[:, -1, :]
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        h_n = h_n.view(-1, self.hidden_size)
        out = self.linear(h_n)

        return out

    def forward(self, x):
        out = self.forward1(x)
        out_3D = torch.reshape(out, (out.size(0), out.size(1), 1))
        x = torch.cat((x, out_3D), 1)

        for i in range(pred_length - 1):
            next_out = self.forward1(x[:, i + 1:])
            out_3D = torch.reshape(next_out, (next_out.size(0), next_out.size(1), 1))
            x = torch.cat((x, out_3D), 1)
            out = torch.cat((out, next_out), 1)
        return out


def train_model(x_train, y_train, input_size, hidden_size, num_layers, n_epochs=10001, lr=1e-3, saved_lstm=None):
    sequence_length = x_train.size(1)

    if saved_lstm is None:
        lstm = LSTM(input_size, hidden_size, num_layers, sequence_length)
    else:
        lstm = saved_lstm

    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr=lr)

    for epoch in range(n_epochs):
        y_out = lstm.forward(x_train)
        optimizer.zero_grad()

        loss = criterion(y_out, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch: {epoch} | loss: {round(loss.item(), 4)}")

    return lstm


def predict(x_test, y_test, savepath=None):
    y_test_np = y_test.detach().numpy()[:, 0]
    y_test_np = y_test_np.reshape(-1)

    y_preds = np.zeros_like(y_test_np)
    x_so_far = x_test[0, :, :].detach().numpy()
    x_so_far = x_so_far.reshape(-1)

    x_curr = x_test[0, :, :]
    x_curr = Variable(torch.reshape(x_curr, (1, seq_len, x_test.size(2))))
    # x_curr = Variable(x_curr.reshape(1, 1, x_test.size(2)))
    for i in range(len(y_preds)):
        next_pred_tensor = lstm.forward1(x_curr)
        next_pred = next_pred_tensor.detach().numpy()
        next_pred = next_pred.reshape(-1)[0]
        y_preds[i] = next_pred

        x_so_far = np.append(x_so_far, next_pred)
        x_curr = Variable(torch.tensor(x_so_far[-seq_len:]))
        x_curr = torch.reshape(x_curr, (1, seq_len, x_test.size(2)))

    # y_preds = (lstm.forward(x_test)).detach().numpy()
    # y_preds = y_preds.reshape(-1)

    plt.figure(figsize=(10, 3))
    plt.title("SQUID Prediction (Testing) with LSTM")
    plt.xlabel("$t$")
    plt.plot(y_test_np, label="squid data", color="blue")
    plt.plot(y_preds, label="prediction", color="red")
    plt.legend()
    # plt.show()
    if savepath is None:
        savepath = "lstm_squid_preds_50_25.png"
    plt.savefig(savepath)

    mse = (1.0 / len(y_preds)) * np.sum((y_preds - y_test_np) ** 2)
    return mse


train_data, test_data = get_data(1)
print(train_data.shape)
print(test_data.shape)

prediction_lengths = [5, 10, 15, 20, 25]
sequence_lengths = [10, 20, 30, 40, 50]
hidden_sizes = [8, 16, 32, 64, 128]

trial_mses = 100000. * np.ones((len(prediction_lengths), len(sequence_lengths), len(hidden_sizes)))

# pred_length = 25
# seq_len = 50
print("Tuning Hyperparams...")
temp = 0
for i in range(len(prediction_lengths)):
    for j in range(len(sequence_lengths)):
        for k in range(len(hidden_sizes)):
            try:
                if temp > 17:
                    print("Trial Indices: ")
                    print([i, j, k])
                    pred_length = prediction_lengths[i]
                    seq_len = sequence_lengths[j]
                    hidden_size = hidden_sizes[k]

                    x_train, y_train = preprocess_data(train_data, seq_len)
                    x_test, y_test = preprocess_data(test_data, seq_len)

                    input_size = 1
                    # hidden_size = 16
                    num_layers = 1

                    # model_names = ["lstm_thurs_1001.pb", "lstm_mon_2.pb", "lstm_mon_3.pb", "lstm_mon_4.pb", "lstm_mon_5.pb", "lstm_mon_6.pb", "lstm_mon_7.pb", "lstm_mon_8.pb", "lstm_mon_9.pb", "lstm_mon_10.pb"]
                    # savepath_names = ["lstm_squid_preds_100.png", "lstm_squid_preds_200.png", "lstm_squid_preds_300.png", "lstm_squid_preds_400.png", "lstm_squid_preds_500.png", "lstm_squid_preds_600.png", "lstm_squid_preds_700.png", "lstm_squid_preds_800.png", "lstm_squid_preds_900.png", "lstm_squid_preds_1000.png"]
                    #
                    lstm = train_model(x_train, y_train, input_size, hidden_size, num_layers, n_epochs=5000)
                    save_pb = "lstm_" + str(pred_length) + "_" + str(seq_len) + "_" + str(hidden_size) + ".pb"
                    torch.save(lstm.state_dict(), save_pb)

                    savepath = "lstm_" + str(pred_length) + "_" + str(seq_len) + "_" + str(hidden_size) + ".png"
                    mse = predict(x_test[10000 // 50:], y_test[10000 // 50:], savepath=savepath)
                    trial_mses[i, j, k] = mse
                    print("Mean-Squared Error: " + str(mse))
                else:
                    temp += 1
            except:
                print(f"These hyperparams caused a crash. {[i, j, k]}")

min_loss = trial_mses[0, 0, 0]
optimal_hyperparams_index = [0, 0, 0]
for i in range(len(prediction_lengths)):
    for j in range(len(sequence_lengths)):
        for k in range(len(hidden_sizes)):
            mse = trial_mses[i, j, k]
            if mse < min_loss:
                min_loss = mse
                optimal_hyperparams_index = [i, j, k]

print("Optimal Hyperparameter Indices: ")
print(optimal_hyperparams_index)

# lstm = train_model(x_train, y_train, input_size, hidden_size, num_layers, n_epochs=2001)
# torch.save(lstm.state_dict(), "lstm_50_25.pb")
# lstm = LSTM(input_size, hidden_size, num_layers, x_train.size(1))
# lstm.load_state_dict(torch.load("lstm_50_25.pb"))
# lstm = train_model(x_train, y_train, input_size, hidden_size, num_layers, n_epochs=8001, saved_lstm=lstm)
# lstm.eval()
# torch.save(lstm.state_dict(), "lstm_50_25.pb")
#
# #
# # print(y_test.shape)
# # print(y_train.shape)
# predict(x_test[10000//50:], y_test[10000//50:])
