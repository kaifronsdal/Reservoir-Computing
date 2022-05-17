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

def get_data(axon_num, train_time=1300):
    data = create_data(50, 0.05, axon_num, 40 * train_time)
    train_data = data[:train_time]
    test_data = data[:40 * train_time]
    return train_data, test_data


train_data, test_data = get_data(1)


# NOTE: must reshape data to be compatible with LSTM
def preprocess_data(data):
    x = Variable(torch.Tensor(data[:-1].reshape(-1, 1)))
    x = torch.reshape(x, (x.size(0), 1, x.size(1)))

    y = Variable(torch.Tensor(data[1:].reshape(-1, 1)))

    return x, y

x_train, y_train = preprocess_data(train_data)
print(y_train.size())
x_test, y_test = preprocess_data(test_data)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.linear(out)

        return out

def train_model(x_train, y_train, input_size, hidden_size, num_layers, n_epochs=1001, lr=1e-3):
    sequence_length = x_train.size(1)
    lstm = LSTM(input_size, hidden_size, num_layers, sequence_length)

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

input_size = 1
hidden_size = 16
num_layers = 1
lstm = train_model(x_train, y_train, input_size, hidden_size, num_layers)

def predict(x_test, y_test):
    y_preds = (lstm.forward(x_test)).detach().numpy()
    y_preds = y_preds.reshape(-1)

    y_test_np = y_test.detach().numpy()
    y_test_np = y_test_np.reshape(-1)

    plt.figure(figsize=(10, 3))
    plt.title("SQUID Prediction (Testing) with LSTM")
    plt.xlabel("$t$")
    plt.plot(y_test_np, label="squid data", color="blue")
    plt.plot(y_preds, label="prediction", color="red")
    plt.legend()
    # plt.show()
    plt.savefig("lstm_squid_preds.png")

predict(x_test, y_test)

# The caveat: need to train LSTM for more than 1000 timesteps, else it doesn't fully learn the spiking behavior!