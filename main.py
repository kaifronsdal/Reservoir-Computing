import numpy as np
import matplotlib.pyplot as plt
from reservior import Reservoir
from sklearn.linear_model import LinearRegression, Ridge
from matrix_generation import _random_sparse


def main():
    # print(_random_sparse(3, 3))
    r = Reservoir(1, 20)
    # X = r.run(np.arange(0, 10, 1))
    X = np.sin(np.linspace(0, 100 * np.pi, 40000)).reshape(-1, 1)+np.sin(np.linspace(0, 2320 * np.pi, 40000)).reshape(-1, 1)
    states = r.train(X)
    model = Ridge(alpha=0.0000001)
    model.fit(states, X)
    X = np.sin(np.linspace(0, 400 * np.pi, 4*40000)).reshape(-1, 1)+np.sin(np.linspace(0, 4*2320 * np.pi, 4*40000)).reshape(-1, 1)
    states = r.train(X)
    # pred = r.run(100, model)
    pred = model.predict(states)

    plt.figure(figsize=(10, 3))
    plt.title("A sine wave")
    plt.xlabel("$t$")
    plt.plot(X[80000:82800], label="sin(t)", color="blue")
    plt.plot(pred[80000:82800], label="prediction", color="red")
    plt.legend()
    plt.show()
    # print(X)


if __name__ == "__main__":
    main()
