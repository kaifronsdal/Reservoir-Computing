import numpy as np
import matplotlib.pyplot as plt
from reservior import Reservoir
from sklearn.linear_model import LinearRegression
from matrix_generation import _random_sparse


def main():
    # print(_random_sparse(3, 3))
    r = Reservoir(1, 50)
    # X = r.run(np.arange(0, 10, 1))
    X = np.sin(np.linspace(0, 2 * np.pi, 100)).reshape(-1, 1)
    states = r.train(X)
    model = LinearRegression()
    model.fit(states, X)
    X = np.sin(np.linspace(0, 4 * np.pi, 100)).reshape(-1, 1)
    states = r.train(X)
    # pred = r.run(100, model)
    pred = model.predict(states)

    plt.figure(figsize=(10, 3))
    plt.title("A sine wave")
    plt.xlabel("$t$")
    plt.plot(X, label="sin(t)", color="blue")
    plt.plot(pred, label="prediction", color="red")
    plt.legend()
    plt.show()
    # print(X)


if __name__ == "__main__":
    main()
