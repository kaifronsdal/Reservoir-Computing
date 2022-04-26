import numpy as np
from reservior import Reservoir
from matrix_generation import _random_sparse


def main():
    # print(_random_sparse(3, 3))
    r = Reservoir(1, 20)
    X = r.run(np.arange(0, 10, 1))
    print(X)


if __name__ == "__main__":
    main()
