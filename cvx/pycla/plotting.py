from typing import List

from cvx.pycla.helpers import Frontier
from matplotlib import pyplot as plt


def plot_efficient_frontiers(frontiers: List[Frontier]) -> None:
    plt.figure()

    for frontier in frontiers:
        plt.plot(frontier.sigmas, frontier.mus, label=frontier.name)

    plt.title("Efficient Frontier")
    plt.xlabel("sigma")
    plt.ylabel("mu")

    plt.legend()
    plt.show()
