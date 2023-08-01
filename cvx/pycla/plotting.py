from __future__ import annotations

from matplotlib import pyplot as plt

from cvx.pycla.helpers import Frontier


def plot_efficient_frontiers(frontiers: list[Frontier]) -> None:
    plt.figure()

    for frontier in frontiers:
        plt.plot(frontier.sigmas, frontier.mus, label=frontier.name)

    plt.title("Efficient Frontier")
    plt.xlabel("sigma")
    plt.ylabel("mu")

    plt.legend()
    plt.show()
