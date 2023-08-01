from __future__ import annotations

from unittest.mock import patch

from cvx.pycla.helpers import Frontier
from cvx.pycla.plotting import plot_efficient_frontiers


def test_plotting() -> None:
    with patch("cvx.pycla.plotting.plt.show") as show_patch:
        plot_efficient_frontiers([Frontier([5, 4, 3], [4, 3, 2], "test_frontier")])
        assert show_patch.called
