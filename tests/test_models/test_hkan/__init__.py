"""Shared data, shapes and closed forms of the HKAN model tests.

Guards nothing by itself. It holds the three things every module of this
directory needs, so that each exists once:

* the fixture shapes, pairwise different (``N_IN``, ``HIDDEN``, the two
  ``NUM_BASIS`` values) so that a swapped axis changes a shape or a number;
* one seeded regression problem (:func:`make_data`);
* the basis functions as scipy/numpy closed forms (:data:`CLOSED_FORM`) and the
  block feature formula of the layer's documentation written from that text
  (:func:`block_features`). Neither imports anything from the package under
  test: they are what its two basis tables and its feature tensor are compared
  against.
"""

from typing import Callable, Dict, Tuple

import numpy as np
from scipy.special import expit

N_ROWS: int = 96
N_IN: int = 3
HIDDEN: int = 5
NUM_BASIS: Tuple[int, int] = (7, 4)
SLOPE: float = 5.0

#: Basis functions ``g`` by name, from their textbook definitions.
CLOSED_FORM: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigmoid": expit,
    "gaussian": lambda d: np.exp(-d ** 2),
    "relu": lambda d: np.maximum(d, 0.0),
    "tanh": np.tanh,
    "softplus": lambda d: np.logaddexp(0.0, d),
    "identity": lambda d: d,
}


def make_data(
        seed: int = 11, n_rows: int = N_ROWS, n_in: int = N_IN, offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a seeded float64 regression problem.

    :param seed: Seed of the draw.
    :param n_rows: Number of rows ``N``.
    :param n_in: Number of input columns, at least 3.
    :param offset: Constant added to the target.
    :return: ``(x, y)`` with ``x`` uniform on ``[0, 1]``, shape ``(N, n_in)``,
        and ``y`` of shape ``(N,)``, a smooth function of the first three
        columns plus noise of standard deviation 0.05.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(n_rows, n_in))
    y = (
        np.sin(3.0 * x[:, 0]) + x[:, 1] ** 2 - 0.5 * x[:, 2]
        + 0.05 * rng.normal(size=n_rows) + offset
    )
    return x, y


def block_features(
        basis: str, slope: float, column: np.ndarray, centers: np.ndarray,
) -> np.ndarray:
    """Basis features of one block: ``g(slope * (x_p - center_r))``.

    ``identity`` is ``g(d) = d`` on the unscaled difference (the layer's
    documented rule: the slope belongs to the non-linear basis functions).

    :param basis: Basis name.
    :param slope: The layer's slope.
    :param column: One input column ``x_p``, shape ``(N,)``.
    :param centers: The block's centers, shape ``(m,)``.
    :return: Features of shape ``(N, m)``, float64.
    """
    column = np.asarray(column, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    effective = 1.0 if basis == "identity" else slope
    return CLOSED_FORM[basis](effective * (column[:, None] - centers[None, :]))
