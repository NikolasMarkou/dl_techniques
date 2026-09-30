"""
Regression data for the HKAN trainer
====================================================================

Two sources, both returning the same dict of float64 arrays
(``x_train``, ``y_train``, ``x_test``, ``y_test``):

- :func:`make_dataset`: the five synthetic target functions of the HKAN paper
  (Dudek and Rodak, arXiv:2501.18199, Section VI-A), generated locally from a seed.
- :func:`load_csv_dataset`: two user-supplied comma-separated files whose last
  column is the target. Nothing is downloaded and no data file ships with the repo.

Readings of the paper that this module takes a side on. The paper does not settle
the first two; the evidence on both sides is listed so the choice can be revisited.

- **Scaling: every target and every input is min-max scaled to [0, 1].** For
  scaling TF1 and TF2 too: the paper's Fig. 3 draws TF1 and TF2 on a vertical axis
  from 0 to 1 (unscaled, TF1 spans [-1, 1] and TF2 about [-1.6, 1.8]), and its
  notation table gives the target as ``y in [0, 1]``. Against: the one sentence on
  the subject names only three targets ("the function values and input arguments
  of TF3, TF4, and TF5 were normalized to the range [0, 1]"). The reported TF2
  train RMSE (0.116, the standard deviation of the noise, 0.4 / sqrt(12) = 0.1155)
  cannot tell the two readings apart: noise added after scaling gives the same
  floor as noise added to an unscaled target. The statistics come from the TRAIN
  split only, so a test value outside the train range maps outside ``[0, 1]``.
- **TF2 noise: added after scaling, to the train targets only.** "The TF2 training
  data was perturbed by adding noise generated from U(-0.2, 0.2)". Here the noise
  is added to the SCALED train targets, so it is relative to a unit range (about
  3.4 times larger against the signal than the same noise on the unscaled target,
  whose range is about 3.4). The scaling statistics of the TF2 target are those of
  the NOISELESS train targets; a noisy train target can therefore leave
  ``[0, 1]`` by up to 0.2. The test targets are the scaled noiseless function.
- **TF4 sign: plus.** The paper prints ``1 - cos(2 pi r) - 0.1 r`` with
  ``r = sqrt(sum x_i^2)``. This module uses ``+ 0.1 r``, the standard Salomon
  function, and treats the printed minus as a typo.
- **TF2 formula.** The PDF line is garbled; ``sum_i sin(20 exp(x_i)) x_i^2`` is this
  module's reading of it (the exponent 2 sits on the neighbouring line).
"""

from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------


def _tf1(x: np.ndarray) -> np.ndarray:
    return (2.0 * x[:, 0] - 1.0) * (2.0 * x[:, 1] - 1.0)


def _tf2(x: np.ndarray) -> np.ndarray:
    return np.sum(np.sin(20.0 * np.exp(x)) * x ** 2, axis=1)


def _tf3(x: np.ndarray) -> np.ndarray:
    return -np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


def _tf4(x: np.ndarray) -> np.ndarray:
    # DECISION plan-2026-09-30T082355-4d999dbc/D-041
    # Keep `+ 0.1 * radius` (the standard Salomon function). Do NOT change it to
    # the minus the paper prints "to match the paper": the printed sign is read
    # as a typo, the reading is declared in the module docstring, and the tests
    # pin the plus at hand-computed points. See decisions.md D-041.
    radius = np.sqrt(np.sum(x ** 2, axis=1))
    return 1.0 - np.cos(2.0 * np.pi * radius) + 0.1 * radius


def _tf5(x: np.ndarray) -> np.ndarray:
    index = np.arange(1, x.shape[1] + 1, dtype=np.float64)
    return -np.sum(np.sin(x) * np.sin(index * x ** 2 / np.pi) ** 20, axis=1)


class TargetSpec(NamedTuple):
    """One synthetic target: function, input count, input range, sizes, train noise."""

    function: Callable[[np.ndarray], np.ndarray]
    num_inputs: int
    low: float
    high: float
    num_train: int
    num_test: int
    train_noise: float


#: Sample counts and ranges from the paper's Table VI and Section VI-A.
TARGETS: Dict[str, TargetSpec] = {
    "tf1": TargetSpec(_tf1, 2, 0.0, 1.0, 5000, 10000, 0.0),
    "tf2": TargetSpec(_tf2, 2, 0.0, 1.0, 5000, 10000, 0.2),
    "tf3": TargetSpec(_tf3, 2, -500.0, 500.0, 5000, 10000, 0.0),
    "tf4": TargetSpec(_tf4, 10, -4.0, 4.0, 3750, 1250, 0.0),
    "tf5": TargetSpec(_tf5, 2, 0.0, np.pi, 5000, 10000, 0.0),
    "tf5_5": TargetSpec(_tf5, 5, 0.0, np.pi, 7500, 2500, 0.0),
}

# ---------------------------------------------------------------------


def minmax_scale(
        train: np.ndarray, *others: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """Min-max scale ``train`` to ``[0, 1]`` and apply the SAME map to ``others``.

    The statistics are the per-column minimum and range of ``train`` only. A
    constant column (range 0) is mapped to 0 rather than divided by zero.

    :param train: Array the statistics are taken from, shape ``(N,)`` or ``(N, d)``.
    :type train: np.ndarray
    :param others: Arrays scaled with the train statistics.
    :type others: np.ndarray
    :return: The scaled ``train`` followed by the scaled ``others``.
    :rtype: Tuple[np.ndarray, ...]
    """
    low = train.min(axis=0)
    span = train.max(axis=0) - low
    span = np.where(span > 0, span, 1.0)
    return tuple((array - low) / span for array in (train,) + others)


def make_dataset(
        name: str,
        seed: Optional[int] = None,
        num_train: Optional[int] = None,
        num_test: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Generate one of the paper's synthetic regression problems.

    Inputs are uniform on the target's range. Inputs and target are min-max scaled
    with the train statistics; the TF2 noise is added to the scaled train targets.
    See the module docstring for the readings of the paper behind both.

    :param name: A key of :data:`TARGETS`.
    :type name: str
    :param seed: Seed of the generator; ``0`` is a seed, ``None`` is unseeded.
    :type seed: Optional[int]
    :param num_train: Train rows; defaults to the paper's count.
    :type num_train: Optional[int]
    :param num_test: Test rows; defaults to the paper's count.
    :type num_test: Optional[int]
    :return: ``x_train (N, d)``, ``y_train (N,)``, ``x_test (M, d)``, ``y_test (M,)``,
        all float64.
    :rtype: Dict[str, np.ndarray]
    :raises ValueError: If ``name`` is unknown or a row count is below 2.
    """
    if name not in TARGETS:
        raise ValueError(f"unknown dataset {name!r}; choose one of {sorted(TARGETS)}")
    spec = TARGETS[name]
    num_train = spec.num_train if num_train is None else num_train
    num_test = spec.num_test if num_test is None else num_test
    if num_train < 2 or num_test < 2:
        raise ValueError(
            f"num_train and num_test must be >= 2, got {num_train} and {num_test}"
        )
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(spec.low, spec.high, (num_train, spec.num_inputs))
    x_test = rng.uniform(spec.low, spec.high, (num_test, spec.num_inputs))
    y_train, y_test = spec.function(x_train), spec.function(x_test)
    # DECISION plan-2026-09-30T082355-4d999dbc/D-041
    # EVERY target is scaled, and the TF2 noise is added AFTER the scaling. Do NOT
    # bring back a per-target switch that leaves TF1 and TF2 unscaled, and do NOT
    # add the noise before `minmax_scale`: the scaling statistics would then be
    # those of the noisy targets and the noise would shrink with the target's
    # range (about 3.4 for TF2) instead of being U(-0.2, 0.2) on a unit range.
    # The argument once made for the unscaled reading (the paper's TF2 RMSE) does
    # not separate the two. See decisions.md D-041, which supersedes D-028 item 3.
    x_train, x_test = minmax_scale(x_train, x_test)
    y_train, y_test = minmax_scale(y_train, y_test)
    if spec.train_noise > 0:
        y_train = y_train + rng.uniform(-spec.train_noise, spec.train_noise, num_train)
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


def _read_csv(path: Union[str, Path]) -> np.ndarray:
    """Read one numeric comma-separated file as a float64 ``(N, d + 1)`` array."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"CSV file not found: {path}")
    try:
        table = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    except ValueError as error:
        raise ValueError(f"{path} is not a numeric comma-separated file: {error}") from error
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError(
            f"{path} must hold at least 2 rows and 2 columns (inputs then the "
            f"target), got shape {table.shape}"
        )
    if not np.all(np.isfinite(table)):
        raise ValueError(f"{path} holds a non-finite value")
    return table


def load_csv_dataset(
        train_csv: Union[str, Path],
        test_csv: Union[str, Path],
        scale: bool = False,
) -> Dict[str, np.ndarray]:
    """Load a regression problem from two comma-separated files.

    Every row is one sample; the last column is the target, the others are the
    inputs. There is no header row. The files are used as they are unless ``scale``
    is set, in which case inputs and target are min-max scaled with the TRAIN
    file's statistics.

    :param train_csv: Path of the train file.
    :type train_csv: Union[str, Path]
    :param test_csv: Path of the test file.
    :type test_csv: Union[str, Path]
    :param scale: Min-max scale with train statistics.
    :type scale: bool
    :return: Same dict as :func:`make_dataset`.
    :rtype: Dict[str, np.ndarray]
    :raises ValueError: If a file is missing, not numeric, too small, non-finite,
        or the two files have different column counts.
    """
    train, test = _read_csv(train_csv), _read_csv(test_csv)
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            f"train has {train.shape[1]} columns but test has {test.shape[1]}"
        )
    x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
    if scale:
        x_train, x_test = minmax_scale(x_train, x_test)
        y_train, y_test = minmax_scale(y_train, y_test)
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

# ---------------------------------------------------------------------
