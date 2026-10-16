"""Instrument: which kernels of a built model replay one initializer draw.

Not a test module (no ``test_`` prefix, so pytest does not collect it). Its
consumers are ``test_mobilenet/test_no_two_conv_kernels_replay_one_draw.py``
(D-076) and ``test_convnext/test_no_two_conv_kernels_replay_one_draw.py``
(D-091); its positive control is in the first.

A seedless ``keras.initializers.Initializer`` INSTANCE handed to several
weights replays one sample at every site, and a shorter draw comes out as the
longer draw's prefix (scaled by the fan factor), so two kernels can carry the
same random numbers without sharing a shape (``initializers/clone.py``). The
instrument compares every pair, not only equal sizes: the leading ``PREFIX``
values of each flattened kernel, so a prefix replay across shapes is caught.
"""

import itertools
from typing import List, Tuple

import keras
import numpy as np

#: Leading values compared per flattened kernel. Large enough that two
#: independent draws never correlate above the threshold by chance (the null
#: standard deviation of r at n = 256 is 1 / sqrt(255) = 0.063), small enough
#: that the all-pairs sweep stays cheap.
PREFIX = 256

#: Shortest kernel compared. A 3 x 3 depthwise kernel of one channel has 9
#: values; below 16 the correlation is too noisy to read.
MIN_SIZE = 16

#: A replayed draw correlates at 1 up to float rounding (scaled He/Glorot
#: draws: r = 0.99999999999); independent draws sit near 0.
REPLAY_R = 0.9999


def replayed_pairs(model: keras.Model) -> Tuple[List[str], int]:
    """Return ``(replayed pairs, pairs compared)`` over the built weights.

    Interface contract (callers: the two guards named in the module
    docstring): ``model`` must be built. Every pair of trainable ``ndim >= 2``
    weights with at least ``MIN_SIZE`` values and a non-constant prefix is
    compared on its leading ``min(PREFIX, size_a, size_b)`` flattened values;
    a pair whose Pearson ``|r|`` exceeds ``REPLAY_R`` is reported as
    ``"path_a ~ path_b (r=...)"``. Raises nothing.

    :param model: A built model.
    :return: The replayed pairs and the number of pairs compared.
    """
    kernels = []
    for weight in model.weights:
        if not weight.trainable:
            continue
        values = np.asarray(weight, dtype=np.float64)
        if values.ndim < 2 or values.size < MIN_SIZE or np.ptp(values) == 0.0:
            continue
        kernels.append((weight.path, values.ravel()))
    replayed: List[str] = []
    compared = 0
    for (path_a, a), (path_b, b) in itertools.combinations(kernels, 2):
        n = min(PREFIX, a.size, b.size)
        x, y = a[:n], b[:n]
        if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
            continue
        compared += 1
        r = float(np.corrcoef(x, y)[0, 1])
        if abs(r) > REPLAY_R:
            replayed.append(f"{path_a} ~ {path_b} (r={r:.6f})")
    return replayed, compared
