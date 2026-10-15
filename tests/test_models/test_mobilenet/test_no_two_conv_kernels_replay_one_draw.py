"""No two kernels of a freshly built MobileNet replay one initializer draw.

A seedless ``keras.initializers.Initializer`` INSTANCE handed to several
weights replays one sample at every site, and a shorter draw comes out as the
longer draw's prefix (scaled by the fan factor), so two kernels can carry the
same random numbers without sharing a shape (``initializers/clone.py``).

MEASURED at fb905f29b with the instrument below (seeded builds at 32 x 32,
hybrid V4 at 128 x 128; the counts are D-016's):

====================  ==================================
variant               replayed pairs / pairs compared
====================  ==================================
V1 (every width)      351 / 378
V2 (every width)      630 / 1431
V3 large              1128 / 2080
V3 small              861 / 1485
V4 small              15 / 861
V4 medium             27 / 2016
V4 large              40 / 3570
V4 hybrid_medium      27 / 2701
V4 hybrid_large       40 / 4465
====================  ==================================

V1-V3 handed the model's resolved instance to the stem, every block and the
classifier; inside ``DepthwiseSeparableBlock`` and
``UniversalInvertedBottleneck`` one instance reached every conv. V4 passes a
string, so only the in-block sharing tied it. After D-076 every count is 0.

The instrument below compares every pair, not only equal sizes: the leading
``_PREFIX`` values of each flattened kernel, so a prefix replay across
different shapes is caught too. Its positive control is
``test_the_detector_sees_a_planted_shared_instance``.
"""

import itertools
from typing import List, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.mobilenet import (
    MobileNetV1,
    MobileNetV2,
    MobileNetV3,
    MobileNetV4,
)

#: Leading values compared per flattened kernel. Large enough that two
#: independent draws never correlate above the threshold by chance (the null
#: standard deviation of r at n = 256 is 1 / sqrt(255) = 0.063), small enough
#: that the all-pairs sweep stays cheap.
_PREFIX = 256

#: Shortest kernel compared. A 3 x 3 depthwise kernel of one channel has 9
#: values; below 16 the correlation is too noisy to read.
_MIN_SIZE = 16

#: A replayed draw correlates at 1 up to float rounding (scaled He/Glorot
#: draws: r = 0.99999999999); independent draws sit near 0.
_REPLAY_R = 0.9999

_VARIANTS = [
    (cls, variant)
    for cls in (MobileNetV1, MobileNetV2, MobileNetV3, MobileNetV4)
    for variant in cls.MODEL_VARIANTS
]


def _replayed_pairs(model: keras.Model) -> Tuple[List[str], int]:
    """Return ``(replayed pairs, pairs compared)`` over the built weights.

    :param model: A built model.
    :return: ``"path_a ~ path_b (r=...)"`` for every pair of trainable
        ``ndim >= 2`` weights with at least ``_MIN_SIZE`` values whose leading
        ``min(_PREFIX, size_a, size_b)`` flattened values correlate above
        ``_REPLAY_R``, and the number of pairs compared.
    """
    kernels = []
    for weight in model.weights:
        if not weight.trainable:
            continue
        values = np.asarray(weight, dtype=np.float64)
        if values.ndim < 2 or values.size < _MIN_SIZE or np.ptp(values) == 0.0:
            continue
        kernels.append((weight.path, values.ravel()))
    replayed: List[str] = []
    compared = 0
    for (path_a, a), (path_b, b) in itertools.combinations(kernels, 2):
        n = min(_PREFIX, a.size, b.size)
        x, y = a[:n], b[:n]
        if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
            continue
        compared += 1
        r = float(np.corrcoef(x, y)[0, 1])
        if abs(r) > _REPLAY_R:
            replayed.append(f"{path_a} ~ {path_b} (r={r:.6f})")
    return replayed, compared


def _build(cls, variant: str) -> keras.Model:
    # The hybrid V4 variants run Mobile MQA, whose key/value path downsamples
    # once more; at 32 x 32 its softmax runs over a single key, which Keras
    # refuses under the suite's error::UserWarning. 128 leaves 2 x 2 keys.
    side = 128 if variant.startswith("hybrid") else 32
    keras.utils.set_random_seed(1234)
    model = cls.from_variant(variant, num_classes=10, input_shape=(side, side, 3))
    model(np.zeros((2, side, side, 3), dtype="float32"))
    return model


def test_the_detector_sees_a_planted_shared_instance():
    """Positive control: one shared seedless instance across two shapes."""
    keras.utils.set_random_seed(1234)
    shared = keras.initializers.get("he_normal")
    inputs = keras.Input((8, 8, 4))
    x = keras.layers.Conv2D(16, 3, kernel_initializer=shared)(inputs)
    x = keras.layers.Conv2D(32, 1, kernel_initializer=shared)(x)
    planted = keras.Model(inputs, x)
    replayed, compared = _replayed_pairs(planted)
    assert compared == 1
    assert len(replayed) == 1, replayed


@pytest.mark.parametrize(
    "cls,variant", _VARIANTS, ids=[f"{c.__name__}-{v}" for c, v in _VARIANTS]
)
def test_no_two_kernels_replay_one_draw(cls, variant):
    model = _build(cls, variant)
    replayed, compared = _replayed_pairs(model)
    # Anti-vacuity: a model exposing no comparable pair proves nothing.
    assert compared >= 100, f"only {compared} kernel pairs compared"
    assert not replayed, (
        f"{cls.__name__}({variant!r}): {len(replayed)} of {compared} kernel "
        "pairs replay one initializer draw. A seedless initializer INSTANCE "
        "reaches several convs; wrap each site in "
        "dl_techniques.initializers.clone.clone_initializer (D-076). "
        f"Pairs: {replayed[:10]}"
    )
