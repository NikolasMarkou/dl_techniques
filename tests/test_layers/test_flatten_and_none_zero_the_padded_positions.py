"""With a mask, ``flatten`` and ``none`` return every padded position as zeros.

Guard for plan-2026-10-14T062616-81938e81 step 5.4 (D-123 item c, D-203; GUIDE
10.1 "padding reached pooled outputs"). Measured at 69b68fe5a:
``SequencePooling('flatten')(x, mask=m)`` equalled ``flatten(x)``, so the padded
tokens' content was baked into the vector, and ``none`` returned ``x`` as it
was. ``exclude_positions`` still does not apply to these two (D-005).
"""

import numpy as np
import pytest

from dl_techniques.layers.sequence_pooling import SequencePooling

B, S, D = 2, 5, 3
MASK = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], "float32")


def _x(seed=0):
    return np.random.default_rng(seed).normal(size=(B, S, D)).astype("float32")


@pytest.mark.parametrize("strategy", ["none", "flatten"])
def test_padded_positions_come_back_as_zeros(strategy):
    x = _x()
    out = np.asarray(SequencePooling(strategy=strategy)(x, mask=MASK)).reshape(B, S, D)
    np.testing.assert_array_equal(out, x * MASK[..., None])


@pytest.mark.parametrize("strategy", ["none", "flatten"])
def test_padding_content_never_reaches_the_output(strategy):
    x, flooded = _x(), _x()
    flooded[0, 3:] = np.nan
    flooded[0, 3, 0] = np.inf
    layer = SequencePooling(strategy=strategy)
    a = np.asarray(layer(x, mask=MASK))
    b = np.asarray(layer(flooded, mask=MASK))
    np.testing.assert_array_equal(a[0], b[0])


@pytest.mark.parametrize("strategy", ["none", "flatten"])
def test_without_a_mask_the_sequence_is_unchanged(strategy):
    x = _x()
    out = np.asarray(SequencePooling(strategy=strategy)(x)).reshape(B, S, D)
    np.testing.assert_array_equal(out, x)


@pytest.mark.parametrize("strategy", ["none", "flatten"])
def test_exclude_positions_still_does_not_apply(strategy):
    x = _x()
    out = np.asarray(
        SequencePooling(strategy=strategy, exclude_positions=[0])(x)
    ).reshape(B, S, D)
    np.testing.assert_array_equal(out, x)
