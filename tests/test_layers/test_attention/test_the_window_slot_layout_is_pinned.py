"""Guard: one ``SingleWindowAttention`` instance may not change its slot layout.

Why this test exists
--------------------
``window_slots`` is read as PYTHON state inside ``call()``, so it is baked into a
``tf.function`` trace at trace time, and a graph is not retraced when it changes.
Setting a different slot map at the SAME sequence length therefore silently
returns the FIRST trace's answer. Measured at ``window_size=4``, ``N=6`` before
the guard existed::

    eager  |A - B|      = 1.624966e-02
    traced |A - B|      = 0.0            <- stale
    traced B vs eager A = 0.0            <- returns A's answer
    traced B vs eager B = 1.624966e-02

The layer's own D-015 anchor closes off both obvious repairs: ``window_slots``
cannot go back on the ``call()`` signature (measured to break every functional
Keras consumer), and comparing the map inside ``call()`` cannot work because
``call()`` does not re-run per graph invocation -- not re-running IS the defect.
So the layout is pinned at the setter instead, turning a silent wrong answer into
a named refusal.

Why the length key matters: a different length changes the input shape, which
retraces the graph, so the layout genuinely may differ there.
``partition_mode='band'`` is length-polymorphic by design, and a length-agnostic
guard would reject it wrongly -- ``test_a_different_length_is_still_allowed``
pins that.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.single_window_attention import (
    SingleWindowAttention,
)

DIM = 32
WINDOW = 4
HEADS = 4
N = 6

SLOTS_A = np.array([0, 1, 2, 4, 5, 6], dtype=np.int32)
SLOTS_B = np.array([0, 1, 2, 3, 8, 9], dtype=np.int32)


@pytest.fixture(name="layer")
def _layer():
    keras.utils.set_random_seed(0)
    return SingleWindowAttention(dim=DIM, window_size=WINDOW, num_heads=HEADS)


@pytest.fixture(name="tokens")
def _tokens():
    return np.random.default_rng(0).normal(size=(2, N, DIM)).astype("float32")


def test_a_second_different_layout_at_the_same_length_is_refused(layer, tokens):
    """The regression itself."""
    layer.set_window_slots(SLOTS_A)
    layer(tokens, training=False)
    layer.set_window_slots(None)

    with pytest.raises(ValueError, match="different slot map for length"):
        layer.set_window_slots(SLOTS_B)


def test_the_same_layout_may_be_set_repeatedly(layer, tokens):
    """`WindowAttention._attend` sets then clears the map on EVERY call.

    A guard that rejected the second identical set would break the only
    supported caller on its second invocation.
    """
    for _ in range(3):
        layer.set_window_slots(SLOTS_A)
        layer(tokens, training=False)
        layer.set_window_slots(None)


def test_a_different_length_is_still_allowed(layer, tokens):
    """A different length retraces, so a different layout there is legitimate."""
    layer.set_window_slots(SLOTS_A)
    layer(tokens, training=False)
    layer.set_window_slots(None)

    longer = np.array([0, 1, 2, 3, 4, 5, 8], dtype=np.int32)
    layer.set_window_slots(longer)
    assert layer._window_slots is not None
    assert int(layer._window_slots.shape[0]) == longer.shape[0]


def test_clearing_the_layout_is_always_allowed(layer):
    """`_attend` clears in a `finally`; that path must never raise."""
    layer.set_window_slots(SLOTS_A)
    layer.set_window_slots(None)
    layer.set_window_slots(None)
    assert layer._window_slots is None


def test_two_layouts_on_two_instances_are_independent(tokens):
    """The documented remedy in the error message must actually work."""
    keras.utils.set_random_seed(0)
    first = SingleWindowAttention(dim=DIM, window_size=WINDOW, num_heads=HEADS)
    keras.utils.set_random_seed(0)
    second = SingleWindowAttention(dim=DIM, window_size=WINDOW, num_heads=HEADS)

    first.set_window_slots(SLOTS_A)
    out_a = np.asarray(first(tokens, training=False))
    second.set_window_slots(SLOTS_B)
    out_b = np.asarray(second(tokens, training=False))

    assert np.abs(out_a - out_b).max() > 1e-4, (
        "two different slot layouts produced the same output, so the layout is "
        "no longer reaching the relative-position bias"
    )
