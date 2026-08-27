"""Guard: a ``shift_size > 0`` layer must work inside a subclassed ``keras.Model``.

Why this test exists
--------------------
``_compute_attention_mask`` used to return a bare ``keras.Variable``. A bare
``Variable`` is not attributed to the layer that created it, so Keras charged it
to whichever layer was on the build stack. Inside a subclassed ``keras.Model``
that is the Model -- already built by then -- and every ``shift_size > 0`` layer
died on its FIRST call with::

    ValueError: You cannot add new elements of state (variables or sub-layers)
    to a layer that is already built.

``fit()`` and ``predict()`` failed the same way.

Why the existing suite could not see it: every SW-MSA test in
``test_progressive_focused_attention.py`` calls the bare layer directly, and the
one test that wraps the layer in a Model uses ``shift_size=0``, which takes the
``self._attn_mask = None`` branch and never builds a mask at all.

Why this can fail if the implementation is wrong: reverting the mask to a
``keras.Variable`` (or to ``self.add_weight``) makes every case below raise, and
materialising it as a tensor inside ``build()`` instead raises the companion
``cannot be accessed from here ... FuncGraph(name=scratch_graph) ... out of
scope`` error. Both were observed while writing this guard.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.progressive_focused_attention import (
    ProgressiveFocusedAttention,
)

DIM = 32
HEADS = 4
WINDOW = 4
SHIFT = 2
H = W = 8


def _unwrap(out):
    """The layer returns ``(output, attn_map)``; models here need the output."""
    return out[0] if isinstance(out, (tuple, list)) else out


class SwmsaWrapper(keras.Model):
    def __init__(self, shift_size: int, **kwargs):
        super().__init__(**kwargs)
        self.att = ProgressiveFocusedAttention(
            dim=DIM,
            num_heads=HEADS,
            window_size=WINDOW,
            shift_size=shift_size,
        )

    def call(self, x, training=None):
        return _unwrap(self.att(x, training=training))


@pytest.fixture(name="batch")
def _batch():
    return np.random.default_rng(0).normal(size=(2, H, W, DIM)).astype("float32")


@pytest.mark.parametrize("shift_size", [0, SHIFT])
def test_a_bare_call_inside_a_model_does_not_raise(batch, shift_size):
    """The regression itself: the first call through a Model must not raise."""
    model = SwmsaWrapper(shift_size)
    out = model(batch, training=False)
    assert tuple(out.shape) == (2, H, W, DIM)
    assert np.all(np.isfinite(np.asarray(out)))


@pytest.mark.parametrize("shift_size", [0, SHIFT])
def test_predict_inside_a_model_does_not_raise(batch, shift_size):
    model = SwmsaWrapper(shift_size)
    assert model.predict(batch, verbose=0).shape == (2, H, W, DIM)


@pytest.mark.parametrize("shift_size", [0, SHIFT])
def test_fit_inside_a_model_does_not_raise(batch, shift_size):
    model = SwmsaWrapper(shift_size)
    model.compile(optimizer="sgd", loss="mse")
    target = np.random.default_rng(1).normal(size=batch.shape).astype("float32")
    history = model.fit(batch, target, epochs=1, verbose=0)
    assert np.isfinite(history.history["loss"][0])


def test_the_mask_carries_no_state(batch):
    """The mask is derived from config, so it must not become a weight.

    Guards the other direction: a future "fix" that routes the mask through
    ``self.add_weight`` would make a derived constant part of the checkpoint.
    ``shift_size`` must not change the weight count.
    """
    shifted = SwmsaWrapper(SHIFT)
    unshifted = SwmsaWrapper(0)
    shifted(batch, training=False)
    unshifted(batch, training=False)
    assert len(shifted.weights) == len(unshifted.weights)


def test_the_shifted_mask_is_still_live(batch):
    """A mask that silently became a no-op would pass every test above."""
    keras.utils.set_random_seed(3)
    shifted = ProgressiveFocusedAttention(
        dim=DIM, num_heads=HEADS, window_size=WINDOW, shift_size=SHIFT
    )
    keras.utils.set_random_seed(3)
    unshifted = ProgressiveFocusedAttention(
        dim=DIM, num_heads=HEADS, window_size=WINDOW, shift_size=0
    )
    delta = np.abs(
        np.asarray(_unwrap(shifted(batch, training=False)))
        - np.asarray(_unwrap(unshifted(batch, training=False)))
    ).max()
    assert delta > 1e-3, (
        f"shift_size={SHIFT} produced output within {delta} of shift_size=0: the "
        "SW-MSA mask is no longer doing anything"
    )
