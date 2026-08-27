"""Guard: ``NonLocalAttention``'s additive mask must not produce NaN in fp16.

Why this test exists
--------------------
The layer's public contract is an ADDITIVE mask: ``0`` keeps a position, a large
negative value masks it out, documented in ``call()`` and in the module's ASCII
diagram. The mask was cast straight to the score dtype, so under
``keras.mixed_precision.set_global_policy("mixed_float16")`` the documented
``-1e9`` sentinel became ``-inf`` (float16 tops out at 65504) and a FULLY masked
query row softmaxed ``all -inf`` to ``0/0``. Measured on a 16x16 feature map with
one fully-masked row: **32 NaNs** in the output.

This is the same fp16 mask-NaN family that was already fixed at ten other sites in
this package by routing them through ``common.apply_attention_mask``. This site
could not use that helper -- it takes a KEEP PREDICATE, and converting an additive
mask to one means inferring polarity from magnitudes, which the helper explicitly
refuses to do and which would silently break every caller. So the mask is clamped
into the compute dtype's finite range instead, in a dtype where the sentinel is
still finite.

Why the existing suite could not see it: ``grep -n "mixed" `` on
``test_non_local_attention.py`` returns nothing -- the file has no
mixed-precision coverage at all.

Why this can fail if the implementation is wrong: dropping the clamp reinstates
the NaN, which is how this guard was proven RED.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.non_local_attention import NonLocalAttention

H = W = 16
CHANNELS = 16
ATTENTION_CHANNELS = 8
HW = H * W
SENTINEL = -1e9


@pytest.fixture(name="mixed_float16")
def _mixed_float16():
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(previous)


@pytest.fixture(name="feature_map")
def _feature_map():
    return (
        np.random.default_rng(0)
        .normal(size=(2, H, W, CHANNELS))
        .astype("float32")
    )


def _layer():
    keras.utils.set_random_seed(0)
    return NonLocalAttention(attention_channels=ATTENTION_CHANNELS)


@pytest.mark.usefixtures("mixed_float16")
def test_a_fully_masked_row_is_finite(feature_map):
    """The regression itself: `softmax(all -inf)` is `0/0`."""
    mask = np.zeros((2, HW, HW), dtype="float32")
    mask[:, 0, :] = SENTINEL

    out = np.asarray(_layer()(feature_map, attention_mask=mask, training=False))
    assert np.all(np.isfinite(out)), (
        f"{int(np.isnan(out).sum())} NaNs from a fully-masked query row: the "
        "additive sentinel overflowed to -inf in float16"
    )


@pytest.mark.usefixtures("mixed_float16")
def test_every_row_fully_masked_is_still_finite(feature_map):
    """The degenerate extreme, not just one bad row."""
    mask = np.full((2, HW, HW), SENTINEL, dtype="float32")
    out = np.asarray(_layer()(feature_map, attention_mask=mask, training=False))
    assert np.all(np.isfinite(out))


@pytest.mark.usefixtures("mixed_float16")
def test_the_mask_still_changes_the_output(feature_map):
    """A clamp that silently neutered the mask would pass the checks above.

    Why this can fail if the implementation is wrong: clamping to 0.0, or
    dropping the addition entirely, makes this delta exactly 0.0.
    """
    layer = _layer()
    mask = np.zeros((2, HW, HW), dtype="float32")
    mask[:, :, HW // 2:] = SENTINEL

    masked = np.asarray(layer(feature_map, attention_mask=mask, training=False))
    unmasked = np.asarray(layer(feature_map, training=False))
    delta = np.abs(masked.astype("float32") - unmasked.astype("float32")).max()
    assert delta > 1e-3, (
        f"masking half the keys moved the output by only {delta}: the additive "
        "mask is no longer reaching the scores"
    )


@pytest.mark.usefixtures("mixed_float16")
def test_the_output_carries_the_layers_own_compute_dtype(feature_map):
    """Finite is not enough -- a value can be finite and the wrong dtype."""
    layer = _layer()
    mask = np.zeros((2, HW, HW), dtype="float32")
    mask[:, 0, :] = SENTINEL
    out = layer(feature_map, attention_mask=mask, training=False)
    assert layer.compute_dtype == "float16"
    assert keras.backend.standardize_dtype(out.dtype) == layer.compute_dtype


def test_float32_masking_is_unchanged(feature_map):
    """The default policy must keep the numerics it already had."""
    layer = _layer()
    mask = np.zeros((2, HW, HW), dtype="float32")
    mask[:, :, HW // 2:] = SENTINEL
    out = np.asarray(layer(feature_map, attention_mask=mask, training=False))
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
