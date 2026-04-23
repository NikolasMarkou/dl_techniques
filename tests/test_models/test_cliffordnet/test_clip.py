"""Tests for :class:`CliffordCLIP`.

Focus: variant-ladder alignment with :class:`CliffordNet` / :class:`CliffordNetLM`,
optional head_dropout gate, and serialization round-trip. Intentionally narrow --
the CliffordNet backbone itself has broader coverage in ``test_model.py``, so we
only exercise CLIP-specific wiring here.
"""

from __future__ import annotations

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.models.cliffordnet import CliffordCLIP


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def tiny_build_shape():
    return {"image": (None, 32, 32, 3), "text": (None, 16)}


@pytest.fixture
def tiny_sample():
    rng = np.random.default_rng(0)
    return {
        "image": rng.standard_normal((2, 32, 32, 3)).astype("float32"),
        "text": rng.integers(0, 50257, size=(2, 16)).astype("int32"),
    }


def _build_nano(vocab_size=50257, **overrides):
    kwargs = dict(
        vocab_size=vocab_size,
        image_size=32,
        context_length=16,
        vision_patch_size=4,
        dropout_rate=0.0,
        vision_stochastic_depth_rate=0.0,
        text_stochastic_depth_rate=0.0,
    )
    kwargs.update(overrides)
    return CliffordCLIP.from_variant("nano", **kwargs)


# ---------------------------------------------------------------------
# Variant alignment
# ---------------------------------------------------------------------


def test_nano_matches_cliffordnet_nano_depth_and_shifts():
    """CLIP nano must share channels / depth / shifts with CliffordNet.nano."""
    m = _build_nano()
    assert m.vision_channels == 128
    assert m.text_channels == 128
    assert m.vision_depth == 12
    assert m.text_depth == 12
    assert m.vision_shifts == [1, 2]
    assert m.text_shifts == [1, 2]


def test_nano_g_enables_global_context_on_vision_only():
    m = CliffordCLIP.from_variant(
        "nano_g",
        vocab_size=50257,
        image_size=32,
        context_length=16,
        vision_patch_size=4,
    )
    m.build({"image": (None, 32, 32, 3), "text": (None, 16)})
    assert m.vision_use_global_context is True
    # Text tower must not receive a global-context branch -- mirrors
    # CliffordNetLM which has no global-context variant.
    assert all(
        getattr(block, "use_global_context", False) is False
        for block in m.text_blocks
    )


# ---------------------------------------------------------------------
# head_dropout gate
# ---------------------------------------------------------------------


def test_head_dropout_none_at_rate_zero(tiny_build_shape):
    m = _build_nano()
    m.build(tiny_build_shape)
    assert m.vision_head_dropout is None
    assert m.text_head_dropout is None


def test_head_dropout_materialised_at_positive_rate(tiny_build_shape):
    m = _build_nano(dropout_rate=0.1)
    m.build(tiny_build_shape)
    assert isinstance(m.vision_head_dropout, keras.layers.Dropout)
    assert isinstance(m.text_head_dropout, keras.layers.Dropout)


def test_rate_zero_is_deterministic_under_training_true(
    tiny_build_shape, tiny_sample
):
    """With head_dropout gated off and stochastic_depth=0, training=True is bit-exact."""
    m = _build_nano()
    m.build(tiny_build_shape)
    o1 = m(tiny_sample, training=True)
    o2 = m(tiny_sample, training=True)
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(o1["image_features"]),
        keras.ops.convert_to_numpy(o2["image_features"]),
    )


def test_positive_rate_is_stochastic_under_training_true(
    tiny_build_shape, tiny_sample
):
    m = _build_nano(dropout_rate=0.1)
    m.build(tiny_build_shape)
    o1 = m(tiny_sample, training=True)
    o2 = m(tiny_sample, training=True)
    # At dropout_rate=0.1 with stochastic_depth=0, head_dropout is the only
    # source of randomness; outputs must differ.
    assert not np.array_equal(
        keras.ops.convert_to_numpy(o1["image_features"]),
        keras.ops.convert_to_numpy(o2["image_features"]),
    )


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


def test_from_variant_serialization_round_trip(tiny_sample):
    """Save → load → forward produces matching outputs on every dict key."""
    m = _build_nano()
    m.build({"image": (None, 32, 32, 3), "text": (None, 16)})

    pre = m(tiny_sample, training=False)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "clip.keras")
        m.save(path)
        loaded = keras.models.load_model(path)
        post = loaded(tiny_sample, training=False)

    assert sorted(pre.keys()) == sorted(post.keys())
    for k in pre:
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(pre[k]),
            keras.ops.convert_to_numpy(post[k]),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"key {k} mismatches after round-trip",
        )
