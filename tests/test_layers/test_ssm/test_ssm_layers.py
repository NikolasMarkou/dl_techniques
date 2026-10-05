"""Behavioral tests for SelectiveSSMLayer and ContextMambaLayer."""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.ssm import ContextMambaLayer, SelectiveSSMLayer


@pytest.fixture
def tiny_sequence() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, size=(2, 8, 16)).astype(np.float32)


def test_selective_ssm_forward_is_finite(tiny_sequence: np.ndarray) -> None:
    layer = SelectiveSSMLayer(d_model=16, d_state=4, d_conv=3, expand=1)
    y = layer(tiny_sequence, training=False)
    y_np = np.asarray(y)
    assert y_np.shape == (2, 8, 16)
    assert np.all(np.isfinite(y_np))


def test_selective_ssm_serialization_round_trip(tiny_sequence: np.ndarray) -> None:
    inputs = keras.Input(shape=(8, 16))
    outputs = SelectiveSSMLayer(
        d_model=16, d_state=4, d_conv=3, expand=1
    )(inputs)
    model = keras.Model(inputs, outputs)
    original = np.asarray(model(tiny_sequence, training=False))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ssm.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        restored = np.asarray(loaded(tiny_sequence, training=False))
    np.testing.assert_allclose(original, restored, atol=1e-6, rtol=0)


def test_explicit_build_matches_lazy_build() -> None:
    def _relative(model: keras.Model):  # type: ignore[no-untyped-def]
        return sorted(w.path.split("/", 1)[-1] for w in model.weights)

    def _build():  # type: ignore[no-untyped-def]
        inputs = keras.Input(shape=(8, 16))
        outputs = SelectiveSSMLayer(d_model=16, d_state=4, d_conv=3, expand=1)(
            inputs
        )
        return keras.Model(inputs, outputs)

    explicit = _build()
    explicit.build((None, 8, 16))
    lazy = _build()
    lazy(np.zeros((1, 8, 16), dtype=np.float32))
    assert _relative(explicit) == _relative(lazy)


def test_context_mamba_forward_shapes() -> None:
    rng = np.random.default_rng(1)
    features = rng.normal(0, 1, size=(2, 3, 4, 16)).astype(np.float32)
    context = rng.normal(0, 1, size=(2, 1, 16)).astype(np.float32)
    layer = ContextMambaLayer(d_model=16, d_state=4, d_conv=3, expand=1)
    enhanced, updated = layer([features, context], training=False)
    assert tuple(enhanced.shape) == (2, 3, 4, 16)
    assert tuple(updated.shape) == (2, 1, 16)
    assert np.all(np.isfinite(np.asarray(enhanced)))
    assert np.all(np.isfinite(np.asarray(updated)))


def test_context_mamba_context_carries_history() -> None:
    rng = np.random.default_rng(2)
    features = rng.normal(0, 1, size=(1, 2, 4, 8)).astype(np.float32)
    ctx_a = np.zeros((1, 1, 8), dtype=np.float32)
    ctx_b = np.ones((1, 1, 8), dtype=np.float32)
    layer = ContextMambaLayer(d_model=8, d_state=4, d_conv=2, expand=1)
    enh_a, updated_a = layer([features, ctx_a], training=False)
    enh_b, updated_b = layer([features, ctx_b], training=False)
    # Updated context aggregates history including the incoming context.
    assert (
        np.abs(np.asarray(updated_a) - np.asarray(updated_b)).max() > 0.0
    )
    # Incoming context must reach the FRAMES: under the old tail-append
    # framing the causal scan gives frames no path to later tokens, so the
    # enhanced outputs would be bit-identical here.
    enh_diff = np.abs(np.asarray(enh_a) - np.asarray(enh_b)).max()
    assert enh_diff > 0.0


def test_zero_bridge_pins_context_update() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(0, 1, size=(1, 2, 4, 8)).astype(np.float32)
    context = rng.normal(0, 1, size=(1, 1, 8)).astype(np.float32)
    layer = ContextMambaLayer(d_model=8, d_state=4, d_conv=2, expand=1)
    layer.build([(None, 2, 4, 8), (None, 1, 8)])
    assert layer.bridge_token is not None
    saved = np.asarray(layer.bridge_token)
    layer.bridge_token.assign(np.zeros_like(saved))
    _, updated_zero = layer([features, context], training=False)
    np.testing.assert_allclose(
        np.asarray(updated_zero), np.zeros((1, 1, 8)), atol=0.0
    )
    layer.bridge_token.assign(saved)
    _, updated_live = layer([features, context], training=False)
    assert np.abs(np.asarray(updated_live)).max() > 0.0


def test_explicit_build_matches_lazy_build_context() -> None:
    def _relative(layer: ContextMambaLayer):  # type: ignore[no-untyped-def]
        return sorted(w.path.split("/", 1)[-1] for w in layer.weights)

    explicit = ContextMambaLayer(d_model=16, d_state=4, d_conv=3, expand=1)
    explicit.build([(None, 2, 4, 16), (None, 1, 16)])
    lazy = ContextMambaLayer(d_model=16, d_state=4, d_conv=3, expand=1)
    rng = np.random.default_rng(5)
    lazy(
        [
            rng.normal(0, 1, size=(1, 2, 4, 16)).astype(np.float32),
            rng.normal(0, 1, size=(1, 1, 16)).astype(np.float32),
        ]
    )
    assert _relative(explicit) == _relative(lazy)


def test_context_mamba_serialization_round_trip() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(0, 1, size=(2, 2, 4, 16)).astype(np.float32)
    context = rng.normal(0, 1, size=(2, 1, 16)).astype(np.float32)
    feat_in = keras.Input(shape=(2, 4, 16))
    ctx_in = keras.Input(shape=(1, 16))
    enhanced, updated = ContextMambaLayer(
        d_model=16, d_state=4, d_conv=3, expand=1
    )([feat_in, ctx_in])
    model = keras.Model([feat_in, ctx_in], [enhanced, updated])
    orig_enh, orig_ctx = model([features, context], training=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ctx.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        rest_enh, rest_ctx = loaded([features, context], training=False)
    np.testing.assert_allclose(
        np.asarray(orig_enh), np.asarray(rest_enh), atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        np.asarray(orig_ctx), np.asarray(rest_ctx), atol=1e-6, rtol=0
    )
