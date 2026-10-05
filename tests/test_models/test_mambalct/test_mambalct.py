"""Tests for the MambaLCT tracker: shapes, context sensitivity, serialization."""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.mambalct import MambaLCT, create_mambalct


def _tiny_model() -> MambaLCT:
    return MambaLCT(
        template_size=32,
        search_size=64,
        context_len=1,
        stage_dims=[64, 128, 512],
        stage_depths=[1, 1, 1],
        num_heads=[2, 4, 8],
        fusion_heads=8,
        head_hidden_dim=32,
        name="mambalct_tiny",
    )


@pytest.fixture
def tiny_inputs() -> tuple:
    rng = np.random.default_rng(0)
    template = rng.uniform(0, 1, size=(1, 32, 32, 3)).astype(np.float32)
    search = rng.uniform(0, 1, size=(1, 2, 64, 64, 3)).astype(np.float32)
    return template, search


def test_forward_shapes_and_finite(tiny_inputs: tuple) -> None:
    model = _tiny_model()
    out = model(list(tiny_inputs), training=False)
    assert tuple(out["scores"].shape) == (1, 2, 1)
    assert tuple(out["boxes"].shape) == (1, 2, 4)
    assert tuple(out["updated_context"].shape) == (1, 1, 512)
    for value in out.values():
        assert np.all(np.isfinite(np.asarray(value)))


def test_boxes_in_unit_range(tiny_inputs: tuple) -> None:
    model = _tiny_model()
    out = model(list(tiny_inputs), training=False)
    boxes = np.asarray(out["boxes"])
    assert boxes.min() >= 0.0 and boxes.max() <= 1.0


def test_context_changes_output(tiny_inputs: tuple) -> None:
    model = _tiny_model()
    template, search = tiny_inputs
    rng = np.random.default_rng(7)
    # Random (not constant) contexts: a per-token norm maps any constant
    # token to beta, so constant pairs can only differ by fp dust.
    ctx_a = rng.normal(0, 1, size=(1, 1, 512)).astype(np.float32)
    ctx_b = rng.normal(0, 1, size=(1, 1, 512)).astype(np.float32)
    out_a = model([template, search, ctx_a], training=False)
    out_b = model([template, search, ctx_b], training=False)
    # Incoming context must reach the frames: under a tail-append framing
    # the causal scan gives frames no path to later tokens, so this would be
    # bit-identical.
    box_diff = np.abs(np.asarray(out_a["boxes"]) - np.asarray(out_b["boxes"])).max()
    assert box_diff > 1e-4
    # History aggregation: different frames with the same context must move
    # the update substantially (the bridge output is the history summary).
    rng2 = np.random.default_rng(9)
    other_search = rng2.uniform(0, 1, size=(1, 2, 64, 64, 3)).astype(np.float32)
    out_c = model([template, other_search, ctx_a], training=False)
    hist_diff = np.abs(
        np.asarray(out_a["updated_context"]) - np.asarray(out_c["updated_context"])
    ).max()
    assert hist_diff > 1e-4
    assert np.all(np.isfinite(np.asarray(out_c["updated_context"])))


def _relative_weights(model: keras.Model):  # type: ignore[no-untyped-def]
    return sorted(w.path.split("/", 1)[-1] for w in model.weights)


def test_explicit_build_matches_lazy_build() -> None:
    explicit = _tiny_model()
    explicit.build([(None, 32, 32, 3), (None, 2, 64, 64, 3), (None, 1, 512)])
    lazy = _tiny_model()
    rng = np.random.default_rng(8)
    lazy(
        [
            rng.uniform(0, 1, size=(1, 32, 32, 3)).astype(np.float32),
            rng.uniform(0, 1, size=(1, 2, 64, 64, 3)).astype(np.float32),
            np.zeros((1, 1, 512), dtype=np.float32),
        ]
    )
    assert _relative_weights(explicit) == _relative_weights(lazy)
    # Pair nest builds the identical population (context is zeros-filled).
    pair = _tiny_model()
    pair.build([(None, 32, 32, 3), (None, 2, 64, 64, 3)])
    assert _relative_weights(pair) == _relative_weights(lazy)


def test_symbolic_frames_build() -> None:
    # Regression: Python arithmetic on shape tensors broke build_from_config
    # with a dynamic frames axis (and with it, pretrained weight loading).
    model = _tiny_model()
    model.build([(None, 32, 32, 3), (None, None, 64, 64, 3), (None, 1, 512)])
    assert model.built
    lazy = _tiny_model()
    lazy(
        [
            np.zeros((1, 32, 32, 3), dtype=np.float32),
            np.zeros((1, 2, 64, 64, 3), dtype=np.float32),
            np.zeros((1, 1, 512), dtype=np.float32),
        ]
    )
    assert _relative_weights(model) == _relative_weights(lazy)


def test_subclass_save_load_restores_weights(tiny_inputs: tuple) -> None:
    template, search = tiny_inputs
    context = np.zeros((1, 1, 512), dtype=np.float32)
    model = _tiny_model()
    model.build([(None, 32, 32, 3), (None, 2, 64, 64, 3), (None, 1, 512)])
    saved = [np.asarray(w).copy() for w in model.weights]
    assert saved, "donor has no weights to compare"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mambalct_sub.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        # BEFORE any forward on `loaded`: afterwards a build gap would be
        # silently filled with fresh random weights at the same count.
        assert len(loaded.weights) == len(saved)
        for donor, restored in zip(saved, loaded.weights):
            np.testing.assert_allclose(
                donor, np.asarray(restored), atol=0.0,
                err_msg="subclass weights were not restored",
            )
        out_orig = model([template, search, context], training=False)
        out_rest = loaded([template, search, context], training=False)
    for key in ("scores", "boxes", "updated_context"):
        np.testing.assert_allclose(
            np.asarray(out_orig[key]), np.asarray(out_rest[key]),
            atol=1e-6, rtol=0,
        )


def test_variant_contract() -> None:
    model = create_mambalct("mambalct-256")
    assert model.template_size == 128
    assert model.search_size == 256
    with pytest.raises(ValueError, match="mambalct-256"):
        create_mambalct("nope")
    with pytest.raises(NotImplementedError):
        create_mambalct("mambalct-256", pretrained=True)


def test_serialization_round_trip(tiny_inputs: tuple) -> None:
    template, search = tiny_inputs
    context = np.zeros((1, 1, 512), dtype=np.float32)
    t_in = keras.Input(shape=(32, 32, 3))
    s_in = keras.Input(shape=(2, 64, 64, 3))
    c_in = keras.Input(shape=(1, 512))
    tiny = _tiny_model()
    out = tiny([t_in, s_in, c_in])
    model = keras.Model([t_in, s_in, c_in], out)
    orig = model([template, search, context], training=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mambalct.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        rest = loaded([template, search, context], training=False)
    for key in ("scores", "boxes", "updated_context"):
        np.testing.assert_allclose(
            np.asarray(orig[key]), np.asarray(rest[key]), atol=1e-6, rtol=0
        )
