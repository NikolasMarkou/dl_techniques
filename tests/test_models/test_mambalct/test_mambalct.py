"""Tests for the MambaLCT tracker: shapes, context sensitivity, serialization."""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.mambalct import MambaLCT, create_mambalct
from dl_techniques.models.vision.mambalct.uca_encoder import UcaEncoder


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
    """Incoming context reaches the frames, and the frames reach the context.

    The defects this guards are PATH defects: under a tail-append framing the
    causal scan gives the frames no path to the context, and a bridge that
    ignored the frames would return the context unchanged. Either makes the
    two compared outputs bit-identical. So the assertion is the house
    dead-component convention (``dead_component_oracle.component_response``,
    ``atol=0.0``): on a deterministic forward, any difference at all is a
    response.

    Until plan-2026-10-14T062616-81938e81 step 6 this compared against a pasted
    ``1e-4`` on an unseeded model. MEASURED over 20 init seeds on GPU 1
    (``scratchpad/step6/probes/b6.py``): box difference 5.6e-05 .. 4.3e-04
    (median 2.1e-04), BELOW 1e-4 for 3 of 20 seeds; context-history difference
    3.9e-03 .. 1.1e-02; the same input twice differed by exactly 0.0 in every
    output for all 20. The 1e-4 bar sat inside the effect's own spread, so the
    verdict was decided by the init draw (baseline B-6 red at 3.2e-05, green
    8/8 at step 1). The premise is now asserted, the model is seeded, and the
    bar is the bit-identity the defect produces.
    """
    keras.utils.set_random_seed(0)
    model = _tiny_model()
    template, search = tiny_inputs
    rng = np.random.default_rng(7)
    # Random (not constant) contexts: a per-token norm maps any constant
    # token to beta, so constant pairs can only differ by fp dust.
    ctx_a = rng.normal(0, 1, size=(1, 1, 512)).astype(np.float32)
    ctx_b = rng.normal(0, 1, size=(1, 1, 512)).astype(np.float32)
    out_a = model([template, search, ctx_a], training=False)
    out_a_again = model([template, search, ctx_a], training=False)
    out_b = model([template, search, ctx_b], training=False)

    # The premise of a zero bar: the forward is deterministic.
    for key in out_a:
        np.testing.assert_array_equal(
            np.asarray(out_a[key]), np.asarray(out_a_again[key]),
            err_msg=f"the same input gave two {key!r}; a difference below "
                    f"would be noise, not a response")

    # Incoming context must reach the frames: under a tail-append framing
    # the causal scan gives frames no path to later tokens, so this would be
    # bit-identical.
    box_diff = np.abs(np.asarray(out_a["boxes"]) - np.asarray(out_b["boxes"])).max()
    assert box_diff > 0.0, "the boxes do not depend on the incoming context"
    # History aggregation: different frames with the same context must move
    # the update (the bridge output is the history summary).
    rng2 = np.random.default_rng(9)
    other_search = rng2.uniform(0, 1, size=(1, 2, 64, 64, 3)).astype(np.float32)
    out_c = model([template, other_search, ctx_a], training=False)
    hist_diff = np.abs(
        np.asarray(out_a["updated_context"]) - np.asarray(out_c["updated_context"])
    ).max()
    assert hist_diff > 0.0, "the updated context does not depend on the frames"
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


def _tiny_uca_encoder() -> UcaEncoder:
    return UcaEncoder(stage_dims=[16, 32], stage_depths=[1, 1], num_heads=[2, 4])


def _count_calls(layer: keras.layers.Layer) -> list:
    """Wrap ``layer.call`` so each execution appends to the returned list.

    Interface contract: returns the (initially empty) list the wrapper appends
    ``1`` to on every call of ``layer.call``; the layer is mutated in place and
    is never restored, so use it on a throwaway instance.
    """
    calls: list = []
    original_call = layer.call

    def counting_call(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        return original_call(*args, **kwargs)

    layer.call = counting_call
    return calls


def test_the_uca_encoder_applies_its_stem_norm_exactly_once() -> None:
    """The stem is conv -> norm ONCE (the pre-migration encoder).

    The migration onto ``ConvStem`` (which already owns the norm) left a second
    ``self.stem_norm(x)`` in ``call()``: the stem norm ran twice and the encoder
    computed a different function. Counting the calls of ``stem.norm1`` during
    one forward sees it. Positive controls: the counter sees the single call of
    ``final_norm``, and it reads 2 on a planted encoder that applies the stem
    norm twice, so a reading of 1 is a real count and not a blind counter.
    """
    x = np.random.default_rng(0).uniform(0, 1, (1, 16, 16, 3)).astype("float32")

    encoder = _tiny_uca_encoder()
    stem_norm_calls = _count_calls(encoder.stem.norm1)
    final_norm_calls = _count_calls(encoder.final_norm)
    encoder(x, training=False)
    assert len(final_norm_calls) == 1, "the counter did not see final_norm run"
    assert len(stem_norm_calls) == 1, (
        f"the stem norm ran {len(stem_norm_calls)} times in one forward; "
        f"the pre-migration encoder applied it once"
    )

    planted = _tiny_uca_encoder()
    planted_calls = _count_calls(planted.stem.norm1)
    stem_call = planted.stem.call

    def stem_then_norm_again(*args, **kwargs):  # type: ignore[no-untyped-def]
        out = stem_call(*args, **kwargs)
        return planted.stem.norm1(out, training=kwargs.get("training"))

    planted.stem.call = stem_then_norm_again
    planted(x, training=False)
    assert len(planted_calls) == 2, "the counter cannot see a planted double norm"


def test_the_uca_encoder_has_no_stem_aliases() -> None:
    """``stem`` is the only stem handle: no ``stem_conv`` / ``stem_norm`` alias."""
    encoder = _tiny_uca_encoder()
    assert not hasattr(encoder, "stem_conv")
    assert not hasattr(encoder, "stem_norm")
