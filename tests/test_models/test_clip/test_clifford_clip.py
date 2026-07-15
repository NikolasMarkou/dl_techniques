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
import tensorflow as tf

from dl_techniques.models.clip.clifford_clip import CliffordCLIP


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def tiny_build_shape():
    return {"image": (None, 64, 64, 3), "text": (None, 16)}


@pytest.fixture
def tiny_sample():
    rng = np.random.default_rng(0)
    return {
        "image": rng.standard_normal((2, 64, 64, 3)).astype("float32"),
        "text": rng.integers(0, 50257, size=(2, 16)).astype("int32"),
    }


def _build_nano(vocab_size=50257, **overrides):
    kwargs = dict(
        vocab_size=vocab_size,
        image_size=64,
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
    """CLIP nano vision tower is staged ([D, D, 2D, 2D] per D-002) but
    preserves total depth against the pre-refactor isotropic ladder; the
    text tower remains isotropic per D-001."""
    m = _build_nano()
    # Vision tower (hierarchical): stage_channels [D, D, 2D, 2D],
    # stage_depths sum to the legacy depth, and the head is sized off the
    # last-stage channel count.
    assert m.vision_stage_channels == [128, 128, 256, 256]
    assert m.vision_stage_depths == [3, 3, 3, 3]
    assert sum(m.vision_stage_depths) == 12
    assert m.vision_stage_shifts == [
        [1, 2], [1, 2], [1, 2, 4], [1, 2, 4]
    ]
    # Derived back-compat scalars.
    assert m.vision_channels == 256       # last-stage channels (head sizing)
    assert m.vision_depth == 12           # sum of stage depths
    assert m.vision_shifts == [1, 2]      # representative (stage 0)
    # Text tower (still isotropic).
    assert m.text_channels == 128
    assert m.text_depth == 12
    assert m.text_shifts == [1, 2]


def test_vision_body_halves_spatial_per_stage():
    """For image=64, patch=4 (post-stem 16x16) and 4 stages, the post-stem
    feature map must halve at each PatchMerging boundary: 16 -> 8 -> 4 -> 2.
    """
    m = _build_nano()
    m.build({"image": (None, 64, 64, 3), "text": (None, 16)})
    # Three merges between four stages.
    assert len(m.vision_merge_layers) == 3
    spatial = 64 // m.vision_patch_size  # 16
    channels = m.vision_stage_channels[0]
    for i, merge in enumerate(m.vision_merge_layers):
        out_shape = merge.compute_output_shape((1, spatial, spatial, channels))
        assert out_shape[1] == (spatial + 1) // 2, (
            f"merge {i}: spatial {spatial} -> {out_shape[1]} (expected {(spatial+1)//2})"
        )
        spatial = out_shape[1]
        # PatchMerging emits 2*src; an optional Dense projects to next stage.
        proj = m.vision_merge_projections[i]
        if proj is None:
            channels = 2 * channels
        else:
            channels = m.vision_stage_channels[i + 1]
    # After 3 halvings starting from 16: 16 -> 8 -> 4 -> 2.
    assert spatial == 2


def test_nano_g_enables_global_context_on_vision_only():
    m = CliffordCLIP.from_variant(
        "nano_g",
        vocab_size=50257,
        image_size=64,
        context_length=16,
        vision_patch_size=4,
    )
    m.build({"image": (None, 64, 64, 3), "text": (None, 16)})
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
    m.build({"image": (None, 64, 64, 3), "text": (None, 16)})

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


def test_text_use_global_context_round_trip(tiny_build_shape, tiny_sample):
    """text_use_global_context=True survives get_config + .keras round-trip
    and produces matching outputs (mirrors vision_use_global_context)."""
    m = _build_nano(text_use_global_context=True)
    m.build(tiny_build_shape)
    assert m.get_config()["text_use_global_context"] is True
    # Text tower blocks must actually receive the global-context branch.
    assert all(
        getattr(block, "use_global_context", False) is True
        for block in m.text_blocks
    )

    pre = m(tiny_sample, training=False)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "clip_gt.keras")
        m.save(path)
        loaded = keras.models.load_model(path)
        post = loaded(tiny_sample, training=False)

    assert loaded.get_config()["text_use_global_context"] is True
    for k in pre:
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(pre[k]),
            keras.ops.convert_to_numpy(post[k]),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"key {k} mismatches after round-trip",
        )


def test_text_use_global_context_default_false(tiny_build_shape):
    """Default preserves prior behavior: text tower has no global-context."""
    m = _build_nano()
    m.build(tiny_build_shape)
    assert m.get_config()["text_use_global_context"] is False
    assert all(
        getattr(block, "use_global_context", False) is False
        for block in m.text_blocks
    )


# ---------------------------------------------------------------------
# T1 guard tests (gradient flow / fp16-fit / degenerate config / loss)
# ---------------------------------------------------------------------
#
# These guards target the bugs the review PROVED (B1 dead query pool at the
# old 32px config, B2 fp16 mask NaN) plus the S1 stage bound. Per the
# institutional rule "a guard that has not been shown RED is not a guard",
# test (c) was verified to FAIL (NaN loss / collapsed loss scale) against a
# one-off temporary revert of the B2 fix before being finalised.


def _guard_batch(n=4, seed=0):
    """Deterministic (image, text) batch for the gradient/loss guards."""
    rng = np.random.default_rng(seed)
    image = rng.standard_normal((n, 64, 64, 3)).astype("float32")
    # Keep a couple of pad (0) tokens so the text ctx pool exercises its
    # masked-softmax branch — the exact site of the B2 fp16 NaN bug.
    text = rng.integers(1, 50257, size=(n, 16)).astype("int32")
    text[:, -2:] = 0
    return {"image": image, "text": text}


def _toy_contrastive_loss(out):
    """Minimal contrastive objective: pull matched pairs together.

    ``-mean(diag(logits_per_image))`` depends on both towers' matched-pair
    features and on ``logit_scale``, so its gradient reaches every trainable
    weight — exactly what the dead-parameter guards need.
    """
    lpi = out["logits_per_image"]
    return -keras.ops.mean(keras.ops.diagonal(lpi))


def _ctx_attention(pool):
    """The ``AttentionPooling`` nested inside a ``SequencePooling`` ctx pool.

    Post-refactor the learned-query pools are generic ``SequencePooling``
    instances with ``strategy='attention'``; the learnable scoring layer lives
    at ``pool.learnable_components['attention']`` (an ``AttentionPooling``).
    """
    return pool.learnable_components["attention"]


def test_query_pool_weight_receives_nonzero_gradient():
    """B1 guard: both AttentionPooling scoring weights get a non-zero gradient.

    At the old 32px config the post-stem map collapsed to 1x1 and the vision
    context pool's softmax-over-1 was a dead-gradient no-op (invisible to any
    forward-only or aggregate-fraction check). At 64px the map is >=2x2, so
    the scoring weights must receive gradient. Asserted per-weight, not by
    aggregate.

    Post-refactor the learned-query pools are generic ``SequencePooling``
    instances (``vision_ctx_pool`` / ``text_ctx_pool``, replacing the old
    bespoke ``vision_query_pool`` / ``text_query_pool``); the learnable scoring
    path is the nested ``AttentionPooling``'s ``context_vector`` (direct analog
    of the old ``.query``) plus its ``attention_dense`` kernel that projects
    each token before scoring. Both must be alive for the pool to learn.
    Requires a ``learned_query*`` head_kind (the only ones with attention ctx
    pools); nano defaults to ``learned_query_residual``.
    """
    keras.utils.set_random_seed(0)
    m = _build_nano(head_kind="learned_query_residual")
    inputs = _guard_batch()

    v_att = _ctx_attention(m.vision_ctx_pool)
    t_att = _ctx_attention(m.text_ctx_pool)

    with tf.GradientTape(persistent=True) as tape:
        out = m(inputs, training=True)
        loss = _toy_contrastive_loss(out)

    for name, w in (
        ("vision_ctx_pool.context_vector", v_att.context_vector),
        ("vision_ctx_pool.attention_dense", v_att.attention_dense.kernel),
        ("text_ctx_pool.context_vector", t_att.context_vector),
        ("text_ctx_pool.attention_dense", t_att.attention_dense.kernel),
    ):
        g = tape.gradient(loss, w)
        assert g is not None, f"{name} gradient is None (disconnected)"
        g = keras.ops.convert_to_numpy(g)
        assert np.all(np.isfinite(g)), f"{name} gradient non-finite"
        assert np.any(g != 0.0), (
            f"{name} received an all-zero gradient — a dead pool "
            f"(the B1 1x1-collapse no-op)"
        )
    del tape


def test_all_trainable_weights_have_finite_nonzero_gradient():
    """Per-weight: every trainable weight gets a finite gradient, and NONE is
    all-zero. Tighter than plain-CLIP's aggregate ``>0.8*len`` threshold —
    a single dead parameter (per the B1 lesson) is a hard failure here.
    """
    keras.utils.set_random_seed(0)
    m = _build_nano()
    inputs = _guard_batch()

    with tf.GradientTape() as tape:
        out = m(inputs, training=True)
        loss = _toy_contrastive_loss(out)

    weights = list(m.trainable_weights)
    grads = tape.gradient(loss, weights)

    none_grad = [w.path for w, g in zip(weights, grads) if g is None]
    assert not none_grad, f"weights with None gradient (disconnected): {none_grad}"

    non_finite = []
    all_zero = []
    for w, g in zip(weights, grads):
        gv = keras.ops.convert_to_numpy(g)
        if not np.all(np.isfinite(gv)):
            non_finite.append(w.path)
        if not np.any(gv != 0.0):
            all_zero.append(w.path)

    assert not non_finite, f"weights with non-finite gradient: {non_finite}"
    # Zero legitimately-dead weights expected under this toy loss; if any
    # appear they are named here rather than absorbed by a loosened fraction.
    assert not all_zero, (
        f"{len(all_zero)}/{len(weights)} trainable weights received an "
        f"all-zero gradient (dead parameters): {all_zero}"
    )


def test_fp16_fit_moves_weights_no_nan_no_scale_collapse():
    """THE fp16 guard (B2). Under ``mixed_float16`` with a dynamic
    ``LossScaleOptimizer``, 5 real train steps must: (i) keep the loss finite
    every step, (ii) NOT collapse the loss scale, (iii) actually move weights.

    Proven RED against a one-off temporary revert of the B2 mask fix (the
    ``-1e9`` additive sentinel casts to fp16 ``-inf`` and ``0*-inf=NaN``
    poisons every real token through the text pool's softmax): NaN loss at
    step 0 and the dynamic scale collapses toward the fp16 minimum.
    """
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        keras.utils.set_random_seed(0)
        m = _build_nano()
        inputs = _guard_batch()

        opt = keras.mixed_precision.LossScaleOptimizer(
            keras.optimizers.Adam(1e-3)
        )

        # Snapshot weights (float64) before any step.
        m(inputs, training=False)  # materialise weights
        before = {
            w.path: keras.ops.convert_to_numpy(w).astype("float64")
            for w in m.trainable_weights
        }
        start_scale = float(opt.initial_scale)  # dynamic_scale exists post-apply

        losses = []
        for _ in range(5):
            with tf.GradientTape() as tape:
                out = m(inputs, training=True)
                loss = _toy_contrastive_loss(out)
                scaled = opt.scale_loss(loss)
            grads = tape.gradient(scaled, m.trainable_weights)
            opt.apply_gradients(zip(grads, m.trainable_weights))
            losses.append(float(keras.ops.convert_to_numpy(loss)))

        # (i) loss finite every step.
        assert np.all(np.isfinite(losses)), (
            f"loss went non-finite over 5 fp16 steps: {losses}"
        )

        # (ii) the dynamic loss scale did not collapse (it halves only when a
        # step's gradients are non-finite; a healthy run never collapses it).
        end_scale = float(keras.ops.convert_to_numpy(opt.dynamic_scale))
        assert end_scale >= start_scale / 16.0, (
            f"LossScaleOptimizer scale collapsed {start_scale:.3e} -> "
            f"{end_scale:.3e} over 5 steps — ~every step was rejected "
            f"(the B2 fp16 NaN)"
        )

        # (iii) weights actually moved.
        after = {
            w.path: keras.ops.convert_to_numpy(w).astype("float64")
            for w in m.trainable_weights
        }
        moved = sum(
            1 for p in before if np.max(np.abs(after[p] - before[p])) > 0.0
        )
        assert moved > 0, (
            "0 trainable weights moved over 5 fp16 fit steps — the model "
            "trained on nothing"
        )
    finally:
        keras.mixed_precision.set_global_policy("float32")


def test_degenerate_1x1_config_raises():
    """S1 guard: a config whose vision tower would collapse to 1x1 spatial
    RAISES at construction (instead of silently building a dead query pool).

    nano is 4 stages, default patch=4; at image_size=32 the post-stem map is
    8x8 and the final stage would be 8/2^4 < 1 -> rejected.
    """
    with pytest.raises(ValueError, match="spatial"):
        CliffordCLIP.from_variant("nano", vocab_size=50257, image_size=32)


def test_end_to_end_fit_with_clip_contrastive_loss():
    """Wire the ACTUAL production loss (``CLIPContrastiveLoss``) end-to-end:
    one real train step feeding the model's dict output into the loss; assert
    the loss is finite and at least one weight moves.
    """
    from dl_techniques.losses.clip_contrastive_loss import CLIPContrastiveLoss

    keras.utils.set_random_seed(0)
    m = _build_nano()
    inputs = _guard_batch()
    loss_fn = CLIPContrastiveLoss()

    opt = keras.optimizers.Adam(1e-3)
    m(inputs, training=False)  # materialise weights
    before = {
        w.path: keras.ops.convert_to_numpy(w).astype("float64")
        for w in m.trainable_weights
    }
    # Dummy y_true (contrastive loss derives targets from batch structure).
    y_true = keras.ops.zeros((inputs["image"].shape[0],))

    with tf.GradientTape() as tape:
        out = m(inputs, training=True)
        loss = loss_fn(y_true, out)
    grads = tape.gradient(loss, m.trainable_weights)
    opt.apply_gradients(zip(grads, m.trainable_weights))

    assert np.isfinite(float(keras.ops.convert_to_numpy(loss))), (
        "CLIPContrastiveLoss produced a non-finite loss"
    )
    after = {
        w.path: keras.ops.convert_to_numpy(w).astype("float64")
        for w in m.trainable_weights
    }
    moved = sum(
        1 for p in before if np.max(np.abs(after[p] - before[p])) > 0.0
    )
    assert moved > 0, "no weight moved after one CLIPContrastiveLoss step"


# ---------------------------------------------------------------------
# G1 opt-in vision positional encoding
# ---------------------------------------------------------------------


def test_vision_positional_encoding_default_off_is_byte_identical(
    tiny_build_shape,
):
    """Default (flag OFF): the model exposes ``vision_positional_encoding is
    False`` and materialises NO ``vision_pos_embed`` weight — the vision tower
    is byte-identical to the pre-G1 build."""
    m = _build_nano()
    m.build(tiny_build_shape)
    assert m.vision_positional_encoding is False
    assert m.vision_pos_embed is None
    assert m.get_config()["vision_positional_encoding"] is False
    assert not any(
        "vision_pos_embed" in w.name for w in m.weights
    ), "default-off model must not carry a positional-encoding weight"


def test_vision_positional_encoding_on_round_trips(tiny_sample):
    """Flag ON: a ``vision_pos_embed`` weight exists, changes the forward, and
    survives get_config + .keras round-trip (proves the build()-created
    conditional weight serialises — no lazy-build drop)."""
    m = _build_nano(vision_positional_encoding=True)
    m.build({"image": (None, 64, 64, 3), "text": (None, 16)})
    assert m.vision_positional_encoding is True
    assert m.get_config()["vision_positional_encoding"] is True
    assert any(
        "vision_pos_embed" in w.name for w in m.weights
    ), "flag-on model must carry a positional-encoding weight"

    # The non-zero positional weight must actually shift the image features
    # versus an otherwise-identical flag-off model (same seed/init not shared,
    # so just assert the flag-on features are not trivially the flag-off ones).
    off = _build_nano()
    off.build({"image": (None, 64, 64, 3), "text": (None, 16)})
    feat_on = keras.ops.convert_to_numpy(
        m.encode_image(tiny_sample["image"], training=False)
    )
    feat_off = keras.ops.convert_to_numpy(
        off.encode_image(tiny_sample["image"], training=False)
    )
    assert not np.allclose(feat_on, feat_off, rtol=1e-4, atol=1e-4), (
        "positional encoding did not change the image features"
    )

    pre = m(tiny_sample, training=False)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "clip_pos.keras")
        m.save(path)
        loaded = keras.models.load_model(path)
        post = loaded(tiny_sample, training=False)

    assert loaded.get_config()["vision_positional_encoding"] is True
    assert any("vision_pos_embed" in w.name for w in loaded.weights)
    for k in pre:
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(pre[k]),
            keras.ops.convert_to_numpy(post[k]),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"key {k} mismatches after round-trip",
        )


# ---------------------------------------------------------------------
# SequencePooling head-kind coverage (post-refactor: pools are
# SequencePooling instances, attention nested in learnable_components)
# ---------------------------------------------------------------------


ALL_HEAD_KINDS = ["plain", "mean_max", "learned_query", "learned_query_residual"]


@pytest.mark.parametrize("head_kind", ALL_HEAD_KINDS)
def test_all_head_kinds_forward_and_round_trip(tiny_sample, head_kind):
    """Every head_kind builds, forwards finite, and round-trips through
    ``.keras`` byte-consistently.

    Post-refactor all z_det/z_ctx pooling routes through generic
    ``SequencePooling`` instances (``*_det_pool`` mean / ``*_ctx_pool`` max or
    attention). This proves those nested pools serialize for EVERY head_kind —
    the attention ctx pools in particular build a sublayer symbolically during
    model build(), the classic site of a lazy-build round-trip weight drop.
    """
    m = _build_nano(head_kind=head_kind)
    m.build({"image": (None, 64, 64, 3), "text": (None, 16)})

    pre = m(tiny_sample, training=False)

    # Output dict contract + shape/finiteness on the two feature keys.
    assert sorted(pre.keys()) == sorted(
        ["image_features", "text_features",
         "logits_per_image", "logits_per_text", "logit_scale"]
    )
    for k in ("image_features", "text_features"):
        v = keras.ops.convert_to_numpy(pre[k])
        assert v.shape == (2, m.embed_dim), (
            f"{head_kind}: {k} shape {v.shape} != (2, {m.embed_dim})"
        )
        assert np.all(np.isfinite(v)), f"{head_kind}: {k} non-finite"

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "clip_head.keras")
        m.save(path)
        loaded = keras.models.load_model(path)
        post = loaded(tiny_sample, training=False)

    for k in ("image_features", "text_features"):
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(pre[k]),
            keras.ops.convert_to_numpy(post[k]),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"{head_kind}: {k} mismatches after .keras round-trip",
        )


def test_attention_pool_hidden_dim_matches_channels():
    """F3 guard: the attention ctx pools use ``attention_hidden_dim=<tower
    channels>`` — NOT the ``SequencePooling`` default of 256.

    A silent fall-through to the 256 default would (a) change the text pool's
    capacity (text_channels=128) and (b) be a latent shape/behavior regression.
    We introspect the nested ``AttentionPooling.hidden_dim`` and its
    ``context_vector`` last-dim, which must equal each tower's channel count.
    """
    m = _build_nano(head_kind="learned_query_residual")
    m.build({"image": (None, 64, 64, 3), "text": (None, 16)})

    v_att = _ctx_attention(m.vision_ctx_pool)
    t_att = _ctx_attention(m.text_ctx_pool)

    assert v_att.hidden_dim == m.vision_channels, (
        f"vision ctx pool hidden_dim {v_att.hidden_dim} != vision_channels "
        f"{m.vision_channels} (SequencePooling default-256 regression)"
    )
    assert t_att.hidden_dim == m.text_channels, (
        f"text ctx pool hidden_dim {t_att.hidden_dim} != text_channels "
        f"{m.text_channels} (SequencePooling default-256 regression)"
    )
    # context_vector last dim mirrors hidden_dim — belt-and-suspenders.
    assert v_att.context_vector.shape[-1] == m.vision_channels
    assert t_att.context_vector.shape[-1] == m.text_channels
