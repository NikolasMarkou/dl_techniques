"""Permanent build+forward smoke test for the dino family (v1).

Part of the 2026-06-15 model build/forward sweep. The construction-order crash
(`add_weight(cls_token)` before `super().__init__`) was FIXED in
plan_2026-06-15_39a31d4a (D-001): the CLS token is now owned by the
`ClassTokenPrepend` sub-layer. This test is therefore a REAL forward+finiteness
assertion, no longer an xfail.

Since plan-2026-08-01T105809-dc0c402e step 6 the package DOES export its public
surface, so ``from dl_techniques.models.dino import create_dino_v1`` works; these
tests keep importing from the submodule so a package-level export regression is
not the thing that makes them fail. The converged signature is
``create_dino_v1(variant, *, image_size, patch_size, num_classes, include_top, ...)``
— ``input_shape`` was REMOVED from the factory (it raises ``TypeError``); the input
shape is derived from ``image_size``. The patch grid is ``image_size // patch_size``,
so ``image_size=32, patch_size=16`` yields a 2x2 patch grid. ``num_classes=10`` +
default ``include_top=True`` returns logits ``(B, 10)``.

The DINOHead / DINOv1 ``.keras`` round-trips (previously known-broken) are now
FIXED and covered separately in ``test_model_v1.py``.

Beyond the smoke test, this module pins the four defects repaired in
plan-2026-08-01T105809-dc0c402e step 3 (D-010, D-011, D-012 plus the
``image_size % patch_size`` guard). Each of those guards was RED-proven by
reverting the corresponding fix in place; see the plan's ``verification.md``.
"""

import os

import keras
import numpy as np
import pytest


def _assert_finite(value):
    arr = np.asarray(value)
    assert arr is not None
    assert not np.any(np.isnan(arr))
    assert not np.any(np.isinf(arr))


def test_smoke_build_and_forward():
    from dl_techniques.models.dino.dino_v1 import create_dino_v1

    model = create_dino_v1(
        "small",
        image_size=32,
        patch_size=16,
        num_classes=10,
    )

    images = np.random.rand(2, 32, 32, 3).astype("float32")
    out = model(images, training=False)

    # Finiteness asserted OUTSIDE any try so a NaN/Inf fails loudly (no xfail).
    # Output may be a tensor (logits) or dict depending on head config.
    if isinstance(out, dict):
        for v in out.values():
            _assert_finite(v)
    else:
        _assert_finite(out)

    # include_top=True + num_classes=10 -> logits (B, 10).
    if not isinstance(out, dict):
        assert tuple(np.asarray(out).shape) == (2, 10)


# ---------------------------------------------------------------------
# D-010 — qkv_bias reaches the attention layer as `use_bias`
# ---------------------------------------------------------------------
#
# Before the fix, `attention_args` was gated on the string
# "multi_head_attention", which is not a registry key (the key is
# "multi_head"), so the gate never fired. Both halves of the old expression
# were independently dead: `create_attention_layer` SILENTLY DROPS an
# unrecognized `qkv_bias` kwarg instead of raising.


def _small_dino(**kwargs):
    """A 2-block DINOv1 on a 2x2 patch grid — cheap enough for weight probes."""
    from dl_techniques.models.dino.dino_v1 import DINOv1

    defaults = dict(
        embed_dim=32,
        depth=2,
        num_heads=4,
        patch_size=16,
        image_size=32,
        num_classes=5,
        input_shape=(32, 32, 3),
    )
    defaults.update(kwargs)
    return DINOv1(**defaults)


def _attention_bias_variables(model):
    biases = []
    for i in range(model.depth):
        block = model.get_layer(f"transformer_block_{i}")
        biases.extend(w for w in block.attention.weights if "bias" in w.path)
    return biases


def test_qkv_bias_false_creates_no_attention_bias_weights():
    model = _small_dino(qkv_bias=False)
    biases = _attention_bias_variables(model)
    assert biases == [], (
        "qkv_bias=False must produce attention sub-layers with NO bias "
        f"weights, found {[w.path for w in biases]}"
    )


def test_qkv_bias_true_creates_live_attention_bias_weights():
    model = _small_dino(qkv_bias=True)
    biases = _attention_bias_variables(model)

    # 2 bias variables per block (qkv projection + output projection).
    assert len(biases) == 2 * model.depth, (
        "qkv_bias=True must produce 2 attention bias weights per block, got "
        f"{[w.path for w in biases]}"
    )

    # Existence is not enough: biases initialize to ZEROS, so a bias that is
    # present but disconnected from the forward path would still pass a
    # count-only assertion. Seed every bias with a NON-ZERO value and require
    # the forward output to move.
    x = np.random.default_rng(11).random((2, 32, 32, 3)).astype("float32")
    before = keras.ops.convert_to_numpy(model(x, training=False))
    for i, w in enumerate(biases):
        w.assign(np.full(w.shape, 0.5 + 0.1 * i, dtype="float32"))
    after = keras.ops.convert_to_numpy(model(x, training=False))

    assert not np.allclose(before, after, atol=1e-6), (
        "Attention bias weights exist but do not affect the forward pass — "
        "the bias is not wired into the attention computation."
    )


# ---------------------------------------------------------------------
# F-05 — image_size % patch_size guard (v2/v3 both have it, v1 did not)
# ---------------------------------------------------------------------


# NOTE (MEASURED, and it makes the obvious version of this test VACUOUS):
# with the DINOv1 guard removed, `image_size=30, patch_size=16` still raises
# when `input_shape` AGREES with `image_size` — but from PatchEmbedding2D, with
# a different message ("Input height (30) must be divisible by patch height
# (16)"). A test matching only "must be divisible by" therefore passes with the
# guard deleted. The genuinely silent case is `input_shape` DISAGREEING with
# `image_size`: construction then succeeds with a WRONG num_patches (1 instead
# of 4) and only fails at forward time inside PositionalEmbedding.call with an
# opaque InvalidArgumentError. Both cases are pinned below, on the DINOv1
# message (which names image_size AND patch_size), not on the exception type.


def test_non_divisible_image_size_raises_from_dino_not_from_patch_embed():
    from dl_techniques.models.dino.dino_v1 import DINOv1

    with pytest.raises(Exception, match=r"image_size \(30, 30\) must be "
                                        r"divisible by patch_size"):
        DINOv1(
            embed_dim=32,
            depth=1,
            num_heads=4,
            patch_size=16,
            image_size=30,
            num_classes=5,
            input_shape=(30, 30, 3),
        )


def test_non_divisible_image_size_raises_at_construction_not_at_forward():
    """The case no downstream layer catches: input_shape != image_size."""
    from dl_techniques.models.dino.dino_v1 import DINOv1

    with pytest.raises(Exception, match=r"image_size \(30, 30\) must be "
                                        r"divisible by patch_size"):
        DINOv1(
            embed_dim=32,
            depth=1,
            num_heads=4,
            patch_size=16,
            image_size=30,
            num_classes=5,
            input_shape=(32, 32, 3),
        )


def test_divisible_image_size_still_builds():
    # Non-vacuity control for the guard above: the same construction with a
    # divisible size must NOT raise.
    model = _small_dino()
    assert model.num_patches == 4


# ---------------------------------------------------------------------
# D-011 — norm_last_layer is live (UnitNorm on the last projection kernel)
# ---------------------------------------------------------------------


def _head_column_norms(head):
    kernel = keras.ops.convert_to_numpy(head.last_layer.kernel)
    return np.linalg.norm(kernel, axis=0)


def _built_head(**kwargs):
    from dl_techniques.models.dino.dino_v1 import DINOHead

    defaults = dict(in_dim=16, out_dim=8, nlayers=2, hidden_dim=32,
                    bottleneck_dim=16)
    defaults.update(kwargs)
    head = DINOHead(**defaults)
    head.build((None, 16))
    return head


def test_norm_last_layer_true_pins_kernel_column_norms_at_build():
    head = _built_head(norm_last_layer=True)
    norms = _head_column_norms(head)
    np.testing.assert_allclose(
        norms, np.ones_like(norms), atol=1e-5,
        err_msg="norm_last_layer=True must give unit-norm last-layer columns",
    )
    assert head.last_layer.kernel_constraint is not None


def test_norm_last_layer_false_leaves_kernel_unconstrained():
    head = _built_head(norm_last_layer=False)
    norms = _head_column_norms(head)
    assert head.last_layer.kernel_constraint is None
    # Non-vacuity: the flag must MATTER. A truncated-normal kernel over 16 rows
    # is nowhere near unit column norm, so this separates the two branches.
    assert np.max(np.abs(norms - 1.0)) > 0.1, (
        f"norm_last_layer=False produced near-unit column norms {norms}; the "
        "True/False branches are indistinguishable and this guard is vacuous."
    )


def test_norm_last_layer_true_survives_optimizer_steps():
    from dl_techniques.models.dino.dino_v1 import DINOHead

    keras.utils.set_random_seed(7)
    head = DINOHead(in_dim=16, out_dim=8, nlayers=2, hidden_dim=32,
                    bottleneck_dim=16, norm_last_layer=True)
    model = keras.Sequential([keras.Input((16,)), head])
    model.compile(optimizer=keras.optimizers.SGD(0.5), loss="mse")
    x = np.random.default_rng(4).random((16, 16)).astype("float32")
    y = np.random.default_rng(5).random((16, 8)).astype("float32")
    model.fit(x, y, epochs=1, batch_size=8, verbose=0)

    norms = _head_column_norms(head)
    np.testing.assert_allclose(
        norms, np.ones_like(norms), atol=1e-5,
        err_msg="UnitNorm constraint did not hold after optimizer steps",
    )


def test_norm_last_layer_survives_keras_round_trip(tmp_path):
    from dl_techniques.models.dino.dino_v1 import DINOHead

    head = DINOHead(in_dim=16, out_dim=8, nlayers=2, hidden_dim=32,
                    bottleneck_dim=16, norm_last_layer=True)
    model = keras.Sequential([keras.Input((16,)), head])
    x = np.random.default_rng(6).random((4, 16)).astype("float32")
    before = keras.ops.convert_to_numpy(model(x, training=False))

    path = os.path.join(str(tmp_path), "head.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    after = keras.ops.convert_to_numpy(loaded(x, training=False))
    np.testing.assert_allclose(before, after, atol=1e-5)

    reloaded_head = loaded.layers[0]
    assert reloaded_head.norm_last_layer is True
    assert reloaded_head.last_layer.kernel_constraint is not None
    norms = _head_column_norms(reloaded_head)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------
# D-012 — get_last_selfattention raises instead of returning zeros
# ---------------------------------------------------------------------


def test_get_last_selfattention_raises_naming_the_missing_capability():
    model = _small_dino()
    x = np.random.default_rng(8).random((2, 32, 32, 3)).astype("float32")
    with pytest.raises(NotImplementedError, match="return_attention_scores"):
        model.get_last_selfattention(x)


# ---------------------------------------------------------------------
# F-05 — normalization_type rename + variant key-set parity
# ---------------------------------------------------------------------


def test_normalization_type_is_configurable_and_round_trips():
    from dl_techniques.models.dino.dino_v1 import DINOv1

    model = _small_dino(normalization_type="rms_norm")
    config = model.get_config()
    assert config["normalization_type"] == "rms_norm"
    assert "norm_layer" not in config
    rebuilt = DINOv1.from_config(config)
    assert rebuilt.normalization_type == "rms_norm"


def test_variant_key_sets_match_across_the_dino_family():
    from dl_techniques.models.dino.dino_v1 import DINOv1
    from dl_techniques.models.dino.dino_v2 import DINOv2VisionTransformer
    from dl_techniques.models.dino.dino_v3 import DINOv3

    v1_keys = set(DINOv1.MODEL_VARIANTS)
    assert v1_keys == {"tiny", "small", "base", "large", "giant"}
    assert v1_keys == set(DINOv2VisionTransformer.MODEL_VARIANTS)
    assert v1_keys == set(DINOv3.MODEL_VARIANTS)


def test_giant_variant_shares_the_family_dimensions():
    from dl_techniques.models.dino.dino_v1 import DINOv1
    from dl_techniques.models.dino.dino_v2 import DINOv2VisionTransformer
    from dl_techniques.models.dino.dino_v3 import DINOv3

    shared = ("embed_dim", "depth", "num_heads", "mlp_ratio")
    v1 = DINOv1.MODEL_VARIANTS["giant"]
    v2 = DINOv2VisionTransformer.MODEL_VARIANTS["giant"]
    v3 = DINOv3.MODEL_VARIANTS["giant"]
    for key in shared:
        assert v1[key] == v2[key] == v3[key], f"giant disagrees on {key}"
