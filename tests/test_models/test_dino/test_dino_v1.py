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

Beyond the smoke test, this module pins the four defects repaired in
plan-2026-08-01T105809-dc0c402e step 3 (D-010, D-011, D-012 plus the
``image_size % patch_size`` guard). Each of those guards was RED-proven by
reverting the corresponding fix in place; see the plan's ``verification.md``.

MERGE RECORD (step 7). ``tests/test_models/test_dino/test_model_v1.py`` was the
naming outlier in this directory (every sibling is ``test_dino_v{1,2,3}.py``).
Its cases were NOT duplicates of this file's — they were complementary — so all
FOUR of them moved here verbatim-by-name, converging on v2's pattern of smoke +
round-trip inline in one file per version. Nothing was renamed and nothing was
dropped; the name-for-name correspondence is::

    test_model_v1.py                          -> test_dino_v1.py
    TestDINOHead::test_forward_shape          -> TestDINOHead::test_forward_shape
    TestDINOHead::test_keras_round_trip       -> TestDINOHead::test_keras_round_trip
    TestDINOv1::test_forward_logits           -> TestDINOv1::test_forward_logits
    TestDINOv1::test_keras_round_trip         -> TestDINOv1::test_keras_round_trip

Both round-trips were STRENGTHENED in the move: they now seed non-zero weights,
assert non-vacuity of the pre-save output, and were RED-proven by perturbing one
reloaded weight (a model that restores ZERO weights satisfies a shape-only
round-trip test — recorded repo lesson). ``test_model_v1.py`` is deleted; a
count of tests going UP is not evidence that nothing was lost, so the mapping
above is the evidence.
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


# =====================================================================
# MERGED FROM tests/test_models/test_dino/test_model_v1.py (step 7)
# =====================================================================
#
# The original file's header claimed these round-trips "were previously broken
# (DINOHead sublayers built lazily; DINOv1 patch_size/image_size deserialized as
# lists breaking `//`)". Both are FIXED; the claim is kept here as history, not
# as a live caveat, and neither round-trip is xfailed.


class TestDINOHead:

    def test_forward_shape(self):
        from dl_techniques.models.dino.dino_v1 import DINOHead

        head = DINOHead(in_dim=64, out_dim=128, nlayers=3,
                        hidden_dim=256, bottleneck_dim=64)
        x = np.random.default_rng(0).random((4, 64)).astype("float32")
        y = head(x, training=False)
        assert tuple(y.shape) == (4, 128)

    def test_keras_round_trip(self, tmp_path):
        """Asserts VALUES, not shapes.

        RED-proven by perturbing ONE reloaded weight after load: a model that
        restores ZERO weights satisfies a shape-only round-trip test, so the
        numeric assertion below is the whole guard. The non-vacuity floor
        matters too — a head whose output were ~0 everywhere would satisfy
        `assert_allclose` against any other ~0 output.
        """
        from dl_techniques.models.dino.dino_v1 import DINOHead

        head = DINOHead(in_dim=64, out_dim=128, nlayers=3,
                        hidden_dim=256, bottleneck_dim=64)
        model = keras.Sequential([keras.Input((64,)), head])
        _seed_trainable(model, 1234)
        x = np.random.default_rng(1).random((4, 64)).astype("float32")
        before = keras.ops.convert_to_numpy(model(x, training=False))
        assert np.abs(before).max() > 1e-3, (
            f"non-vacuity: the pre-save output is ~0 (absmax "
            f"{np.abs(before).max():.3e}), so this round-trip would pass against "
            "a model that restored nothing"
        )

        path = os.path.join(str(tmp_path), "dino_head.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="DINOHead differs after round-trip")


class TestDINOv1:

    def _model(self):
        from dl_techniques.models.dino.dino_v1 import create_dino_v1

        return create_dino_v1("small", image_size=32, patch_size=16,
                              num_classes=10)

    def test_forward_logits(self):
        model = self._model()
        x = np.random.default_rng(2).random((2, 32, 32, 3)).astype("float32")
        out = model(x, training=False)
        assert tuple(out.shape) == (2, 10)

    def test_keras_round_trip(self, tmp_path):
        """Asserts VALUES, not shapes — see TestDINOHead::test_keras_round_trip."""
        model = self._model()
        _seed_trainable(model, 4321)
        x = np.random.default_rng(3).random((2, 32, 32, 3)).astype("float32")
        before = keras.ops.convert_to_numpy(model(x, training=False))
        assert np.abs(before).max() > 1e-3, (
            f"non-vacuity: pre-save output absmax {np.abs(before).max():.3e}"
        )

        path = os.path.join(str(tmp_path), "dino_v1.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))
        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="DINOv1 differs after .keras round-trip")


def _seed_trainable(model, seed):
    """Assign seeded NON-ZERO values to every TRAINABLE weight.

    Deliberately not `model.weights`: non-trainable weights include
    normalization moving statistics and (on the v3 rope path) rotation tables,
    which must not be overwritten with noise. A default init leaves biases at
    zero, which makes several probes here structurally blind.
    """
    rng = np.random.default_rng(seed)
    for w in model.trainable_weights:
        w.assign(rng.normal(0.0, 0.3, size=w.shape).astype(w.dtype))


# =====================================================================
# step 7 — dtype-policy coverage (the DINO suite had NEVER had any)
# =====================================================================
#
# `keras.mixed_precision.set_global_policy` is process-global; the restore-safe
# `dtype_policy` fixture lives in this directory's own conftest.py (the
# `tests/test_layers/` one is NOT reachable from here — MEASURED, see that
# file's docstring).
#
# EVERY test below asserts the ACTIVE policy inside its own body. Requesting the
# fixture is not evidence: if the policy silently failed to apply, the body would
# run under float32 and pass.


def _assert_policy_is_active(expected):
    active = keras.mixed_precision.dtype_policy().name
    assert active == expected, (
        f"the dtype_policy fixture requested {expected!r} but the ACTIVE global "
        f"policy is {active!r} — this test is running in the wrong regime and "
        "every assertion below is vacuous"
    )
    return keras.mixed_precision.dtype_policy()


class TestDINOHeadMixedPrecision:

    def test_forward_survives_every_dtype_policy(self, dtype_policy):
        from dl_techniques.models.dino.dino_v1 import DINOHead

        policy = _assert_policy_is_active(dtype_policy)

        head = DINOHead(in_dim=64, out_dim=32, nlayers=3, hidden_dim=128,
                        bottleneck_dim=64)
        head.build((None, 64))
        assert head.compute_dtype == policy.compute_dtype
        x = np.random.default_rng(21).random((4, 64)).astype("float32")
        y = head(x, training=False)

        assert keras.backend.standardize_dtype(y.dtype) == policy.compute_dtype
        arr = np.asarray(keras.ops.convert_to_numpy(y), dtype="float64")
        assert np.all(np.isfinite(arr)), f"non-finite head output under {dtype_policy}"
        assert np.abs(arr).max() > 0.0, (
            f"the head output is identically zero under {dtype_policy} — this is "
            "the D-020 silent-collapse signature, not a passing forward test"
        )


class TestDINOv1MixedPrecision:

    def test_forward_survives_every_dtype_policy(self, dtype_policy):
        from dl_techniques.models.dino.dino_v1 import DINOv1

        policy = _assert_policy_is_active(dtype_policy)

        model = DINOv1(embed_dim=32, depth=2, num_heads=4, patch_size=16,
                       image_size=32, num_classes=5)
        x = np.random.default_rng(22).random((2, 32, 32, 3)).astype("float32")
        out = model(x, training=False)

        assert keras.backend.standardize_dtype(out.dtype) == policy.compute_dtype
        arr = np.asarray(keras.ops.convert_to_numpy(out), dtype="float64")
        assert arr.shape == (2, 5)
        assert np.all(np.isfinite(arr)), f"non-finite DINOv1 logits under {dtype_policy}"
        # Input-sensitivity: a collapsed / dead forward would be constant.
        other = np.random.default_rng(23).random((2, 32, 32, 3)).astype("float32")
        arr2 = np.asarray(keras.ops.convert_to_numpy(model(other, training=False)),
                          dtype="float64")
        assert np.abs(arr - arr2).max() > 1e-4, (
            f"DINOv1's output does not depend on its input under {dtype_policy}"
        )


# ---------------------------------------------------------------------
# D-020 — DINOHead's L2 normalization must not overflow fp16
# ---------------------------------------------------------------------
#
# THIS TEST MUST NOT BE SHRUNK. The defect is invisible at toy sizes: with
# bottleneck_dim=64 and default-initialized weights, `sum(x**2)` stays far
# under fp16's 65504 and the buggy and fixed versions agree to 1e-3. It only
# fires at the ORDINARY DINO head scale, which is why the dimensions below are
# the paper's (hidden_dim=2048, bottleneck_dim=256) and why the weights are
# seeded to a realistic post-training magnitude rather than left at init.
# MEASURED with the fix reverted: fp16 output absmax 0.0, 100% of entries
# exactly zero, against float32 absmax 0.2536 on bit-identical weights.


def _head_at_dino_scale(seed):
    from dl_techniques.models.dino.dino_v1 import DINOHead

    head = DINOHead(in_dim=384, out_dim=512, nlayers=3, hidden_dim=2048,
                    bottleneck_dim=256, norm_last_layer=False)
    head.build((None, 384))
    rng = np.random.default_rng(seed)
    for w in head.trainable_weights:
        w.assign(rng.normal(0.0, 0.5, size=w.shape).astype("float32"))
    return head


class TestDINOHeadFp16NormalizationOverflow:

    def test_fp16_head_output_is_not_silently_zero_at_dino_scale(self, dtype_policy):
        if dtype_policy != "mixed_float16":
            pytest.skip("this guard is specific to mixed_float16")
        _assert_policy_is_active("mixed_float16")

        head = _head_at_dino_scale(31)
        x = np.random.default_rng(32).normal(size=(8, 384)).astype("float32")
        y = np.asarray(keras.ops.convert_to_numpy(head(x, training=False)),
                       dtype="float64")

        # Establish that this configuration REALLY does overflow fp16 if the
        # reduction runs in compute_dtype — otherwise the assertion below is
        # a guard against nothing.
        pre = x
        for layer in head.mlp_layers:
            pre = layer(pre)
        sumsq = np.square(
            np.asarray(keras.ops.convert_to_numpy(pre), dtype="float64")
        ).sum(-1).max()
        assert sumsq > 65504.0, (
            f"non-vacuity FAILED: pre-normalization sum(x**2) is only {sumsq:.4g}, "
            "which fits in fp16, so this test cannot see the overflow it exists "
            "to catch. Increase bottleneck_dim / the seeded weight scale."
        )

        assert np.all(np.isfinite(y))
        assert np.count_nonzero(y) > 0, (
            "the DINOHead output is EXACTLY ZERO everywhere under mixed_float16 "
            f"(pre-normalization sum(x**2) = {sumsq:.4g} > 65504). The L2 "
            "normalization overflowed fp16 and x/inf collapsed to 0 — a silently "
            "dead projection head with no NaN, no Inf and no error. See D-020."
        )
        assert np.abs(y).max() > 1e-3, np.abs(y).max()

    def test_fp16_head_matches_float32_at_dino_scale(self):
        """The stronger form: same weights, fp16 vs float32, values compared."""
        previous = keras.mixed_precision.global_policy().name
        try:
            keras.mixed_precision.set_global_policy("float32")
            _assert_policy_is_active("float32")
            x = np.random.default_rng(33).normal(size=(8, 384)).astype("float32")
            y32 = np.asarray(
                keras.ops.convert_to_numpy(_head_at_dino_scale(31)(x, training=False)),
                dtype="float64",
            )

            keras.mixed_precision.set_global_policy("mixed_float16")
            _assert_policy_is_active("mixed_float16")
            y16 = np.asarray(
                keras.ops.convert_to_numpy(_head_at_dino_scale(31)(x, training=False)),
                dtype="float64",
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)
            assert keras.mixed_precision.global_policy().name == previous

        assert np.abs(y32).max() > 1e-2, np.abs(y32).max()
        # fp16 accumulates real error over a 2048-wide MLP; the assertion is
        # about AGREEMENT IN SCALE, not bit-equality. The defect this catches
        # drives the fp16 arm to exactly 0.0, i.e. a ratio of 0.
        ratio = float(np.abs(y16).max() / np.abs(y32).max())
        assert 0.5 < ratio < 2.0, (
            f"fp16 head output magnitude is {ratio:.4g}x the float32 one "
            f"(fp16 absmax {np.abs(y16).max():.4g}, float32 absmax "
            f"{np.abs(y32).max():.4g}). A ratio near 0 is the D-020 overflow."
        )
        np.testing.assert_allclose(y16, y32, atol=0.15 * np.abs(y32).max())


# ---------------------------------------------------------------------
# step 7 coverage audit — names in `__all__` with ZERO test coverage
# ---------------------------------------------------------------------
#
# The audit greps the test BODIES (not the filenames) for every name in
# `dl_techniques.models.dino.__all__`. Two names had zero body hits anywhere
# under `tests/`: `create_dino_teacher_student_pair` (an exported public factory,
# named in the README, never constructed by any test) and `ModelVariant`. Both
# are closed here; the remaining audit result is reported in the step's status.


class TestCreateDINOTeacherStudentPair:
    """`create_dino_teacher_student_pair` had ZERO test references before step 7."""

    def _pair(self, **kwargs):
        from dl_techniques.models.dino.dino_v1 import (
            create_dino_teacher_student_pair,
        )

        defaults = dict(variant="tiny", image_size=32, patch_size=16,
                        dino_out_dim=64)
        defaults.update(kwargs)
        return create_dino_teacher_student_pair(**defaults)

    def test_returns_two_distinct_projection_head_models(self):
        teacher, student = self._pair()
        assert teacher is not student
        assert teacher.name == "dino_teacher"
        assert student.name == "dino_student"
        # `include_projection_head=True` + `num_classes=0` -> the DINOHead output
        # width, NOT a class count. A pair that quietly returned backbones would
        # give embed_dim here instead.
        x = np.random.default_rng(41).random((2, 32, 32, 3)).astype("float32")
        t_out = np.asarray(keras.ops.convert_to_numpy(teacher(x, training=False)))
        s_out = np.asarray(keras.ops.convert_to_numpy(student(x, training=False)))
        assert t_out.shape == (2, 64), t_out.shape
        assert s_out.shape == (2, 64), s_out.shape

    def test_the_two_models_have_separate_variables_but_equal_values(self):
        """Two separate weight SETS, carrying identical VALUES (D-034).

        The two halves are orthogonal and both matter. Separate variables are
        what makes the EMA an EMA rather than an alias: a pair that shared
        weights would pass every shape assertion and make an EMA update a
        no-op. Equal values are what makes it DINO: the teacher is an EMA of
        the student's own trajectory starting from the student's own
        initialization, so a teacher drawn independently at random is not an
        EMA teacher at all. Before D-034 this factory returned an
        independently-drawn teacher (MEASURED: 55 of 157 tensors differed,
        outputs differed by 0.3002) and no test looked at values.
        """
        teacher, student = self._pair()
        assert len(teacher.weights) == len(student.weights)

        worst = max(
            float(np.abs(np.asarray(t) - np.asarray(s)).max())
            for t, s in zip(teacher.weights, student.weights)
        )
        # Non-vacuity: at least one tensor must be non-trivial, or "all equal"
        # would be satisfied by a pair of all-zero models.
        assert max(float(np.abs(np.asarray(s)).max())
                   for s in student.weights) > 0.1
        assert worst == 0.0, (
            f"the teacher was NOT initialized from the student: max|delta| "
            f"{worst:.6e}"
        )
        assert not any(tw is sw for tw in teacher.weights for sw in student.weights), (
            "teacher and student SHARE weight objects — an EMA update would be a "
            "no-op and the two towers are not independent"
        )
        # Independence is also observable: perturbing the student must not move
        # the teacher's output.
        x = np.random.default_rng(42).random((2, 32, 32, 3)).astype("float32")
        before = np.asarray(keras.ops.convert_to_numpy(teacher(x, training=False)))
        for w in student.trainable_weights:
            w.assign(np.asarray(w) + 0.25)
        after = np.asarray(keras.ops.convert_to_numpy(teacher(x, training=False)))
        np.testing.assert_allclose(before, after, atol=1e-6)

    def test_input_shape_is_refused(self):
        from dl_techniques.models.dino.dino_v1 import (
            create_dino_teacher_student_pair,
        )

        with pytest.raises(TypeError, match="image_size"):
            create_dino_teacher_student_pair(
                variant="tiny", image_size=32, patch_size=16, dino_out_dim=64,
                input_shape=(32, 32, 3),
            )


def test_model_variant_literal_lists_exactly_the_shipped_variants():
    """`ModelVariant` is exported but was never referenced by any test.

    It is a `Literal` type alias, so nothing enforces it at runtime — which is
    precisely why it can drift away from the tables it annotates without any
    test noticing.
    """
    import typing

    from dl_techniques.models.dino import ModelVariant
    from dl_techniques.models.dino.dino_v1 import DINOv1

    assert set(typing.get_args(ModelVariant)) == set(DINOv1.MODEL_VARIANTS)
    assert set(typing.get_args(ModelVariant)) == {
        "tiny", "small", "base", "large", "giant"
    }


# ---------------------------------------------------------------------
# D-033 — an explicit architecture override must WIN over the variant table,
# not collide with it. `DINOv1.from_variant` used to spell the four table keys
# out beside a bare `**kwargs`, so `from_variant("tiny", embed_dim=32)` raised
# `TypeError: DINOv1() got multiple values for keyword argument 'embed_dim'`.
# MEASURED at close-out: `DINOv2.from_variant` and `DINOv3.from_variant`
# already merged (`config.update(kwargs)`), so ONLY v1 was affected — and
# `create_dino_teacher_student_pair` forwards `**kwargs` straight into it,
# which is where the defect was actually hit (twice, at steps 8 and 9).
# ---------------------------------------------------------------------

_OVERRIDE = dict(embed_dim=32, depth=1, num_heads=2)


def test_from_variant_honours_an_explicit_architecture_override():
    from dl_techniques.models.dino.dino_v1 import DINOv1

    model = DINOv1.from_variant(
        "tiny", image_size=32, patch_size=16, **_OVERRIDE
    )
    # Non-vacuity: the override must actually DIFFER from the variant table,
    # or an equality assertion passes with the override silently discarded.
    table = DINOv1.MODEL_VARIANTS["tiny"]
    assert table["embed_dim"] != _OVERRIDE["embed_dim"], (
        "the tiny variant's embed_dim now equals the override; this test can "
        "no longer tell an honoured override from a discarded one"
    )
    assert model.embed_dim == 32
    assert model.depth == 1
    assert model.num_heads == 2


def test_from_variant_without_an_override_still_uses_the_variant_table():
    """Control: the merge must not have broken the default path."""
    from dl_techniques.models.dino.dino_v1 import DINOv1

    model = DINOv1.from_variant("tiny", image_size=32, patch_size=16)
    table = DINOv1.MODEL_VARIANTS["tiny"]
    assert model.embed_dim == table["embed_dim"]
    assert model.depth == table["depth"]
    assert model.num_heads == table["num_heads"]
    assert model.mlp_ratio == table["mlp_ratio"]


def test_teacher_student_pair_honours_an_architecture_override():
    """The call site the defect was actually hit from (steps 8 and 9)."""
    from dl_techniques.models.dino.dino_v1 import (
        create_dino_teacher_student_pair,
    )

    teacher, student = create_dino_teacher_student_pair(
        "tiny", image_size=32, patch_size=16, dino_out_dim=32, **_OVERRIDE
    )
    for m in (teacher, student):
        assert m.embed_dim == 32
        assert m.depth == 1
        assert m.num_heads == 2


def test_all_three_from_variant_agree_on_override_precedence():
    """Convergence guard: v1 was the outlier; keep all three merged."""
    from dl_techniques.models.dino.dino_v1 import DINOv1
    from dl_techniques.models.dino.dino_v2 import DINOv2
    from dl_techniques.models.dino.dino_v3 import DINOv3

    v1 = DINOv1.from_variant("tiny", image_size=32, patch_size=16, **_OVERRIDE)
    v2 = DINOv2.from_variant("tiny", image_size=32, patch_size=16, **_OVERRIDE)
    v3 = DINOv3.from_variant("tiny", image_size=32, patch_size=16, **_OVERRIDE)
    assert v1.embed_dim == 32
    assert v2.backbone.embed_dim == 32
    assert v3.embed_dim == 32
