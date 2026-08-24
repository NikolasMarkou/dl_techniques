"""Tests for the dino family (v3).

Originally a build+forward smoke test from the 2026-06-15 model sweep
(plan_2026-06-15_2a23a001, D-004): v3 had never been forward-run end-to-end
because the `.item()` in `_build_encoder` (NumPy-only, crashes on a
`keras.ops.linspace` tensor) was fixed to `float(r)` in that plan.

dino has no ``__init__`` exports, so import from the submodule directly.
``image_size`` accepts an int OR a ``(H, W)`` tuple. The patch grid is derived
from ``image_size // patch_size`` at build time, so a small ``image_size`` is
legal: with ``patch_size=16`` a 32x32 image yields a 2x2 patch grid.

plan-2026-08-01T105809-dc0c402e step 5 added the RoPE / honest-docstring /
``get_last_selfattention`` / ``image_size``-normalization cases below, including
``.keras`` round-trips at BOTH ``positional_embedding_type`` values. (The old
header claimed a ``DINOHead`` round-trip was "known-broken"; that is FALSE — it
round-trips — so the claim is gone rather than carried forward.)
"""

import re

import keras
import numpy as np
import pytest


def _assert_finite(value):
    arr = np.asarray(value)
    assert arr is not None
    assert not np.any(np.isnan(arr))
    assert not np.any(np.isinf(arr))


def test_smoke_build_and_forward():
    from dl_techniques.models.dino.dino_v3 import create_dino_v3

    # v3 forwards cleanly (plan_2026-06-15_2a23a001) -- no xfail safety net, so
    # any future regression in build/forward fails loudly.
    model = create_dino_v3(
        "small",
        image_size=(32, 32),
        num_classes=10,
    )

    images = np.random.rand(2, 32, 32, 3).astype("float32")
    out = model(images, training=False)

    # Output may be a tensor (logits) or dict depending on head config.
    if isinstance(out, dict):
        for v in out.values():
            _assert_finite(v)
    else:
        _assert_finite(out)

    # include_top=True + num_classes=10 -> logits (B, 10).
    if not isinstance(out, dict):
        assert tuple(np.asarray(out).shape) == (2, 10)


# =====================================================================
# plan-2026-08-01T105809-dc0c402e / step 5
# =====================================================================

from dl_techniques.models.dino.dino_v3 import DINOv3  # noqa: E402
from dl_techniques.layers.attention.group_query_attention import (  # noqa: E402
    GroupedQueryAttention,
)
from dl_techniques.layers.embedding.rotary_position_embedding import (  # noqa: E402
    RotaryPositionEmbedding,
)

# 32x32 image / 16x16 patches -> a 2x2 patch grid -> 4 patch tokens + 1 CLS.
_IMG, _PATCH, _DIM, _DEPTH, _HEADS = 32, 16, 32, 2, 4


def _tiny_v3(**overrides):
    kwargs = dict(
        image_size=_IMG,
        patch_size=_PATCH,
        num_classes=0,
        include_top=False,
        embed_dim=_DIM,
        depth=_DEPTH,
        num_heads=_HEADS,
    )
    kwargs.update(overrides)
    return DINOv3(**kwargs)


def _seed_trainable(model, seed):
    """Assign seeded NON-ZERO values to every TRAINABLE weight.

    Deliberately NOT ``model.weights``: ``RotaryPositionEmbedding`` stores its
    cos/sin rotation tables as NON-trainable ``add_weight`` variables, so a
    blanket re-assignment would overwrite the rotation with noise and make every
    RoPE measurement below meaningless. A default init would also make several of
    these probes structurally blind (biases init to zeros).
    """
    rng = np.random.default_rng(seed)
    for w in model.trainable_weights:
        w.assign(rng.normal(0.0, 0.35, size=w.shape).astype("float32"))


def _permute_patch_blocks(images, order):
    """Permute the 2x2 grid of 16x16 image blocks.

    Because ``PatchEmbedding2D`` is a stride==kernel Conv2D, each block maps to
    exactly one token, so permuting blocks permutes the patch-token sequence and
    changes nothing else. That this construction really is a pure token
    permutation is not asserted by hand — it is PROVEN by the control arm below,
    which is permutation-equivariant only if the construction is exact.
    """
    b = images.reshape(images.shape[0], 2, _PATCH, 2, _PATCH, 3)
    b = b.transpose(0, 1, 3, 2, 4, 5).reshape(images.shape[0], 4, _PATCH, _PATCH, 3)
    b = b[:, list(order)]
    b = b.reshape(images.shape[0], 2, 2, _PATCH, _PATCH, 3).transpose(0, 1, 3, 2, 4, 5)
    return np.ascontiguousarray(b.reshape(images.shape))


class TestDINOv3RoPE:
    """RoPE must be LIVE, not merely configured.

    ``create_attention_layer`` SILENTLY DROPPED an unknown key (MEASURED, D-010 —
    then the opposite of ``create_ffn_layer``, which raises), so a rope kwarg sent
    to an attention type that does not accept it yielded a working-looking model
    with RoPE entirely absent, and every shape/forward/round-trip test still passed.
    HISTORICAL as of 2026-08-17 (plan-2026-08-17T183311-79c63e38/D-011): that
    factory now raises too. The guard is NOT redundant — the raise only catches an
    UNDECLARED key, while the failure this class pins is a rope kwarg the target
    type declares and never wires, plus the ``.assign()``-in-``build()`` trap below.
    "It built without error" therefore still proves NOTHING here.
    """

    def test_rope_path_wires_a_real_rotation_into_every_block(self):
        model = _tiny_v3(positional_embedding_type="rope")
        assert len(model.encoder_layers) == _DEPTH
        for blk in model.encoder_layers:
            attn = blk.attention
            assert isinstance(attn, GroupedQueryAttention), type(attn)
            # num_kv_heads == num_heads => num_groups == 1 => no K/V repeat,
            # i.e. this GQA is plain multi-head attention plus the rotation.
            assert attn.num_kv_heads == _HEADS
            assert attn.num_groups == 1
            # The silent-drop hazard lands exactly here: a dropped rope kwarg
            # leaves `attn.rope is None` while everything else still works.
            assert isinstance(attn.rope, RotaryPositionEmbedding)
            assert attn.rope_percentage == 1.0
            assert attn.rope_theta == 10000.0

    def test_learned_path_uses_plain_multi_head_attention(self):
        model = _tiny_v3(positional_embedding_type="learned")
        for blk in model.encoder_layers:
            assert not isinstance(blk.attention, GroupedQueryAttention)

    def test_rope_is_live_under_a_patch_token_permutation(self):
        """The load-bearing probe.

        Contrast design: "output changes when tokens are permuted" alone does NOT
        isolate RoPE — a ViT with a LEARNED absolute positional embedding is
        order-sensitive too. So the control is NOT the learned model. Both arms
        here are the SAME ``positional_embedding_type='rope'`` architecture with
        BIT-IDENTICAL trainable weights; the only difference is
        ``rope_percentage`` (1.0 vs 0.0), i.e. the rotation itself. Under
        ``'rope'`` no learned table exists, so the 0.0 arm carries NO positional
        information whatever and must be exactly permutation-equivariant: its CLS
        output cannot move. Any movement in the 1.0 arm is therefore attributable
        to the rotation and to nothing else.
        """
        rope = _tiny_v3(positional_embedding_type="rope", rope_percentage=1.0)
        ctrl = _tiny_v3(positional_embedding_type="rope", rope_percentage=0.0)
        _seed_trainable(rope, 4242)
        _seed_trainable(ctrl, 4242)
        assert len(rope.trainable_weights) == len(ctrl.trainable_weights)
        for a, b in zip(rope.trainable_weights, ctrl.trainable_weights):
            assert np.array_equal(np.asarray(a), np.asarray(b)), (
                "the two arms must differ ONLY by the rotation"
            )

        x = np.random.default_rng(7).random((3, _IMG, _IMG, 3)).astype("float32")
        xp = _permute_patch_blocks(x, [2, 0, 3, 1])
        assert not np.array_equal(x, xp), "the permutation must actually move pixels"

        r0 = np.asarray(rope(x, training=False))
        r1 = np.asarray(rope(xp, training=False))
        c0 = np.asarray(ctrl(x, training=False))
        c1 = np.asarray(ctrl(xp, training=False))

        # Non-vacuity: the features must be big enough for a delta to be visible.
        assert np.abs(r0).max() > 0.1, np.abs(r0).max()
        assert np.abs(c0).max() > 0.1, np.abs(c0).max()

        ctrl_delta = float(np.abs(c0 - c1).max())
        rope_delta = float(np.abs(r0 - r1).max())

        assert ctrl_delta < 1e-4, (
            "CONTROL BROKEN: with the rotation disabled and no learned positional "
            "embedding the model must be permutation-equivariant, but its CLS "
            f"output moved by {ctrl_delta:.3e}. Either the permutation is not a "
            "pure token permutation, or positional information is leaking in."
        )
        assert rope_delta > 100.0 * max(ctrl_delta, 1e-6), (
            "ROPE IS INERT: permuting the patch tokens left the CLS output "
            f"essentially unchanged (rope delta {rope_delta:.3e} vs control "
            f"{ctrl_delta:.3e}). A configured-but-dropped rope kwarg looks exactly "
            "like this."
        )
        assert rope_delta > 1e-2, rope_delta

    def test_learned_positional_embedding_is_omitted_under_rope(self):
        rope = _tiny_v3(positional_embedding_type="rope")
        learned = _tiny_v3(positional_embedding_type="learned")

        assert rope.pos_embed is None
        assert learned.pos_embed is not None

        rope_names = [w.path for w in rope.weights]
        learned_names = [w.path for w in learned.weights]
        assert not [n for n in rope_names if "positional_embedding" in n], rope_names
        assert [n for n in learned_names if "positional_embedding" in n]

        assert "positional_embedding" not in [layer.name for layer in rope.layers]
        assert "positional_embedding" in [layer.name for layer in learned.layers]

        # The table is (sequence_length, embed_dim) = (5, 32) = 160 parameters.
        n_rope = sum(int(np.prod(w.shape)) for w in rope.weights if "rope" not in w.path)
        n_learned = sum(int(np.prod(w.shape)) for w in learned.weights)
        assert n_learned - n_rope >= rope.sequence_length * _DIM - 0

    def test_rope_config_round_trips_through_get_config(self):
        model = _tiny_v3(
            positional_embedding_type="rope", rope_theta=5000.0, rope_percentage=0.5
        )
        cfg = model.get_config()
        assert cfg["positional_embedding_type"] == "rope"
        assert cfg["rope_theta"] == 5000.0
        assert cfg["rope_percentage"] == 0.5
        clone = DINOv3.from_config(cfg)
        assert clone.encoder_layers[0].attention.rope_theta == 5000.0
        assert clone.encoder_layers[0].attention.rope_percentage == 0.5

    @pytest.mark.parametrize("pos_type", ["learned", "rope"])
    def test_keras_round_trip_is_numerically_exact(self, pos_type, tmp_path):
        """New forward-path code needs its own round-trip proof.

        Asserts VALUES, not shapes: a model that restores ZERO weights satisfies
        a shape-only round-trip test (recorded repo lesson).
        """
        model = _tiny_v3(positional_embedding_type=pos_type)
        _seed_trainable(model, 99)
        x = np.random.default_rng(5).random((2, _IMG, _IMG, 3)).astype("float32")
        before = np.asarray(model(x, training=False))
        assert np.abs(before).max() > 0.1, "non-vacuity: outputs must be non-trivial"

        path = str(tmp_path / f"dino_v3_{pos_type}.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        after = np.asarray(reloaded(x, training=False))

        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)
        assert reloaded.positional_embedding_type == pos_type
        if pos_type == "rope":
            attn = reloaded.encoder_layers[0].attention
            assert isinstance(attn.rope, RotaryPositionEmbedding)
            # The rotation tables are non-trainable weights; they must come back
            # as a real rotation, not as zeros.
            cos = np.asarray(attn.rope.cos_cached)
            assert np.abs(cos).max() > 0.5, cos.max()

    def test_invalid_positional_embedding_type_is_rejected(self):
        with pytest.raises(ValueError, match="positional_embedding_type must be"):
            _tiny_v3(positional_embedding_type="rotary")


class TestDINOv3GetLastSelfAttention:
    def test_raises_naming_the_missing_flag(self):
        model = _tiny_v3()
        x = np.random.default_rng(1).random((2, _IMG, _IMG, 3)).astype("float32")
        with pytest.raises(NotImplementedError, match="return_attention_weights"):
            model.get_last_selfattention(x)

    def test_raises_on_the_rope_path_too(self):
        model = _tiny_v3(positional_embedding_type="rope")
        x = np.random.default_rng(1).random((2, _IMG, _IMG, 3)).astype("float32")
        with pytest.raises(NotImplementedError, match="TransformerLayer"):
            model.get_last_selfattention(x)

    def test_the_rope_attention_operator_itself_does_expose_probabilities(self):
        """Pins the measured fact the raise's message asserts.

        ``GroupedQueryAttention`` CAN return its attention probabilities; the
        capability is unreachable only because ``TransformerLayer.call`` does not
        forward the flag. If a future change makes it reachable, the raise above
        should be revisited rather than treated as a permanent truth.
        """
        model = _tiny_v3(positional_embedding_type="rope")
        attn = model.encoder_layers[-1].attention
        tokens = np.random.default_rng(2).normal(
            size=(2, model.sequence_length, _DIM)
        ).astype("float32")
        _, probs = attn(tokens, training=False, return_attention_weights=True)
        probs = np.asarray(probs)
        assert probs.shape == (2, _HEADS, model.sequence_length, model.sequence_length)
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
        # ... and it is a function of the input, not a constant.
        other = np.random.default_rng(3).normal(
            size=(2, model.sequence_length, _DIM)
        ).astype("float32")
        _, probs2 = attn(other, training=False, return_attention_weights=True)
        assert float(np.abs(probs - np.asarray(probs2)).max()) > 1e-3


class TestDINOv3ImageSizeNormalization:
    def test_int_image_size_is_accepted_and_normalized(self):
        model = DINOv3(
            image_size=_IMG,
            patch_size=_PATCH,
            num_classes=0,
            include_top=False,
            embed_dim=_DIM,
            depth=1,
            num_heads=_HEADS,
        )
        assert model.image_size == (_IMG, _IMG)
        assert model.patch_size == (_PATCH, _PATCH)
        assert tuple(model.inputs[0].shape) == (None, _IMG, _IMG, 3)
        x = np.random.default_rng(0).random((2, _IMG, _IMG, 3)).astype("float32")
        assert np.asarray(model(x, training=False)).shape == (2, _DIM)

    def test_int_and_tuple_spellings_build_the_same_model(self):
        a = _tiny_v3(image_size=_IMG, patch_size=_PATCH)
        b = _tiny_v3(image_size=(_IMG, _IMG), patch_size=(_PATCH, _PATCH))
        assert a.image_size == b.image_size
        assert a.num_patches == b.num_patches
        assert a.count_params() == b.count_params()

    def test_int_image_size_reaches_the_divisibility_guard_not_a_typeerror(self):
        """RED-proven by reverting the normalization.

        Without it this raises ``TypeError: 'int' object is not subscriptable``
        from the guard's own subscript, so the message — not the type — is what
        distinguishes a normalized int from an unnormalized one. The regex pins
        the NORMALIZED tuple rendering.
        """
        with pytest.raises(
            ValueError,
            match=re.escape("image_size (30, 30) must be divisible by patch_size (16, 16)"),
        ):
            DINOv3(
                image_size=30,
                patch_size=16,
                num_classes=0,
                include_top=False,
                embed_dim=_DIM,
                depth=1,
                num_heads=_HEADS,
            )


# =====================================================================
# step 7 (plan-2026-08-01T105809-dc0c402e) — dtype-policy coverage
# =====================================================================
#
# The restore-safe parametrized `dtype_policy` fixture lives in this
# directory's own conftest.py; `tests/test_layers/conftest.py`'s copy is NOT
# reachable from `tests/test_models/` (MEASURED). BOTH positional-embedding
# modes are covered: under `rope` the rotation tables are non-trainable
# `add_weight` variables, and a dtype regime that mis-types them would be
# invisible to a learned-path-only test.


class TestDINOv3DtypePolicy:

    @pytest.mark.parametrize("pos_type", ["learned", "rope"])
    def test_forward_under_every_dtype_policy(self, dtype_policy, pos_type):
        active = keras.mixed_precision.dtype_policy().name
        assert active == dtype_policy, (
            f"the dtype_policy fixture requested {dtype_policy!r} but the ACTIVE "
            f"global policy is {active!r} — this test is running in the wrong "
            "regime and every assertion below is vacuous"
        )
        policy = keras.mixed_precision.dtype_policy()

        model = _tiny_v3(positional_embedding_type=pos_type)
        x = np.random.default_rng(31).random((2, _IMG, _IMG, 3)).astype("float32")
        out = model(x, training=False)

        assert keras.backend.standardize_dtype(out.dtype) == policy.compute_dtype
        arr = np.asarray(keras.ops.convert_to_numpy(out), dtype="float64")
        assert arr.shape == (2, _DIM)
        assert np.all(np.isfinite(arr)), (
            f"non-finite DINOv3/{pos_type} output under {dtype_policy}"
        )

        # Input-sensitivity: a dead or collapsed forward is constant, and a
        # constant output passes every shape and finiteness assertion above.
        other = np.random.default_rng(32).random((2, _IMG, _IMG, 3)).astype("float32")
        arr2 = np.asarray(keras.ops.convert_to_numpy(model(other, training=False)),
                          dtype="float64")
        assert np.abs(arr - arr2).max() > 1e-4, (
            f"DINOv3/{pos_type}'s output does not depend on its input under "
            f"{dtype_policy}"
        )

    def test_rope_rotation_tables_stay_a_real_rotation_under_mixed_float16(
        self, dtype_policy
    ):
        """The rope-specific dtype risk, isolated.

        `RotaryPositionEmbedding` caches cos/sin as NON-trainable weights. Under
        `mixed_float16` a table stored in fp16 (or zeroed) would leave the
        forward pass finite and shape-correct while destroying the rotation —
        exactly the inert-RoPE failure `TestDINOv3RoPE` exists to catch, but in
        a regime that test never runs in.
        """
        if dtype_policy != "mixed_float16":
            pytest.skip("this guard is specific to mixed_float16")
        assert keras.mixed_precision.dtype_policy().name == "mixed_float16"

        model = _tiny_v3(positional_embedding_type="rope")
        rope = model.encoder_layers[0].attention.rope
        cos = np.asarray(keras.ops.convert_to_numpy(rope.cos_cached), dtype="float64")
        sin = np.asarray(keras.ops.convert_to_numpy(rope.sin_cached), dtype="float64")
        assert np.all(np.isfinite(cos)) and np.all(np.isfinite(sin))
        assert np.abs(cos).max() > 0.5, np.abs(cos).max()
        # cos^2 + sin^2 == 1 is what makes it a ROTATION rather than an
        # arbitrary elementwise scaling.
        np.testing.assert_allclose(cos ** 2 + sin ** 2, 1.0, atol=1e-3)
