"""DINOv2's register tokens must be R independent learnable vectors.

Guard for C-3b (plan-2026-08-14T233721-d4f9beb2, step 33). Before the fix,
``dino_v2.py`` built the registers as
``Dense(embed_dim, use_bias=False)(ones((1, R, 1)))``. The input feature dim is
1, so the kernel is ``(1, D)`` and every one of the R output rows is
``1.0 * kernel[0]`` -- R bit-identical copies of ONE vector, sharing a single
gradient accumulator: ``D`` parameters where Darcet et al. (and the reference
DINOv2) use ``nn.Parameter(zeros(1, R, D))``, i.e. ``R*D``.

The defect is invisible to the existing ``test_register_tokens_forward``, which
asserts exactly the three things the broken model also satisfies: it builds, it
forwards finite, and it is input-sensitive.

What is deliberately NOT changed: registers stay POSITION-FREE (inserted after
the positional embedding). That is the D-009 anchor's property and it is correct.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.embedding.register_tokens import RegisterTokens
from dl_techniques.models.dino.dino_v2 import create_dino_v2


def _tiny_model(num_register_tokens=4):
    return create_dino_v2(
        "tiny",
        image_size=28,
        patch_size=14,
        num_classes=10,
        num_register_tokens=num_register_tokens,
    )


def _register_weights(model):
    """The (1, R, D) register bank, wherever it lives in the weight tree."""
    hits = [w for w in model.weights if "register" in w.path.lower()]
    assert hits, f"no register weight found; weights: {[w.path for w in model.weights]}"
    assert len(hits) == 1, f"expected one register bank, got {[w.path for w in hits]}"
    return hits[0]


class TestRegisterBankShape:
    def test_bank_is_r_by_d_not_one_by_d(self):
        model = _tiny_model(4)
        w = _register_weights(model)
        shape = tuple(w.shape)
        assert shape[-2] == 4, (
            f"register bank has {shape[-2]} token rows, expected 4 -- a (1, D) "
            f"kernel means all R tokens are one shared vector. shape={shape}"
        )

    def test_rows_are_pairwise_distinct_after_initialization(self):
        """The property the old code violated exactly: R bit-identical rows."""
        model = _tiny_model(4)
        bank = np.asarray(keras.ops.convert_to_numpy(_register_weights(model)))
        rows = bank.reshape(-1, bank.shape[-1])
        assert rows.shape[0] == 4

        scale = float(np.abs(rows).mean())
        assert scale > 0.0, "register bank initialized to exactly zero everywhere"
        tol = 1e-3 * scale  # a REAL tolerance, relative to the init scale
        for i in range(rows.shape[0]):
            for j in range(i + 1, rows.shape[0]):
                delta = float(np.max(np.abs(rows[i] - rows[j])))
                assert delta > tol, (
                    f"register rows {i} and {j} differ by only {delta:.3e} "
                    f"(tol {tol:.3e}) -- they are copies of one vector"
                )

    def test_the_probe_would_fail_if_all_rows_were_equal(self):
        """Anti-vacuity control for the tolerance above: run the same comparison
        against a deliberately tied bank and confirm it fires."""
        rows = np.tile(np.random.RandomState(0).randn(1, 8) * 0.02, (4, 1))
        scale = float(np.abs(rows).mean())
        tol = 1e-3 * scale
        deltas = [
            float(np.max(np.abs(rows[i] - rows[j])))
            for i in range(4)
            for j in range(i + 1, 4)
        ]
        assert all(d <= tol for d in deltas), "control bank was not actually tied"


class TestRegisterTokensGetIndependentGradients:
    def test_each_row_receives_its_own_gradient(self):
        model = _tiny_model(4)
        images = np.random.rand(2, 28, 28, 3).astype("float32")
        masks = np.zeros((2, 4), dtype=bool)

        bank = _register_weights(model)
        with tf.GradientTape() as tape:
            out = model([images, masks], training=True)
            # An asymmetric objective, so the R rows cannot share a gradient by
            # symmetry of the loss itself.
            weights = keras.ops.arange(0, keras.ops.shape(out)[-1], dtype="float32")
            loss = keras.ops.sum(out * weights)
        grad = tape.gradient(loss, bank)
        assert grad is not None, "register bank received no gradient at all"

        g = np.asarray(keras.ops.convert_to_numpy(grad)).reshape(4, -1)
        assert np.any(np.abs(g) > 0), "register gradient is identically zero"
        for i in range(4):
            for j in range(i + 1, 4):
                assert np.max(np.abs(g[i] - g[j])) > 0.0, (
                    f"rows {i} and {j} got identical gradients -- one accumulator"
                )


class TestPositionFreePropertyIsPreserved:
    """D-009: registers receive NO positional signal. Still true after the fix."""

    def test_pos_embed_is_still_sized_cls_plus_patches_only(self):
        model = _tiny_model(4)
        pos = [w for w in model.weights if "pos" in w.path.lower() and "embed" in w.path.lower()]
        assert pos, f"no positional embedding weight found: {[w.path for w in model.weights]}"
        num_patches = (28 // 14) ** 2
        assert any(
            num_patches + 1 in tuple(w.shape) for w in pos
        ), (
            "positional embedding is no longer sized CLS+patches -- registers "
            "must NOT be given a positional signal (D-009)"
        )

    def test_model_still_forwards_finite_and_input_sensitive(self):
        model = _tiny_model(4)
        masks = np.zeros((2, 4), dtype=bool)
        a = np.asarray(model([np.random.rand(2, 28, 28, 3).astype("float32"), masks]))
        b = np.asarray(model([np.random.rand(2, 28, 28, 3).astype("float32"), masks]))
        assert a.shape == (2, 10)
        assert np.all(np.isfinite(a))
        assert np.any(np.abs(a - b) > 1e-6)


class TestRegisterTokensLayerContract:
    def test_output_shape_and_batch_broadcast(self):
        layer = RegisterTokens(num_tokens=3, embed_dim=8)
        ref = keras.ops.zeros((5, 7, 8))
        out = keras.ops.convert_to_numpy(layer(ref))
        assert out.shape == (5, 3, 8)
        # Same token bank for every batch element.
        np.testing.assert_allclose(out[0], out[4])

    def test_width_disagreement_raises(self):
        layer = RegisterTokens(num_tokens=3, embed_dim=8)
        with pytest.raises(ValueError, match="embed_dim"):
            layer(keras.ops.zeros((2, 7, 9)))

    def test_zero_tokens_rejected(self):
        with pytest.raises(ValueError, match="num_tokens"):
            RegisterTokens(num_tokens=0, embed_dim=8)

    def test_round_trip(self, tmp_path):
        """A round trip must preserve the BANK, not just the two integers.

        The old body compared ``clone.num_tokens``/``clone.embed_dim`` against
        the values it had just passed in -- two integers read back out of the
        very config it wrote, on an UNBUILT clone that holds no bank at all.
        That is green for a layer whose R independent vectors collapse to one
        shared vector on reload, which is precisely the defect (C-3b) this file
        exists to guard.
        """
        layer = RegisterTokens(num_tokens=3, embed_dim=8, name="regs")
        cfg = layer.get_config()
        clone = RegisterTokens.from_config(cfg)
        assert clone.num_tokens == 3
        assert clone.embed_dim == 8

        # Values: save a built layer inside a model and reload it.
        inputs = keras.Input(shape=(7, 8))
        model = keras.Model(inputs, layer(inputs))
        bank = _register_weights(model)
        # Break the zero-init so a fresh bank is distinguishable from this one,
        # and so the R rows are provably DIFFERENT from each other.
        bank.assign(
            keras.ops.reshape(
                keras.ops.arange(3 * 8, dtype="float32") + 1.0, (1, 3, 8)
            )
        )

        probe = keras.ops.zeros((2, 7, 8))
        expected = keras.ops.convert_to_numpy(model(probe))
        path = str(tmp_path / "regs.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"RegisterTokens": RegisterTokens}
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(loaded(probe)), expected,
            rtol=1e-6, atol=1e-6,
        )
        # ...and the R rows must still be independent after the reload.
        reloaded_bank = keras.ops.convert_to_numpy(_register_weights(loaded))
        rows = reloaded_bank.reshape(3, 8)
        for i in range(3):
            for j in range(i + 1, 3):
                assert not np.allclose(rows[i], rows[j]), (
                    f"register rows {i} and {j} are identical after reload"
                )
