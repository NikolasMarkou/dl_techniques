"""
Smoke + serialization tests for VQVAERotationTrick.

Covers:
- Init + forward shape on (B, 32, 32, 3)
- One-step fit produces finite loss
- .keras save/load round-trip atol=1e-5
- norm_type knob exercised: layer_norm + rms_norm
- One case per gradient_mode (rotation / reflection / no_grad_scale / ste)
- create_normalization_layer is actually used inside auto encoder/decoder
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.vq_vae_rotation.model import VQVAERotationTrick


@pytest.fixture
def sample_images():
    keras.utils.set_random_seed(0)
    return ops.cast(keras.random.uniform((4, 32, 32, 3), 0.0, 1.0), "float32")


class TestIndicesRoundTripMatchesCall:
    """``decode_from_indices(encode_to_indices(x))`` must equal ``model(x)``.

    Also the reachability statement: ``distance_mode`` is a public constructor
    argument on ``VQVAERotationTrick`` (and on ``create_vq_vae_rotation``), defaulting
    to ``'euclidean'``, so ``'cosine'`` is opt-in but reachable from the model — not
    layer-only. That is what makes the asymmetry a model defect and not just a layer
    one: a trained index prior could not reproduce the decoder's training-time input.

    RED before the fix (plan step 19): the cosine arm's two paths disagreed by the
    ``||x|| / ||e_k||`` factor ``_lookup`` applied and ``quantize_from_indices``
    omitted. The euclidean arm is the control and passed both ways.
    """

    @pytest.mark.parametrize("distance_mode", ["euclidean", "cosine"])
    def test_decode_from_indices_matches_call(self, sample_images, distance_mode):
        m = VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16,
            input_shape=(32, 32, 3), hidden_channels=32,
            downsample_factor=4, num_res_blocks=1, norm_type="layer_norm",
            gradient_mode="ste", distance_mode=distance_mode,
        )
        direct = ops.convert_to_numpy(m(sample_images, training=False))
        via_indices = ops.convert_to_numpy(
            m.decode_from_indices(m.encode_to_indices(sample_images))
        )
        np.testing.assert_allclose(direct, via_indices, atol=1e-5, rtol=1e-5)


class TestForward:
    def test_forward_shape(self, sample_images):
        m = VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16,
            input_shape=(32, 32, 3), hidden_channels=32,
            downsample_factor=4, num_res_blocks=1, norm_type="layer_norm",
        )
        out = m(sample_images)
        assert out.shape == sample_images.shape


class TestTraining:
    def test_one_step_fit(self, sample_images):
        m = VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16,
            input_shape=(32, 32, 3), hidden_channels=32,
            downsample_factor=4, num_res_blocks=1, norm_type="layer_norm",
        )
        m.compile(optimizer="adam")
        hist = m.fit(sample_images, sample_images, epochs=2, batch_size=2, verbose=0)
        losses = hist.history["loss"]
        assert all(np.isfinite(l) for l in losses)


class TestNormTypes:
    @pytest.mark.parametrize("norm_type", ["layer_norm", "rms_norm"])
    def test_norm_type_build(self, sample_images, norm_type):
        m = VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16,
            input_shape=(32, 32, 3), hidden_channels=32,
            downsample_factor=4, num_res_blocks=1, norm_type=norm_type,
        )
        out = m(sample_images)
        assert out.shape == sample_images.shape


class TestGradientModes:
    @pytest.mark.parametrize(
        "mode", ["rotation", "reflection", "no_grad_scale", "ste"]
    )
    def test_grad_mode_forward(self, sample_images, mode):
        m = VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16,
            gradient_mode=mode,
            input_shape=(32, 32, 3), hidden_channels=16,
            downsample_factor=4, num_res_blocks=1, norm_type="layer_norm",
        )
        out = m(sample_images)
        assert out.shape == sample_images.shape


class TestSerialization:
    def test_save_load_round_trip(self, sample_images):
        m = VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16,
            input_shape=(32, 32, 3), hidden_channels=32,
            downsample_factor=4, num_res_blocks=1, norm_type="layer_norm",
        )
        orig = m(sample_images, training=False)
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "vqvae_rot.keras")
            m.save(fp)
            loaded = keras.models.load_model(fp)
            new = loaded(sample_images, training=False)
            np.testing.assert_allclose(
                ops.convert_to_numpy(orig),
                ops.convert_to_numpy(new),
                atol=1e-5, rtol=1e-5,
            )

    def test_save_load_rms_norm(self, sample_images):
        m = VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16,
            input_shape=(32, 32, 3), hidden_channels=32,
            downsample_factor=4, num_res_blocks=1, norm_type="rms_norm",
        )
        orig = m(sample_images, training=False)
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "vqvae_rot_rms.keras")
            m.save(fp)
            loaded = keras.models.load_model(fp)
            new = loaded(sample_images, training=False)
            np.testing.assert_allclose(
                ops.convert_to_numpy(orig),
                ops.convert_to_numpy(new),
                atol=1e-5, rtol=1e-5,
            )


class TestFactoryIntegration:
    def test_norm_factory_in_source(self):
        """SC8: create_normalization_layer must be referenced in the model module."""
        import inspect
        from dl_techniques.models.vq_vae_rotation import model as model_module
        src = inspect.getsource(model_module)
        assert "create_normalization_layer" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


class TestRotationTrickForwardIdentity:
    """Every gradient_mode must emit the CODEBOOK vector in the forward pass."""

    @pytest.mark.parametrize(
        "mode", ["rotation", "reflection", "no_grad_scale", "ste"])
    def test_forward_equals_the_indexed_codebook_vector(self, mode):
        """call() must agree with encode -> quantize_from_indices -> decode.

        RED-proof for `reflection`: it reflected about the hyperplane with
        normal (u + v), which maps u -> -v, so its forward output was exactly
        -q. A single Householder reflection cannot map u -> +v about that
        normal; the two-reflection `rotation` path corrects for it and
        `reflection` had no second term.

        The existing suite parametrizes all four modes and asserts only the
        output SHAPE, which is invariant under a sign flip -- that is why this
        survived.
        """
        from dl_techniques.layers.vector_quantizer_rotation_trick import (
            VectorQuantizerRotationTrick,
        )

        keras.utils.set_random_seed(0)
        vq = VectorQuantizerRotationTrick(
            num_embeddings=16, embedding_dim=8, gradient_mode=mode)
        vq.build((None, 8))

        z = np.random.default_rng(0).normal(size=(32, 8)).astype("float32")

        out = keras.ops.convert_to_numpy(vq(z, training=False))
        idx = vq.get_codebook_indices(z)
        ref = keras.ops.convert_to_numpy(vq.quantize_from_indices(idx))

        # Sign-discriminating and scale-free: compare the distance to +ref
        # against the distance to -ref. This cannot be satisfied by loosening a
        # tolerance, and it is exactly the axis the defect lived on.
        d_plus = float(np.abs(out - ref).max())
        d_minus = float(np.abs(out + ref).max())
        assert d_plus < 0.1 * d_minus, (
            f"gradient_mode={mode!r}: call() is closer to -q ({d_minus:.3e}) "
            f"than it should be relative to +q ({d_plus:.3e}) -- the forward "
            f"pass is not emitting the codebook vector")

        # The residual floor is the rotation-trick arithmetic in float32 with
        # the eps-regularised norms; it is ~5.3e-05 for all three rotation
        # modes and ~3e-08 for the exact 'ste' path.
        np.testing.assert_allclose(
            out, ref, atol=1e-4,
            err_msg=(f"gradient_mode={mode!r}: call() does not emit the "
                     f"codebook vector, so it disagrees with the index path"))


def _structured_images(n: int = 8, size: int = 32, seed: int = 0) -> np.ndarray:
    """Learnable images: smooth sinusoidal texture in [0, 1], seeded."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="ij"
    )
    images = []
    for _ in range(n):
        freq = rng.integers(1, 3)
        phase = rng.random()
        base = 0.5 + 0.5 * np.sin(2 * np.pi * freq * (xx + phase)) * np.cos(
            2 * np.pi * freq * (yy + phase)
        )
        images.append(np.stack([base, np.roll(base, 4, axis=0), 1.0 - base], -1))
    return np.clip(np.asarray(images, dtype="float32"), 0.0, 1.0)


class TestEMACodebookStability:
    """F-65 in the per-head quantizer.

    Same defect as `layers/vector_quantizer.py`: `ema_cluster_size` starts at
    zeros, `ema_embeddings` started at the codebook initializer, and the
    normalize divided by `count + 1e-5`. On step 1 an unassigned code became
    `0.99 * init / 1e-5` ~= 99000 * init.

    MEASURED on this configuration (num_heads=2, use_ema=True, 5 epochs,
    seed 1234): max|codebook| 4509.37 before the fix and 2.257 after. Note the
    unique-code count is a WEAK indicator here -- it read 8 before the fix and
    14 after, because two heads' index sets are pooled -- so the magnitude
    assertion, not the usage assertion, is what carries this proof.
    """

    def test_ema_codebook_does_not_blow_up(self):
        keras.utils.set_random_seed(1234)
        m = VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16,
            input_shape=(32, 32, 3), hidden_channels=32,
            downsample_factor=4, num_res_blocks=1, norm_type="layer_norm",
            use_ema=True, num_heads=2,
        )
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
        images = _structured_images(seed=0)
        m.fit(images, epochs=5, batch_size=4, verbose=0)

        codebook = np.asarray(ops.convert_to_numpy(m.quantizer.embeddings))
        max_abs = float(np.abs(codebook).max())
        assert np.isfinite(max_abs)
        assert max_abs < 50.0, (
            f"per-head EMA codebook blew up: max|codebook| = {max_abs:.4g} "
            f"after 5 epochs. MEASURED 4509 before the bias correction + "
            f"Laplace smoothing and 2.257 after; the bar is 50.0."
        )

        indices = np.asarray(ops.convert_to_numpy(m.encode_to_indices(images)))
        assert len(np.unique(indices)) > 1

    def test_one_ema_update_matches_a_transcribed_oracle(self):
        """One `_update_ema` must reproduce a hand-computed per-head oracle.

        AXIS NOTE, measured not assumed. The Laplace total is summed over the
        CODEBOOK axis K with `keepdims=True`, giving one total per head. That
        is the principled choice -- heads own independent codebooks -- but it
        is NOT observable at runtime here, and this test does not pretend
        otherwise: `cluster_size = sum(one_hot(indices), axis=0)` sums over N,
        so `sum_k cluster_size[h, k] == N` for EVERY head, identically. The
        per-head totals are therefore always equal, and pooling over H would
        scale numerator and denominator by the same H, cancelling. The K-axis
        sum is kept because it is the one that stays correct if counts ever
        become per-head unequal; the claim being tested below is the FORMULA
        (debias + smoothing), which is observable.

        Oracle: after a single update from the zero state, a code that received
        assignments must equal the mean of the vectors assigned to it (the
        debias cancels `1 - decay` exactly), and an unassigned code must be
        ~0 (its numerator is exactly 0).
        """
        from dl_techniques.layers.vector_quantizer_rotation_trick import (
            VectorQuantizerRotationTrick,
        )

        num_heads, k, head_dim = 2, 4, 3
        vq = VectorQuantizerRotationTrick(
            num_embeddings=k, embedding_dim=num_heads * head_dim,
            num_heads=num_heads, use_ema=True, gradient_mode="ste",
        )
        vq.build((None, num_heads * head_dim))

        rng = np.random.default_rng(0)
        n = 10
        flat_heads = rng.normal(size=(n, num_heads, head_dim)).astype("float32")
        # Head 0 uses codes {0, 1}; head 1 uses code {2} only.
        idx = np.zeros((n, num_heads), dtype="int32")
        idx[:, 0] = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1], dtype="int32")
        idx[:, 1] = 2

        vq._update_ema(
            keras.ops.convert_to_tensor(flat_heads),
            keras.ops.convert_to_tensor(idx),
        )
        got = np.asarray(ops.convert_to_numpy(vq.embeddings))
        assert got.shape == (num_heads, k, head_dim)

        for h in range(num_heads):
            for code in range(k):
                members = flat_heads[idx[:, h] == code, h, :]
                if len(members):
                    np.testing.assert_allclose(
                        got[h, code], members.mean(axis=0), atol=1e-4,
                        err_msg=f"head {h} code {code} is not the cluster mean",
                    )
                else:
                    assert np.abs(got[h, code]).max() < 1e-3, (
                        f"head {h} code {code} received nothing yet has "
                        f"magnitude {np.abs(got[h, code]).max():.4g}; before "
                        f"the fix an unassigned code normalized to "
                        f"0.99 * init / 1e-5 ~= 99000 * init"
                    )
