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


class TestRotationNormFloorDoesNotLeakMagnitude:
    """F-67: the rotation-trick norms must be FLOORED, not eps-regularised.

    ``sqrt(sum(x^2) + eps)`` inflates every norm, and because
    ``scale_eff = q_norm / x_norm`` multiplies the output, the inflation does not
    cancel: ``call()`` emitted ``(1 + O(eps/||x||^2)) * q`` while
    ``encode_to_indices -> quantize_from_indices -> decode`` returns the raw
    codebook row ``q``. The gap grows as the encoder's scale shrinks.

    MEASURED at HEAD (num_embeddings=16, embedding_dim=8, seeded N(0, 1) input
    scaled by s), ``max|call() - quantize_from_indices()|``:

        s      ||x||^2    rotation   reflection   no_grad_scale   ste
        1.00   8.2e+00    5.35e-05   5.34e-05     5.35e-05        2.98e-08
        0.10   8.2e-02    6.88e-05   5.18e-05     6.88e-05        7.45e-09
        0.01   8.2e-04    1.92e-03   6.69e-04     1.92e-03        1.86e-09

    i.e. a 4% relative magnitude leak at the small-norm end -- exactly the regime
    a fresh ``Conv2D(embedding_dim, 1)`` head produces -- against an exact ``ste``
    path three to six orders below it. With the floor every rotation mode drops to
    ~2e-08, the float32 arithmetic floor, and stops depending on the input scale.

    The bar below is set at 1e-6: comfortably above the measured post-fix 2.6e-08,
    and 700x below the 1.9e-03 the defect produced. The ``ste`` row is the
    built-in control -- it never used these norms and must pass both before and
    after.
    """

    @pytest.mark.parametrize(
        "mode", ["ste", "rotation", "reflection", "no_grad_scale"])
    @pytest.mark.parametrize("input_scale", [1.0, 0.1, 0.01])
    def test_call_emits_the_codebook_vector_at_every_input_scale(
            self, mode, input_scale):
        from dl_techniques.layers.vector_quantizer_rotation_trick import (
            VectorQuantizerRotationTrick,
        )

        keras.utils.set_random_seed(0)
        vq = VectorQuantizerRotationTrick(
            num_embeddings=16, embedding_dim=8, gradient_mode=mode)
        vq.build((None, 8))

        z = (np.random.default_rng(0).normal(size=(32, 8))
             * input_scale).astype("float32")

        out = np.asarray(ops.convert_to_numpy(vq(z, training=False)))
        idx = vq.get_codebook_indices(z)
        ref = np.asarray(ops.convert_to_numpy(vq.quantize_from_indices(idx)))

        gap = float(np.abs(out - ref).max())
        assert gap < 1e-6, (
            f"gradient_mode={mode!r}, input scaled by {input_scale}: call() and "
            f"the index path disagree by {gap:.3e}. An eps-inflated norm leaks a "
            f"continuous magnitude channel through the discrete bottleneck; the "
            f"norms must be floored with max(||x||, eps)."
        )

    def test_the_leak_is_scale_independent_after_the_floor(self):
        """The defect's signature was scale DEPENDENCE, so pin its absence."""
        from dl_techniques.layers.vector_quantizer_rotation_trick import (
            VectorQuantizerRotationTrick,
        )

        gaps = []
        for input_scale in (1.0, 0.01):
            keras.utils.set_random_seed(0)
            vq = VectorQuantizerRotationTrick(
                num_embeddings=16, embedding_dim=8, gradient_mode="rotation")
            vq.build((None, 8))
            z = (np.random.default_rng(0).normal(size=(32, 8))
                 * input_scale).astype("float32")
            out = np.asarray(ops.convert_to_numpy(vq(z, training=False)))
            ref = np.asarray(ops.convert_to_numpy(
                vq.quantize_from_indices(vq.get_codebook_indices(z))))
            # Relative to the codebook magnitude, which the scaling does not move.
            gaps.append(float(np.abs(out - ref).max())
                        / float(np.abs(ref).max()))

        assert gaps[1] < 10.0 * gaps[0] + 1e-9, (
            f"relative disagreement grew from {gaps[0]:.3e} at unit scale to "
            f"{gaps[1]:.3e} at 1/100 scale -- the error still scales like "
            f"eps/||x||^2, so a norm is still eps-inflated somewhere"
        )


class TestRotationAuxiliaryLossesReachTheObjective:
    """F-66, rotation-trick half: `train_step`/`test_step` sum `self.losses`.

    Same defect and same fix as
    `tests/test_models/test_vq_vae/test_model.py::TestVQVAEAuxiliaryLossesReachTheObjective`;
    this model has its own copy of both steps. The module's architecture diagram
    already said `total_loss = recon(x, x_rec) + sum(layer.losses)`.
    """

    @staticmethod
    def _build(regularizer):
        keras.utils.set_random_seed(0)
        encoder = keras.Sequential([
            keras.layers.Conv2D(8, 3, strides=2, padding="same",
                                activation="relu",
                                kernel_regularizer=regularizer),
            keras.layers.Conv2D(4, 1, padding="same",
                                kernel_regularizer=regularizer),
        ])
        decoder = keras.Sequential([
            keras.layers.Conv2DTranspose(8, 3, strides=2, padding="same",
                                         activation="relu"),
            keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid"),
        ])
        model = VQVAERotationTrick(
            num_embeddings=8, embedding_dim=4,
            encoder=encoder, decoder=decoder)
        model.compile(optimizer=keras.optimizers.SGD(0.0))
        return model

    @pytest.fixture
    def images(self):
        return np.random.default_rng(0).uniform(
            size=(4, 8, 8, 1)).astype("float32")

    def test_an_encoder_regularizer_changes_the_reported_loss(self, images):
        plain = self._build(None)
        regularized = self._build(keras.regularizers.l2(1e-1))
        loss_plain = float(ops.convert_to_numpy(
            plain.train_step(ops.convert_to_tensor(images))["loss"]))
        loss_reg = float(ops.convert_to_numpy(
            regularized.train_step(ops.convert_to_tensor(images))["loss"]))
        assert loss_reg > loss_plain + 1e-4, (
            f"an l2(1e-1) encoder regularizer moved the objective by "
            f"{loss_reg - loss_plain:.3e} -- it is not in the objective"
        )

    def test_test_step_reports_the_same_objective_as_train_step(self, images):
        model = self._build(keras.regularizers.l2(1e-1))
        train_loss = float(ops.convert_to_numpy(
            model.train_step(ops.convert_to_tensor(images))["loss"]))
        for tracker in model.metrics:
            tracker.reset_state()
        test_loss = float(ops.convert_to_numpy(
            model.test_step(ops.convert_to_tensor(images))["loss"]))
        assert abs(train_loss - test_loss) < 1e-4, (
            f"train_step reported {train_loss:.6f} but test_step "
            f"{test_loss:.6f} on identical weights and data"
        )

    def test_the_quantizer_losses_are_still_included(self, images):
        """Control: widening to `self.losses` must not LOSE the VQ terms."""
        model = self._build(None)
        _ = model(images, training=True)
        assert len(model.quantizer.losses) > 0
        quantizer_sum = float(ops.convert_to_numpy(
            ops.sum(ops.stack(model.quantizer.losses))))
        model_sum = float(ops.convert_to_numpy(
            ops.sum(ops.stack(model.losses))))
        assert model_sum == pytest.approx(quantizer_sum, abs=1e-6)


# ---------------------------------------------------------------------
# Gradient flow (plan-2026-08-19-a616f581 step 11)
# ---------------------------------------------------------------------
#
# MEASURED 2026-08-19: 35 trainable weights, 0 dead under the model's REAL
# objective. One detail is load-bearing and is spelled out because getting it
# wrong produces a convincing false finding:
#
#   loss_fn = default_loss (mean-of-squares over the OUTPUT only)
#       -> `vector_quantizer_rotation_trick/embeddings` receives NO gradient.
#   loss_fn = train_step's objective (recon + sum(self.losses))
#       -> 0 of 35 dead.
#
# The codebook is not trained by the reconstruction path at all; it is trained
# by the codebook/commitment terms the quantizer registers with `add_loss`, and
# `VQVAERotationTrick.train_step` optimizes `recon + sum(self.losses)` (see the
# `# DECISION plan-2026-08-18T140459-7991552f/D-026` anchor there, which is
# about summing `self.losses` rather than `self.quantizer.losses`). So a
# gradient-flow test that used the oracle's default loss here would report a
# dead codebook in a perfectly healthy model. The rule the oracle's docstring
# states -- pass `loss_fn=` when the model has a real objective -- is not
# optional for a model whose objective lives in `add_loss`.

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight


class TestVQVAERotationGradientFlow:
    """Every trainable weight is on the backward graph of the REAL objective."""

    def _model(self, gradient_mode="rotation"):
        return VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16, input_shape=(32, 32, 3),
            hidden_channels=32, downsample_factor=4, num_res_blocks=1,
            norm_type="layer_norm", gradient_mode=gradient_mode,
        )

    @pytest.mark.parametrize("gradient_mode", ["rotation", "ste"])
    def test_gradients_reach_every_trainable_weight(self, sample_images, gradient_mode):
        model = self._model(gradient_mode)
        model(sample_images, training=False)

        def train_step_objective(outputs):
            recon = ops.mean(ops.square(sample_images - outputs))
            recon = model.reconstruction_loss_weight * recon
            aux = model.losses
            return recon + (ops.sum(ops.stack(aux)) if aux else 0.0)

        report = assert_gradients_reach_every_trainable_weight(
            model, sample_images, loss_fn=train_step_objective
        )

        assert len(report) == len(model.trainable_weights)
        assert len(report) > 0
        codebook = [p for p in report if "embeddings" in p]
        assert codebook, "no codebook weight found -- has the quantizer changed?"
        assert all(report[p] is not None and report[p] > 0.0 for p in codebook)

    def test_the_codebook_needs_the_add_loss_terms(self, sample_images):
        """The comment above, made falsifiable.

        If the codebook ever became reachable from the reconstruction path
        alone, this test fails and the `loss_fn=` above is no longer required
        -- which is a change worth being told about, not one to discover by
        reading. It compares the SAME instance under the two objectives.
        """
        from ..gradient_flow_oracle import gradient_report

        model = self._model()
        model(sample_images, training=False)

        output_only = gradient_report(model, sample_images)
        codebook = [p for p in output_only if "embeddings" in p]
        assert codebook
        assert all(
            output_only[p] is None or output_only[p] == 0.0 for p in codebook
        ), (
            "the codebook IS reachable from the reconstruction output now; the "
            "loss_fn= in the test above is no longer load-bearing"
        )


# ---------------------------------------------------------------------
# The BACKWARD pass across `gradient_mode` (plan-2026-08-19-a616f581 step 13)
# ---------------------------------------------------------------------
#
# DECISION plan-2026-08-19-a616f581/D-019: these four tests assert the
# PROPERTIES the rotation trick claims -- direction transport and magnitude
# transport -- and NOT a frozen numeric delta. Do NOT replace them with a
# hand-derived closed-form gradient oracle, and do NOT reduce them to a single
# "the four gradients differ by more than X" number.
#
# Why the shape. MEASURED at HEAD, CPU: replacing the whole body of
# `VectorQuantizerRotationTrick._apply_gradient_transform` with the `ste`
# branch -- i.e. deleting the package's entire reason to exist -- left
# `tests/test_models/test_vq_vae_rotation/` at 43 passed / 43 collected and
# `tests/test_layers/test_vector_quantizer_rotation_trick.py` at 60 passed / 60
# collected. 103 green tests, zero of them touching the backward pass. The
# forward pass CANNOT close this gap: every mode emits the codebook vector `q`
# by construction (that is what `TestRotationTrickForwardIdentity` pins), so the
# modes are distinguishable ONLY through their Jacobians.
#
# Why a property and not a delta. A hand-derived closed-form gradient would have
# to be transcribed from `_apply_gradient_transform` itself -- the
# self-referential-oracle trap this repo has hit repeatedly -- and a frozen
# numeric delta would pin one seed's arithmetic rather than the algorithm. The
# rotation trick's claim is geometric and exactly checkable:
#
#     out = scale * R @ x,  R orthogonal with R @ unit(x) = unit(q),
#     scale = ||q|| / ||x||   (detached, except in 'no_grad_scale')
#
# so for an upstream gradient aligned with `unit(q)`, the gradient w.r.t. the
# encoder output is `R^T unit(q) * scale = scale * unit(x)`: it comes back
# pointing along the ENCODER's own direction, rescaled by the codebook/encoder
# magnitude ratio. Straight-through returns the upstream vector untouched --
# along `unit(q)`, ratio exactly 1. Both are exact identities, verified below to
# 1e-5 relative, and both are read off the paper's definition rather than off
# the implementation.
#
# Deliberate scope note: the `unit(q)` probe does NOT separate 'rotation' from
# 'reflection' -- the negated Householder form maps `unit(q) -> unit(x)` too
# (measured: cos = 1.0 for both). That separation is the job of
# `test_the_four_modes_disagree_on_the_backward_pass_alone`, which uses the
# model's real objective and a general upstream gradient. See decisions.md
# D-019.

import tensorflow as tf

from dl_techniques.layers.vector_quantizer_rotation_trick import (
    VectorQuantizerRotationTrick,
)


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


class TestGradientModesDisagreeOnTheBackwardPass:
    """`gradient_mode` is a backward-pass knob; something must observe it."""

    # The modes are ENUMERATED from the layer, never hard-coded here: a fifth
    # mode must not be able to ship untested.
    MODES = VectorQuantizerRotationTrick._GRAD_MODES

    @staticmethod
    def _model() -> VQVAERotationTrick:
        keras.utils.set_random_seed(7)
        return VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16, input_shape=(16, 16, 3),
            hidden_channels=16, downsample_factor=4, num_res_blocks=1,
            norm_type="layer_norm", gradient_mode="rotation",
        )

    @staticmethod
    def _images() -> np.ndarray:
        # numpy, not `keras.random.*`: a `keras.random` draw after
        # `set_random_seed` is NOT reproducible across constructions in one
        # process (it advances a global counter), which silently gave each mode
        # a DIFFERENT input while this test was being written.
        return np.random.default_rng(0).uniform(
            0.0, 1.0, (4, 16, 16, 3)).astype("float32")

    def test_the_four_modes_disagree_on_the_backward_pass_alone(self):
        """One instance, one input, four modes: same loss, four gradients.

        The objective is `train_step`'s own (`recon + sum(self.losses)`), not a
        generic one -- this model's codebook lives entirely in `add_loss`.
        `gradient_mode` is flipped on a SINGLE built instance so that
        "different weights" cannot explain any difference.
        """
        model = self._model()
        images = self._images()
        model(images, training=False)
        z = tf.constant(np.asarray(model.encode(images)))

        grads, losses, outputs = {}, {}, {}
        for mode in self.MODES:
            model.quantizer.gradient_mode = mode
            with tf.GradientTape() as tape:
                tape.watch(z)
                quantized = model.quantizer(z, training=True)
                recon = model.decoder(quantized)
                loss = model.reconstruction_loss_weight * ops.mean(
                    ops.square(images - recon))
                aux = model.losses
                loss = loss + (ops.sum(ops.stack(aux)) if aux else 0.0)
            grads[mode] = np.asarray(tape.gradient(loss, z))
            losses[mode] = float(loss)
            outputs[mode] = np.asarray(quantized)

        # Control: the FORWARD pass is mode-invariant, so any difference below
        # is purely a backward-pass difference. Both are asserted because the
        # loss alone could coincide by luck; the tensors cannot.
        first = self.MODES[0]
        for mode in self.MODES[1:]:
            np.testing.assert_allclose(
                outputs[mode], outputs[first], atol=1e-5,
                err_msg=f"forward pass is NOT mode-invariant: {mode} vs {first}")
            assert abs(losses[mode] - losses[first]) < 1e-5, (
                f"objective moved with gradient_mode ({mode}); the control that "
                "isolates this test to the backward pass is broken")

        # Every pair must disagree. Relative separation, never an absolute
        # bound: the gradient scale here is ~1e-2 and is device-dependent.
        # MEASURED on CPU: the closest pair (rotation vs no_grad_scale, which
        # differ only in whether `scale`'s gradient flows) is 4.7e-2; the
        # widest is 1.02. The float32 floor for two identical paths is ~1e-7,
        # so 1e-3 sits between the two by orders of magnitude.
        seen = []
        for i, a in enumerate(self.MODES):
            for b in self.MODES[i + 1:]:
                denom = max(np.linalg.norm(grads[a]), np.linalg.norm(grads[b]))
                assert denom > 0.0, f"{a}/{b}: both gradients are exactly zero"
                rel = np.linalg.norm(grads[a] - grads[b]) / denom
                seen.append((a, b, rel))
                assert rel > 1e-3, (
                    f"gradient_mode={a!r} and {b!r} produce the SAME gradient "
                    f"w.r.t. the encoder output (rel={rel:.3e}). The modes are "
                    "a backward-pass-only knob, so this means one of them has "
                    "collapsed onto the other.")
        assert len(seen) == len(self.MODES) * (len(self.MODES) - 1) // 2

    def test_rotation_is_not_the_straight_through_copy(self):
        """The `ste`-collapse signature, asserted by name.

        Straight-through hands the upstream gradient back untouched. The
        rotation trick multiplies it by `scale * R^T`. Both are checked on the
        same instance and the same input.
        """
        model = self._model()
        images = self._images()
        model(images, training=False)
        z = tf.constant(np.asarray(model.encode(images)))

        def grad_for(mode):
            model.quantizer.gradient_mode = mode
            with tf.GradientTape() as tape:
                tape.watch(z)
                quantized = model.quantizer(z, training=True)
                recon = model.decoder(quantized)
                loss = model.reconstruction_loss_weight * ops.mean(
                    ops.square(images - recon))
                aux = model.losses
                loss = loss + (ops.sum(ops.stack(aux)) if aux else 0.0)
            return np.asarray(tape.gradient(loss, z))

        g_rot, g_ste = grad_for("rotation"), grad_for("ste")
        denom = max(np.linalg.norm(g_rot), np.linalg.norm(g_ste))
        assert denom > 0.0
        rel = np.linalg.norm(g_rot - g_ste) / denom
        assert rel > 1e-3, (
            f"gradient_mode='rotation' returned the straight-through gradient "
            f"(rel={rel:.3e}); `_apply_gradient_transform` has collapsed onto "
            "its `ste` branch and the package's entire premise is gone.")

    @pytest.mark.parametrize("mode", ["rotation", "reflection", "ste"])
    def test_the_gradient_is_transported_the_way_the_mode_claims(self, mode):
        """Direction AND magnitude transport, as exact identities.

        Upstream gradient := `unit(q)`, the codebook direction.
          rotation/reflection: comes back along `unit(x)`, scaled by
                               ||q||/||x||  (R is orthogonal).
          ste:                 comes back unchanged -- along `unit(q)`, ratio 1.
        """
        model = self._model()
        model.quantizer.gradient_mode = mode
        rows = np.random.default_rng(1).normal(
            size=(6, model.embedding_dim)).astype("float32")
        x = tf.constant(rows)

        q = np.asarray(model.quantizer(x, training=False))
        unit_x, unit_q = _unit(rows), _unit(q)
        upstream = tf.constant(unit_q)

        with tf.GradientTape() as tape:
            tape.watch(x)
            out = model.quantizer(x, training=False)
            projected = ops.sum(out * upstream)
        g = np.asarray(tape.gradient(projected, x))

        # Anti-vacuity: the two directions must actually differ, and the
        # magnitude ratio must be far from 1 -- otherwise "along unit(x),
        # scaled by ||q||/||x||" and "unchanged" would be the same claim.
        cos_xq = np.sum(unit_x * unit_q, axis=-1)
        scale = (np.linalg.norm(q, axis=-1) / np.linalg.norm(rows, axis=-1))
        assert cos_xq.max() < 0.99, (
            f"vacuous probe: unit(x) and unit(q) are nearly parallel "
            f"(max cos={cos_xq.max():.4f})")
        assert scale.max() < 0.5, (
            f"vacuous probe: ||q||/||x|| is close to 1 (max={scale.max():.4f})")

        ratio = np.linalg.norm(g, axis=-1) / np.linalg.norm(unit_q, axis=-1)
        cos_g_x = np.sum(_unit(g) * unit_x, axis=-1)
        cos_g_q = np.sum(_unit(g) * unit_q, axis=-1)

        if mode == "ste":
            np.testing.assert_allclose(cos_g_q, 1.0, atol=1e-5, err_msg=(
                "straight-through must return the upstream gradient untouched"))
            np.testing.assert_allclose(ratio, 1.0, rtol=1e-5, err_msg=(
                "straight-through must not rescale the upstream gradient"))
        else:
            np.testing.assert_allclose(cos_g_x, 1.0, atol=1e-5, err_msg=(
                f"gradient_mode={mode!r}: the gradient is NOT transported back "
                "onto the encoder's own direction; R^T unit(q) != unit(x), so "
                "the transform is not the claimed rotation about (u+v)"))
            np.testing.assert_allclose(ratio, scale, rtol=1e-5, err_msg=(
                f"gradient_mode={mode!r}: |grad| != (||q||/||x||)*|upstream|, "
                "so the transform is not orthogonal-times-scale -- the "
                "magnitude transport the trick exists to provide is broken"))

    def test_no_grad_scale_lets_the_scale_gradient_cancel_the_transport(self):
        """`no_grad_scale` is not a cosmetic alias of `rotation`.

        With `scale = ||q||/||x||` left differentiable, the scale term's
        gradient exactly opposes the rotated term's along `unit(q)`:
        `d/dx [scale * (R x . unit(q))] = d/dx [ ||q|| ] = 0`. So the SAME
        probe that gives `rotation` a ratio of `||q||/||x||` gives
        `no_grad_scale` exactly 0. Under an `ste` collapse it is 1.0.
        """
        model = self._model()
        model.quantizer.gradient_mode = "no_grad_scale"
        rows = np.random.default_rng(1).normal(
            size=(6, model.embedding_dim)).astype("float32")
        x = tf.constant(rows)
        q = np.asarray(model.quantizer(x, training=False))
        upstream = tf.constant(_unit(q))

        with tf.GradientTape() as tape:
            tape.watch(x)
            out = model.quantizer(x, training=False)
            projected = ops.sum(out * upstream)
        g = np.asarray(tape.gradient(projected, x))

        ratio = np.linalg.norm(g, axis=-1)
        assert ratio.max() < 1e-5, (
            "gradient_mode='no_grad_scale' no longer lets the scale gradient "
            f"flow (max |grad| = {ratio.max():.3e}, expected ~0); it has "
            "collapsed onto 'rotation' (ratio ||q||/||x||) or onto 'ste' "
            "(ratio 1.0).")
