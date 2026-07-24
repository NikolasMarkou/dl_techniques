"""Tests for the SparseAutoencoder layer."""

import os
import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.sparse_autoencoder import SparseAutoencoder

B, DIN, DLAT = 4, 10, 20

ALL_VARIANTS = [
    ("relu", {}),
    ("topk", {"k": 4}),
    ("batch_topk", {"k": 4}),
    ("jumprelu", {}),
    ("gated", {}),
]


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, DIN)).astype("float32")


class TestSparseAutoencoder:

    def test_construction(self):
        layer = SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant="topk", k=4)
        assert layer.d_latent == DLAT

    def test_invalid_variant(self):
        with pytest.raises(ValueError):
            SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant="bogus")

    def test_topk_requires_k(self):
        with pytest.raises(ValueError):
            SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant="topk", k=None)

    @pytest.mark.parametrize("variant,kw", [
        ("relu", {}),
        ("topk", {"k": 4}),
        ("batch_topk", {"k": 4}),
        ("jumprelu", {}),
        ("gated", {}),
    ])
    def test_forward_pass(self, sample, variant, kw):
        layer = SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant=variant, **kw)
        out = layer(sample)
        assert tuple(out.shape) == (B, DIN)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_return_latents(self, sample):
        layer = SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant="topk", k=4)
        recon, latents, loss = layer(sample, return_latents=True)
        assert tuple(recon.shape) == (B, DIN)
        assert tuple(latents.shape) == (B, DLAT)

    def test_compute_output_shape(self):
        layer = SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant="topk", k=4)
        assert layer.compute_output_shape((B, DIN)) == (B, DIN)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(DIN,))
        out = SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant="topk", k=4, name="sae")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample, training=False)
        path = os.path.join(tmp_path, "sae.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"SparseAutoencoder": SparseAutoencoder}
        )
        y1 = loaded(sample, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        layer = SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant="topk", k=4, tied_weights=True)
        rebuilt = SparseAutoencoder.from_config(layer.get_config())
        assert rebuilt.d_latent == DLAT and rebuilt.variant == "topk"

    # ------------------------------------------------------------------
    # Step-7 correctness-fix coverage (RED-proven gradient/determinism/state)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("variant,kw", ALL_VARIANTS)
    def test_gradient_flow(self, sample, variant, kw):
        """Every variant's encoder weight must receive a finite nonzero grad.

        For ``gated`` additionally isolate the paper L_aux path with
        ``l1_coefficient=0.0`` and assert ``gate_weight`` gets a finite NONZERO
        gradient (D-004 RED guard: 0.073 with L_aux vs 0.0 without).
        """
        layer = SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant=variant, **kw)
        x = tf.convert_to_tensor(sample)
        with tf.GradientTape() as tape:
            recon, latents, total = layer(x, training=True, return_latents=True)
            loss = tf.reduce_mean(tf.square(x - recon)) + total
        grads = tape.gradient(loss, layer.trainable_weights)

        enc_grad = None
        for w, g in zip(layer.trainable_weights, grads):
            if w is layer.encoder_weight:
                enc_grad = g
        assert enc_grad is not None, "encoder_weight received no gradient"
        enc_np = keras.ops.convert_to_numpy(enc_grad)
        assert np.all(np.isfinite(enc_np))
        assert np.linalg.norm(enc_np) > 0.0

        if variant == "gated":
            # L_aux isolation: l1=0 => sparsity_loss=0 => total is L_aux only.
            gated = SparseAutoencoder(
                d_input=DIN, d_latent=DLAT, variant="gated", l1_coefficient=0.0
            )
            with tf.GradientTape() as tape2:
                _, _, total2 = gated(x, training=True, return_latents=True)
                aux_only = total2
            g_gate = tape2.gradient(aux_only, gated.gate_weight)
            assert g_gate is not None, "gate_weight got no L_aux gradient"
            g_gate_np = keras.ops.convert_to_numpy(g_gate)
            assert np.all(np.isfinite(g_gate_np))
            assert np.linalg.norm(g_gate_np) > 0.0

    def test_batch_topk_inference_determinism(self):
        """A fixed example's inference output is invariant to its batch-mates.

        allclose (rtol/atol 1e-4), NOT bitwise: GPU matmul reduction order gives
        ~6e-8 FP noise but no mask flips (D-002).
        """
        rng = np.random.default_rng(1)
        d_lat = 32
        layer = SparseAutoencoder(d_input=DIN, d_latent=d_lat, variant="batch_topk", k=4)
        # Prime the EMA threshold with a full-batch training pass.
        prime = rng.standard_normal((8, DIN)).astype("float32")
        _ = layer(prime, training=True)

        x = rng.standard_normal((1, DIN)).astype("float32")
        distractors = rng.standard_normal((2, DIN)).astype("float32")

        out_alone = keras.ops.convert_to_numpy(layer(x, training=False))[0]
        batched = np.concatenate([x, distractors], axis=0)
        out_batched = keras.ops.convert_to_numpy(layer(batched, training=False))[0]

        np.testing.assert_allclose(out_alone, out_batched, rtol=1e-4, atol=1e-4)

    def test_dead_latent_counter(self):
        """dead_steps increments for never-firing latents and is 0 for firers."""
        layer = SparseAutoencoder(
            d_input=DIN, d_latent=8, variant="topk", k=1,
            aux_k=2, dead_steps_threshold=0,
        )
        rng = np.random.default_rng(2)
        for _ in range(3):
            x = rng.standard_normal((B, DIN)).astype("float32")
            layer(x, training=True, return_latents=True)
        dead = keras.ops.convert_to_numpy(layer.dead_steps)
        # k=1 over B=4 fires <=4 of 8 latents/step: some never fire (counter>0),
        # some fired in the last step (reset to 0).
        assert dead.max() > 0
        assert dead.min() == 0

    def test_aux_loss_targets_residual(self):
        """_compute_auxiliary_loss output depends on main_reconstruction.

        Same (pre_activation, latents, inputs), differing main_reconstruction
        (inputs => residual 0, zeros => residual == inputs) must yield different
        loss scalars, proving it targets the residual not the raw input (D-003).
        """
        layer = SparseAutoencoder(
            d_input=DIN, d_latent=8, variant="topk", k=1,
            aux_k=2, dead_steps_threshold=0,
        )
        rng = np.random.default_rng(3)
        x = tf.convert_to_tensor(rng.standard_normal((B, DIN)).astype("float32"))
        _ = layer(x, training=False)  # build weights

        pre = layer.encode(x, training=False)
        latents, _ = layer._apply_sparsity(pre, training=False)

        # training=False: no dead_steps mutation between the two calls.
        loss_res = layer._compute_auxiliary_loss(
            pre, latents, x, x, training=False
        )
        loss_zero = layer._compute_auxiliary_loss(
            pre, latents, x, keras.ops.zeros_like(x), training=False
        )
        a = float(keras.ops.convert_to_numpy(loss_res))
        b = float(keras.ops.convert_to_numpy(loss_zero))
        assert not np.isclose(a, b)

    @pytest.mark.parametrize("variant,kw", ALL_VARIANTS)
    def test_return_latents_all_variants(self, sample, variant, kw):
        layer = SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant=variant, **kw)
        recon, latents, loss = layer(sample, return_latents=True)
        assert tuple(recon.shape) == (B, DIN)
        assert tuple(latents.shape) == (B, DLAT)
        loss_np = keras.ops.convert_to_numpy(loss)
        assert loss_np.ndim == 0
        assert np.isfinite(loss_np).all()

    @pytest.mark.parametrize("variant,kw", [
        ("batch_topk", {"k": 4}),
        ("gated", {}),
    ])
    def test_state_serialization(self, sample, tmp_path, variant, kw):
        """Save/load forward-value equality after a state-populating pass (A1).

        For ``batch_topk`` this proves the new non-trainable ``batch_topk_threshold``
        EMA state round-trips. For ``gated`` this is a generic gated-variant
        round-trip: gated has NO ``batch_topk_threshold`` and NO ``dead_steps``
        (it uses the frozen-decoder ``L_aux`` path, not the AuxK dead-latent
        path), so this case proves that path plus its trainable weights (gate +
        magnitude encoders, decoder) serialize and reload byte-faithfully — it is
        NOT a new-state-var persistence check.
        """
        inp = keras.Input(shape=(DIN,))
        out = SparseAutoencoder(
            d_input=DIN, d_latent=DLAT, variant=variant, name="sae", **kw
        )(inp)
        model = keras.Model(inp, out)
        # Populate state (EMA threshold / dead_steps counter).
        _ = model(sample, training=True)
        y0 = model(sample, training=False)

        path = os.path.join(tmp_path, "sae.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"SparseAutoencoder": SparseAutoencoder}
        )
        y1 = loaded(sample, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_input_validation(self):
        with pytest.raises(ValueError):
            SparseAutoencoder(d_input=10, d_latent=20, aux_k=0)
        with pytest.raises(ValueError):
            SparseAutoencoder(d_input=0, d_latent=20)
        with pytest.raises(ValueError):
            SparseAutoencoder(d_input=10, d_latent=-5)

        # aux_k > d_latent must NOT raise: it is gracefully clamped (D-006).
        layer = SparseAutoencoder(
            d_input=10, d_latent=20, aux_k=256, variant="topk", k=4
        )
        x = np.random.default_rng(4).standard_normal((3, 10)).astype("float32")
        out = layer(x)
        assert tuple(out.shape) == (3, 10)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))
