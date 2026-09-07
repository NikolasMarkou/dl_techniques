"""Tests for the SparseAutoencoder layer."""

import os
import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.generative.sparse_autoencoder import SparseAutoencoder

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


# ----------------------------------------------------------------------
# Per-site initializer cloning (plan-2026-09-07T161712-985e4d31/D-002)
# ----------------------------------------------------------------------

class TestInitializerAliasing:
    """``build()`` must draw each weight from its own initializer instance.

    Scope of the claim, stated exactly: per-site cloning gives independent
    draws for a RANDOM SEEDLESS initializer. There are three exemptions, all of
    them correct behaviour rather than defects:

    1. a caller-supplied SEEDED instance (e.g. ``GlorotUniform(seed=7)``) --
       ``clone_initializer`` reproduces an explicit seed deliberately and by
       contract (``src/dl_techniques/initializers/clone.py``), so the clones
       stay tied (``test_a_seeded_initializer_still_aliases_by_contract``);
    2. a DETERMINISTIC initializer (``'zeros'``, ``'ones'``, ``Constant``, and
       ``Identity`` where the weight is 2-D) -- it holds no random state, so
       every site is bit-identical and that is exactly what it is meant to do
       (``test_zeros_bias_stays_identical_positive_control`` on the bias side,
       ``test_a_deterministic_kernel_is_identical_at_every_site`` on the kernel
       side);
    3. a CUSTOM initializer whose ``get_config()``/``from_config()`` round trip
       raises -- ``clone_initializer`` then falls back to ``copy.deepcopy``,
       which copies the already-resolved seed rather than drawing a new one,
       so such a site can silently stay tied. Untested here because it is a
       property of ``clone_initializer`` itself, not of this layer; it is
       named so a caller passing a custom initializer is not misled by the
       sentence above.

    Callers who want reproducibility WITHOUT the tie should seed the process
    with ``keras.utils.set_random_seed()`` and leave the initializer seedless.

    What is NOT the criterion: matching SHAPES. A shared seedless instance
    replays one underlying sample at every site, so a shorter draw is a longer
    draw's prefix up to the fan-based scale -- measured elsewhere in this repo
    at Pearson r = 0.999999999999998 across a rank-5/rank-4 mismatch. Every
    ``add_weight`` fed by a shared instance is cloned here, not just the
    shape-coinciding ones.
    """

    @staticmethod
    def _built(**kwargs):
        layer = SparseAutoencoder(**kwargs)
        layer.build((None, kwargs["d_input"]))
        return layer

    @staticmethod
    def _np(w):
        return keras.ops.convert_to_numpy(w)

    def test_encoder_and_gate_kernels_are_independent_draws(self):
        """Non-square gated SAE: ``encoder_weight`` vs ``gate_weight``.

        Their shapes coincide unconditionally (both ``(d_input, d_latent)``), so
        one shared seedless instance replays the same draw at both sites. Note
        the direction: shape coincidence is what makes the replay BIT-identical
        and therefore visible to ``array_equal``. It is NOT what causes the two
        sites to share a sample; they would share one at any shapes (class
        docstring). Exact for a random seedless initializer; see the class
        docstring for the three exemptions.
        """
        keras.utils.set_random_seed(1234)
        layer = self._built(d_input=64, d_latent=512, variant="gated")
        enc, gate = self._np(layer.encoder_weight), self._np(layer.gate_weight)
        # Anti-vacuity: the arm is only meaningful while the shapes coincide.
        assert enc.shape == gate.shape == (64, 512)
        assert not np.array_equal(enc, gate), (
            "encoder_weight and gate_weight are bit-identical: one seedless "
            "initializer instance was replayed at both add_weight sites. "
            "Clone per site (initializers/clone.py). Note a SEEDED initializer "
            "is a documented exemption and would legitimately be identical."
        )

    def test_square_sae_draws_all_three_kernels_independently(self):
        """Square SAE (``d_input == d_latent``): all three kernels are ``(d, d)``.

        The only configuration in which ``decoder_weight`` joins the
        coincidence. Exact for a seedless initializer; a seeded one is exempt.
        """
        keras.utils.set_random_seed(1234)
        layer = self._built(
            d_input=64, d_latent=64, variant="gated", tied_weights=False
        )
        enc = self._np(layer.encoder_weight)
        dec = self._np(layer.decoder_weight)
        gate = self._np(layer.gate_weight)
        # Anti-vacuity: all three must actually share a shape here.
        assert enc.shape == dec.shape == gate.shape == (64, 64)
        assert not np.array_equal(enc, dec), "encoder_weight == decoder_weight"
        assert not np.array_equal(enc, gate), "encoder_weight == gate_weight"
        assert not np.array_equal(dec, gate), "decoder_weight == gate_weight"

    def test_zeros_bias_stays_identical_positive_control(self):
        """Positive control: cloning a DETERMINISTIC initializer is a no-op.

        At the shipped default ``bias_initializer='zeros'`` every bias is all
        zeros and therefore identical to every other. That is CORRECT, not a
        symmetry defect -- this arm exists so the diversity arms above cannot be
        read as "every weight always differs".
        """
        keras.utils.set_random_seed(1234)
        layer = self._built(d_input=64, d_latent=64, variant="gated")
        enc_b = self._np(layer.encoder_bias)
        dec_b = self._np(layer.decoder_bias)
        gate_b = self._np(layer.gate_bias)
        assert enc_b.shape == dec_b.shape == gate_b.shape == (64,)
        assert np.all(enc_b == 0.0)
        assert np.array_equal(enc_b, gate_b)
        assert np.array_equal(enc_b, dec_b)

    def test_random_bias_initializer_draws_per_site(self):
        """The bias fan-out becomes live under a randomized ``bias_initializer``.

        Inert at the ``'zeros'`` default, so this is the arm that can see the
        bias-site clones at all. Exact for a seedless initializer; a seeded one
        is exempt by contract.
        """
        keras.utils.set_random_seed(1234)
        layer = self._built(
            d_input=64,
            d_latent=512,
            variant="gated",
            bias_initializer=keras.initializers.RandomNormal(stddev=0.5),
        )
        enc_b, gate_b = self._np(layer.encoder_bias), self._np(layer.gate_bias)
        assert enc_b.shape == gate_b.shape == (512,)
        assert not np.array_equal(enc_b, gate_b), (
            "encoder_bias and gate_bias are bit-identical under a randomized "
            "bias_initializer: the shared seedless instance was replayed."
        )

    @pytest.mark.parametrize(
        "init", ["zeros", "ones", keras.initializers.Constant(0.3)]
    )
    def test_a_deterministic_kernel_is_identical_at_every_site(self, init):
        """Exemption 2 on the KERNEL side: identical weights are CORRECT here.

        ``'zeros'``/``'ones'``/``Constant`` hold no per-instance random state,
        so cloning them is a no-op and every kernel site comes out
        bit-identical. The bias side already had this positive control
        (``test_zeros_bias_stays_identical_positive_control``); the kernel side
        had none, so the class docstring's central claim could be
        "strengthened" to the false form "cloning always makes weights differ"
        without anything going RED.
        """
        keras.utils.set_random_seed(1234)
        layer = self._built(
            d_input=64, d_latent=64, variant="gated", kernel_initializer=init
        )
        enc = self._np(layer.encoder_weight)
        gate = self._np(layer.gate_weight)
        dec = self._np(layer.decoder_weight)
        assert enc.shape == gate.shape == dec.shape == (64, 64)
        assert np.array_equal(enc, gate)
        assert np.array_equal(enc, dec), (
            "a deterministic initializer must produce the same values at every "
            "site; cloning it is a no-op by construction"
        )

    def test_a_seeded_initializer_still_aliases_by_contract(self):
        """Documented limitation, pinned: a SEEDED initializer stays aliased.

        ``clone_initializer`` reproduces an explicit seed on purpose
        (``initializers/clone.py:60-65``), so per-site cloning does NOT break
        symmetry here. A RED on this arm means that contract changed and every
        comment, docstring and sibling assertion written around it is now wrong.
        The reproducibility idiom that does NOT alias is
        ``keras.utils.set_random_seed()`` with a seedless initializer.
        """
        keras.utils.set_random_seed(1234)
        layer = self._built(
            d_input=64,
            d_latent=512,
            variant="gated",
            kernel_initializer=keras.initializers.GlorotUniform(seed=7),
        )
        enc, gate = self._np(layer.encoder_weight), self._np(layer.gate_weight)
        assert enc.shape == gate.shape == (64, 512)
        assert np.array_equal(enc, gate), (
            "a seeded initializer no longer replays across add_weight sites: "
            "clone_initializer's seeded contract changed"
        )

    def test_cloning_leaves_the_serialized_initializers_untouched(self):
        """The clone belongs at the ``add_weight`` site, never on the attribute.

        ``self.kernel_initializer``/``self.bias_initializer`` must remain the
        caller's own objects so ``get_config()`` still serializes what was
        passed in.
        """
        kern = keras.initializers.GlorotUniform(seed=7)
        bias = keras.initializers.RandomNormal(stddev=0.5)
        layer = self._built(
            d_input=64,
            d_latent=512,
            variant="gated",
            kernel_initializer=kern,
            bias_initializer=bias,
        )
        assert layer.kernel_initializer is kern
        assert layer.bias_initializer is bias
        cfg = layer.get_config()
        assert cfg["kernel_initializer"] == keras.initializers.serialize(kern)
        assert cfg["bias_initializer"] == keras.initializers.serialize(bias)
