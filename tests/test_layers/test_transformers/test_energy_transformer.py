"""
Test suite for layers/transformers/energy_transformer.py.

Currently covers :class:`HopfieldNetwork`, the Energy Transformer's associative-memory
module (arXiv:2302.07253 eq. 5, 9): ONE tied ``(K, D)`` matrix, no bias, strictly per
token. It defines a scalar energy ``E_HN(g)`` whose ``update()`` returns the hand-coded
closed-form ``-dE_HN/dg``.

(The ``EnergyTransformer`` block lands in this same file in a follow-up step — hence the
per-class test grouping.)

``TestHopfieldNetwork.test_gradient_oracle`` is THE headline test for this module (plan
plan_2026-07-13_57c9833e, success criterion S6b). Each activation carries BOTH an energy
and a gradient, and they must be a matched pair: pairing the relu energy with the softmax
gradient (or vice versa) yields a layer that runs, trains, emits finite output — and does
not descend. The oracle is parametrized over both activations precisely to catch that.
``tf.GradientTape`` is the oracle and is legitimate HERE — it is forbidden only in ``src/``
(decisions.md D-001).

Run:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \\
        tests/test_layers/test_transformers/test_energy_transformer.py -q
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

# LESSONS [I:4]: TF32 on Ampere+ truncates matmul mantissas and makes an einsum
# equivalence check fail spuriously at ~5e-4 against an fp32 reference. Without this, the
# gradient oracle below is a coin-flip that WILL be misread as "the gradient is wrong".
tf.config.experimental.enable_tensor_float_32_execution(False)

from dl_techniques.layers.transformers.energy_transformer import HopfieldNetwork


# ---------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------

BATCH, TOKENS, DIM, MEM = 3, 7, 32, 16
ACTIVATIONS = ["relu", "softmax"]


@pytest.fixture
def input_shape():
    """(batch, tokens, embed_dim)."""
    return (BATCH, TOKENS, DIM)


@pytest.fixture
def sample_input(input_shape):
    rng = np.random.default_rng(1234)
    return rng.standard_normal(size=input_shape).astype("float32")


@pytest.fixture
def layer():
    return HopfieldNetwork(dim=DIM, hopfield_dim=MEM)


def _excite(hopfield: HopfieldNetwork, scale: float = 0.5, seed: int = 7) -> None:
    """Replace the N(0, 0.02) init with visibly-sized memories.

    With the paper's default init `h` is O(1e-2), `relu(h)` kills half of it, and the
    resulting update is O(1e-3) — small enough that a WRONG gradient could still match a
    correct one to within `atol`. The oracle asserts a non-trivial magnitude, so the
    memories must actually excite the layer.
    """
    rng = np.random.default_rng(seed)
    shape = tuple(hopfield.xi.shape)
    hopfield.xi.assign((rng.standard_normal(shape) * scale).astype("float32"))


# ---------------------------------------------------------------------
# HopfieldNetwork
# ---------------------------------------------------------------------

class TestHopfieldNetwork:
    """S6b (the gradient oracle), invariant 3 (no token mixing), + the layer contract."""

    # -- S6b: THE GRADIENT ORACLE ------------------------------------

    @pytest.mark.parametrize("activation", ACTIVATIONS)
    def test_gradient_oracle(self, activation, sample_input):
        """update(g) MUST equal -d/dg [ sum_b energy(g) ], for EVERY activation (S6b).

        This is what proves the energy/gradient PAIR is consistent. A crossed pair (relu
        energy + softmax r, or vice versa) runs, trains, and does not descend.
        """
        hopfield = HopfieldNetwork(
            dim=DIM, hopfield_dim=MEM, activation=activation, hopfield_beta=0.7,
        )
        hopfield.build((BATCH, TOKENS, DIM))
        _excite(hopfield)

        # Larger-variance g so the relu branch is not sitting on a dead half-plane.
        g = tf.Variable(sample_input * 2.0, dtype=tf.float32)

        with tf.GradientTape() as tape:
            energy = tf.reduce_sum(hopfield.energy(g))
        grad = tape.gradient(energy, g)

        assert grad is not None, "energy() has no gradient path to g"

        update = keras.ops.convert_to_numpy(hopfield.update(g))
        pos_grad = grad.numpy()  # tape.gradient(E, g) == +dE/dg; update must be its negation

        max_abs_update = float(np.abs(update).max())

        # Anti-vacuity: a zero-vs-zero comparison must not be able to pass.
        assert max_abs_update > 1e-3, (
            f"update is ~0 (max |update| = {max_abs_update:.3e}) — the oracle would pass "
            "vacuously. Excite the memories."
        )
        assert np.all(np.isfinite(update))
        assert np.all(np.isfinite(pos_grad))

        max_abs_err = float(np.abs(update - (-pos_grad)).max())
        print(
            f"\n[S6b oracle] activation={activation}: "
            f"max|update|={max_abs_update:.6e}  max-abs-error={max_abs_err:.6e}"
        )
        np.testing.assert_allclose(
            update, -pos_grad, rtol=1e-4, atol=1e-5,
            err_msg=(
                f"update() != -dE/dg (activation={activation}); "
                f"max-abs-error={max_abs_err:.3e}. The closed-form gradient in "
                "HopfieldNetwork.update() is WRONG (a crossed energy/gradient pair, or a "
                "softmax over the TOKEN axis instead of the MEMORY axis) — do NOT 'fix' "
                "energy() to match it."
            ),
        )

    # -- invariant 3: strictly per-token, no token mixing -------------

    @pytest.mark.parametrize("activation", ACTIVATIONS)
    def test_no_token_mixing(self, activation, sample_input):
        """Permuting the TOKEN axis of g permutes the output rows identically.

        A softmax taken over the token axis n (instead of the MEMORY axis k) would break
        this — and would also stop being the gradient of the reported energy.
        """
        hopfield = HopfieldNetwork(dim=DIM, hopfield_dim=MEM, activation=activation)
        hopfield.build((BATCH, TOKENS, DIM))
        _excite(hopfield)

        perm = np.random.default_rng(5).permutation(TOKENS)

        u_plain = keras.ops.convert_to_numpy(hopfield.update(sample_input))
        u_perm = keras.ops.convert_to_numpy(hopfield.update(sample_input[:, perm, :]))

        np.testing.assert_allclose(
            u_perm, u_plain[:, perm, :], atol=1e-6,
            err_msg=(
                "HopfieldNetwork mixed tokens — update(permute(g)) != permute(update(g)). "
                "Is the softmax over the TOKEN axis instead of the MEMORY axis?"
            ),
        )

    @pytest.mark.parametrize("activation", ACTIVATIONS)
    def test_perturbing_one_token_leaves_the_others_alone(self, activation, sample_input):
        """Perturbing g[:, j, :] changes update[:, j, :] and NOTHING else (invariant 3)."""
        j = 2
        hopfield = HopfieldNetwork(dim=DIM, hopfield_dim=MEM, activation=activation)
        hopfield.build((BATCH, TOKENS, DIM))
        _excite(hopfield)

        x0 = sample_input.copy()
        x1 = sample_input.copy()
        rng = np.random.default_rng(99)
        x1[:, j, :] += rng.standard_normal((BATCH, DIM)).astype("float32") * 3.0

        u0 = keras.ops.convert_to_numpy(hopfield.update(x0))
        u1 = keras.ops.convert_to_numpy(hopfield.update(x1))

        others = [i for i in range(TOKENS) if i != j]
        np.testing.assert_allclose(
            u0[:, others, :], u1[:, others, :], atol=1e-6,
            err_msg="perturbing token j changed another token's update — token mixing",
        )

        # Anti-vacuity: the perturbation is real and DID move row j.
        assert np.abs(u1[:, j, :] - u0[:, j, :]).max() > 1e-3, (
            "the perturbation changed nothing at all — the comparison above would pass "
            "vacuously"
        )

    # -- energy sign --------------------------------------------------

    def test_relu_energy_is_nonpositive(self, sample_input):
        """E_HN = -0.5 * sum relu(h)^2 <= 0 for the relu branch."""
        hopfield = HopfieldNetwork(dim=DIM, hopfield_dim=MEM, activation="relu")
        hopfield.build((BATCH, TOKENS, DIM))
        _excite(hopfield)

        e = keras.ops.convert_to_numpy(hopfield.energy(sample_input))
        assert np.all(np.isfinite(e))
        assert np.all(e <= 0.0), f"relu energy must be non-positive, got {e}"
        # Non-degenerate: it is not identically zero (a dead relu would pass vacuously).
        assert np.abs(e).max() > 1e-3

    # -- standard layer contract --------------------------------------

    def test_instantiation(self):
        default = HopfieldNetwork(dim=64, hopfield_dim=256)
        assert default.dim == 64
        assert default.hopfield_dim == 256
        assert default.activation == "relu"
        assert default.hopfield_beta == 1.0
        assert isinstance(
            default.kernel_initializer, keras.initializers.TruncatedNormal
        )
        assert default.xi is None
        assert not default.built

        custom = HopfieldNetwork(
            dim=48, hopfield_dim=12, activation="softmax", hopfield_beta=2.5,
            kernel_initializer="glorot_uniform",
        )
        assert custom.dim == 48
        assert custom.hopfield_dim == 12
        assert custom.activation == "softmax"
        assert custom.hopfield_beta == 2.5
        assert isinstance(custom.kernel_initializer, keras.initializers.GlorotUniform)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"dim": 0, "hopfield_dim": 16}, "dim must be a positive integer"),
            ({"dim": -4, "hopfield_dim": 16}, "dim must be a positive integer"),
            ({"dim": 64, "hopfield_dim": 0}, "hopfield_dim must be a positive integer"),
            ({"dim": 64, "hopfield_dim": -1}, "hopfield_dim must be a positive integer"),
            # 'power' is DELIBERATELY not implemented (D-005) — it must raise, not work.
            ({"dim": 64, "hopfield_dim": 16, "activation": "power"},
             "activation must be one of"),
            ({"dim": 64, "hopfield_dim": 16, "activation": "gelu"},
             "activation must be one of"),
            ({"dim": 64, "hopfield_dim": 16, "hopfield_beta": 0.0},
             "hopfield_beta must be a positive number"),
            ({"dim": 64, "hopfield_dim": 16, "hopfield_beta": -1.0},
             "hopfield_beta must be a positive number"),
        ],
    )
    def test_invalid_config(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            HopfieldNetwork(**kwargs)

    def test_invalid_activation_names_the_valid_set(self):
        with pytest.raises(ValueError) as exc:
            HopfieldNetwork(dim=64, hopfield_dim=16, activation="power")
        message = str(exc.value)
        assert "relu" in message and "softmax" in message

    @pytest.mark.parametrize("activation", ACTIVATIONS)
    def test_forward_pass(self, activation, sample_input, input_shape):
        hopfield = HopfieldNetwork(dim=DIM, hopfield_dim=MEM, activation=activation)
        y = hopfield(sample_input, training=False)
        assert tuple(y.shape) == input_shape
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(y)))

    @pytest.mark.parametrize("activation", ACTIVATIONS)
    def test_call_equals_update(self, activation, sample_input):
        hopfield = HopfieldNetwork(dim=DIM, hopfield_dim=MEM, activation=activation)
        y = keras.ops.convert_to_numpy(hopfield(sample_input, training=False))
        u = keras.ops.convert_to_numpy(hopfield.update(sample_input))
        np.testing.assert_allclose(y, u, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("activation", ACTIVATIONS)
    def test_energy_shape(self, activation, sample_input):
        hopfield = HopfieldNetwork(dim=DIM, hopfield_dim=MEM, activation=activation)
        e = hopfield.energy(sample_input)
        assert tuple(e.shape) == (BATCH,)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(e)))

    def test_build_creates_weights(self, input_shape):
        """Exactly ONE trainable weight of shape (K, D), and NO bias variable."""
        hopfield = HopfieldNetwork(dim=DIM, hopfield_dim=MEM)
        hopfield.build(input_shape)

        assert tuple(hopfield.xi.shape) == (MEM, DIM)   # (K, D)
        assert len(hopfield.trainable_weights) == 1, (
            "HopfieldNetwork ties its up- and down-projection into ONE matrix. Two "
            f"weights means the tying was broken. Got: "
            f"{[w.name for w in hopfield.trainable_weights]}"
        )
        assert len(hopfield.non_trainable_weights) == 0
        assert not any("bias" in w.name for w in hopfield.weights), (
            "HopfieldNetwork MUST be bias-free — the paper's energy E_HN is defined "
            "without a bias and the closed-form gradient cannot express one"
        )

    def test_build_rejects_mismatched_feature_dim(self):
        hopfield = HopfieldNetwork(dim=DIM, hopfield_dim=MEM)
        with pytest.raises(ValueError, match="does not match dim"):
            hopfield.build((BATCH, TOKENS, DIM + 1))

    @pytest.mark.parametrize("activation", ACTIVATIONS)
    def test_serialization_cycle(self, activation, sample_input, input_shape):
        inputs = keras.Input(shape=input_shape[1:])
        outputs = HopfieldNetwork(
            dim=DIM, hopfield_dim=MEM, activation=activation, hopfield_beta=1.7,
        )(inputs)
        model = keras.Model(inputs, outputs)

        y_before = keras.ops.convert_to_numpy(model(sample_input, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "hopfield_network.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            y_after = keras.ops.convert_to_numpy(loaded(sample_input, training=False))

        np.testing.assert_allclose(
            y_before, y_after, rtol=1e-6, atol=1e-6,
            err_msg="HopfieldNetwork outputs differ after a .keras round-trip",
        )

    def test_get_config_complete(self):
        hopfield = HopfieldNetwork(
            dim=48, hopfield_dim=12, activation="softmax", hopfield_beta=2.5,
        )
        config = hopfield.get_config()
        for key in ("dim", "hopfield_dim", "activation", "hopfield_beta",
                    "kernel_initializer"):
            assert key in config, f"{key} missing from get_config()"
        assert config["dim"] == 48
        assert config["hopfield_dim"] == 12
        assert config["activation"] == "softmax"
        assert config["hopfield_beta"] == 2.5

    def test_from_config_reconstruction(self, sample_input):
        original = HopfieldNetwork(
            dim=DIM, hopfield_dim=MEM, activation="softmax", hopfield_beta=2.5,
        )
        rebuilt = HopfieldNetwork.from_config(original.get_config())

        assert rebuilt.dim == original.dim
        assert rebuilt.hopfield_dim == original.hopfield_dim
        assert rebuilt.activation == original.activation
        assert rebuilt.hopfield_beta == original.hopfield_beta
        assert isinstance(
            rebuilt.kernel_initializer, keras.initializers.TruncatedNormal
        )

        # Copy the weights over -> identical forward output.
        original.build(sample_input.shape)
        rebuilt.build(sample_input.shape)
        rebuilt.set_weights(original.get_weights())

        y0 = keras.ops.convert_to_numpy(original(sample_input, training=False))
        y1 = keras.ops.convert_to_numpy(rebuilt(sample_input, training=False))
        np.testing.assert_allclose(y0, y1, rtol=1e-6, atol=1e-6)

    def test_compute_output_shape(self, layer, input_shape):
        layer.build(input_shape)
        assert layer.compute_output_shape(input_shape) == input_shape

    def test_compute_output_shape_before_build(self, input_shape):
        unbuilt = HopfieldNetwork(dim=DIM, hopfield_dim=MEM)
        assert not unbuilt.built
        assert unbuilt.compute_output_shape(input_shape) == input_shape
        assert unbuilt.compute_output_shape((None, 5, DIM)) == (None, 5, DIM)
        assert not unbuilt.built

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_variable_batch_size(self, batch_size):
        hopfield = HopfieldNetwork(dim=DIM, hopfield_dim=MEM)
        rng = np.random.default_rng(batch_size)
        x = rng.standard_normal((batch_size, 5, DIM)).astype("float32")
        y = hopfield(x, training=False)
        assert tuple(y.shape) == (batch_size, 5, DIM)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(y)))
        assert tuple(hopfield.energy(x).shape) == (batch_size,)

    @pytest.mark.parametrize("activation", ACTIVATIONS)
    def test_gradient_flow(self, activation, sample_input, input_shape):
        hopfield = HopfieldNetwork(dim=DIM, hopfield_dim=MEM, activation=activation)
        hopfield.build(input_shape)
        _excite(hopfield)
        x = tf.convert_to_tensor(sample_input)

        with tf.GradientTape() as tape:
            y = hopfield(x, training=True)
            loss = tf.reduce_mean(tf.square(y))

        grads = tape.gradient(loss, hopfield.trainable_weights)
        assert len(grads) == 1
        for g, w in zip(grads, hopfield.trainable_weights):
            assert g is not None, f"no gradient for {w.name}"
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(g))), (
                f"non-finite gradient for {w.name}"
            )


if __name__ == "__main__":
    pytest.main([__file__])
