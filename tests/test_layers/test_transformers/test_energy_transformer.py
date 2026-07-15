"""
Test suite for layers/transformers/energy_transformer.py.

Covers the two classes of the module:

- :class:`HopfieldNetwork` — the Energy Transformer's associative-memory module
  (arXiv:2302.07253 eq. 5, 9): ONE tied ``(K, D)`` matrix, no bias, strictly per token. It
  defines a scalar energy ``E_HN(g)`` whose ``update()`` returns the hand-coded closed-form
  ``-dE_HN/dg``.
- :class:`EnergyTransformer` — the ``T``-step energy-descent block (eq. 6, alg. 1).

The TWO headline tests (plan plan_2026-07-13_57c9833e):

- ``TestHopfieldNetwork.test_gradient_oracle`` (S6b). Each activation carries BOTH an
  energy and a gradient, and they must be a matched pair: pairing the relu energy with the
  softmax gradient (or vice versa) yields a layer that runs, trains, emits finite output —
  and does not descend. The oracle is parametrized over both activations to catch that.
  ``tf.GradientTape`` is the oracle and is legitimate HERE — it is forbidden only in
  ``src/`` (decisions.md D-001).
- ``TestEnergyTransformer.test_energy_is_non_increasing`` (S7). The block's whole reason to
  exist. It carries a MANDATORY anti-degeneracy assertion alongside the monotonicity one:
  ``np.diff(E) <= tol`` is trivially satisfiable by a DEAD layer (a constant-zero energy, or
  an update of ~0 so ``x`` never moves), so a green monotonicity check ALONE proves nothing.
  Both assertions, always.

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

from dl_techniques.layers.transformers.energy_transformer import (
    EnergyTransformer,
    HopfieldNetwork,
    WeightedAdjacencyProjector,
)


# ---------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------

BATCH, TOKENS, DIM, MEM = 3, 7, 32, 16
HEADS, HEAD_DIM = 4, 8
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
    @pytest.mark.parametrize("masked", [False, True])
    def test_gradient_oracle(self, activation, masked, sample_input):
        """update(g) MUST equal -d/dg [ sum_b energy(g) ], for EVERY activation (S6b).

        This is what proves the energy/gradient PAIR is consistent. A crossed pair (relu
        energy + softmax r, or vice versa) runs, trains, and does not descend.

        The `masked` cell is the iter-2 / R2 extension (C14): once `energy()` excludes the
        PAD tokens from its token SUM, its gradient is exactly zero at those rows — so
        `update()` must be zero there too, or it stops being `-dE/dg`. This is checked
        TENSOR-WIDE (PAD rows included), which is the whole point: masking one of the pair
        and not the other is invisible at the real-token rows (decisions.md D-005).
        """
        hopfield = HopfieldNetwork(
            dim=DIM, hopfield_dim=MEM, activation=activation, hopfield_beta=0.7,
        )
        hopfield.build((BATCH, TOKENS, DIM))
        _excite(hopfield)

        # Larger-variance g so the relu branch is not sitting on a dead half-plane.
        g = tf.Variable(sample_input * 2.0, dtype=tf.float32)

        mask = None
        if masked:
            keep = np.ones((BATCH, TOKENS), dtype="bool")
            keep[:, -2:] = False                      # the last 2 tokens are PAD
            mask = keras.ops.convert_to_tensor(keep)

        with tf.GradientTape() as tape:
            energy = tf.reduce_sum(hopfield.energy(g, mask=mask))
        grad = tape.gradient(energy, g)

        assert grad is not None, "energy() has no gradient path to g"

        update = keras.ops.convert_to_numpy(hopfield.update(g, mask=mask))

        if masked:
            # The masked rows must be EXACTLY zero (the masked energy does not depend on
            # them at all), and the real rows must not be — otherwise the oracle below is
            # comparing zeros with zeros.
            assert np.abs(update[:, -2:, :]).max() == 0.0, (
                "a masked (PAD) token got a NON-ZERO Hopfield update, but the masked "
                "energy has zero gradient there — update() != -dE/dg (decisions.md D-005)"
            )
            assert np.abs(update[:, :-2, :]).max() > 1e-3, (
                "the real tokens' update is ~0 — the masked oracle would pass vacuously"
            )

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


# ---------------------------------------------------------------------
# EnergyTransformer
# ---------------------------------------------------------------------

def _block(**overrides) -> EnergyTransformer:
    """An EnergyTransformer with the test dimensions; `overrides` win."""
    kwargs = dict(
        embed_dim=DIM,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        hopfield_dim=MEM,
        num_steps=4,
        step_size=0.05,
    )
    kwargs.update(overrides)
    return EnergyTransformer(**kwargs)


def _excite_block(
    block: EnergyTransformer,
    attn_scale: float = 2.0,
    xi_scale: float = 0.2,
    seed: int = 7,
) -> None:
    """Replace the N(0, 0.02) inits of the block's sub-layers with visibly-sized weights.

    TWO separate scales, and the ASYMMETRY IS LOAD-BEARING. Do not "simplify" this to one
    shared scale.

    1. **Why excite at all.** With the paper's default init the energy still descends, but
       only by ~5e-3 out of an |E| of ~1.4e2 — a technically-non-degenerate descent whose
       margin is thin enough to make S7 a precision coin-flip.

    2. **Why the attention weights are excited ~10x harder than the Hopfield memories.**
       Measured, not guessed (plan S10, run live during step 5): with a shared scale of 0.5
       the Hopfield update swamps the attention update (|u_hn| >> |u_att|, because the
       Hopfield update is QUADRATIC in `xi`), and the descent is then carried almost
       entirely by the (correct) Hopfield term. In that regime S7 STAYS GREEN even when
       `EnergyAttention.update()`'s second term `term_k` is DELETED — i.e. S7 is vacuous
       with respect to the single highest-risk line in the whole feature. At
       `attn_scale=2.0, xi_scale=0.2` the attention update leads (|u_att| ~ 7.7e2 vs
       |u_hn| ~ 3.1) while E_HN is still a real, non-zero term of E, and S7 becomes
       sensitive: term_k present -> max diff(E) ~ -4.1e1 (descent, all 4 parametrizations);
       term_k deleted -> max diff(E) ~ +6.4e1 .. +2.8e2 (ASCENT, all 4). The guard bites.
    """
    rng = np.random.default_rng(seed)
    for w, scale in (
        (block.attention.w_key, attn_scale),
        (block.attention.w_query, attn_scale),
        (block.hopfield.xi, xi_scale),
    ):
        w.assign((rng.standard_normal(tuple(w.shape)) * scale).astype("float32"))


# --- weighted-adjacency descent (S7 with Branch A active) -------------------
#
# S7, but with `use_weighted_adjacency=True` and a genuinely EXCITED `Ŵ` hoisted into every
# descent step. This is the strongest BLOCK-LEVEL integration guard for the Branch-A gradient
# (plan-2026-07-15T053724-78001af1): the closed-form `update()` (attention + Hopfield) is
# `-dE/dg` of the block's reported energy ONLY IF the same constant `Ŵ` reaches BOTH `energy()`
# and both `update()` calls. If the `omega_eff = omega·Ŵ` factor were subtly wrong, or `Ŵ` were
# fed to only one path, the energy would ASCEND here (falsification signal #3) even though the
# per-tensor oracle at N∈{64,1024} passed. N >= 64 (never a toy N — LESSONS).
WEIGHTED_TOKENS = 64


def _weighted_block(**overrides) -> EnergyTransformer:
    """An EnergyTransformer with the Branch-A weighted path ON (noiseless, energy-tracing)."""
    kwargs = dict(
        embed_dim=DIM,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        hopfield_dim=MEM,
        num_steps=20,
        step_size=0.01,               # small alpha: a large one may legitimately overshoot
        attn_self=True,               # fully-connected adjacency below -> self-attention allowed
        noise_std=0.0,                # descent is NOT claimed under Langevin noise (eq. 27)
        return_energy=True,
        use_weighted_adjacency=True,
        adjacency_proj_dim=8,         # conv sees 8^2 = 64 channels (the OOM escape hatch)
    )
    kwargs.update(overrides)
    return EnergyTransformer(**kwargs)


def _excite_projector(block: EnergyTransformer, seed: int = 21,
                      kernel_scale: float = 0.5, bias: float = 1.0) -> None:
    """Drive the TRAINABLE projector to emit a non-trivial POSITIVE-and-VARYING ``Ŵ``.

    At init the projector's ``Conv2D`` is glorot / zero-bias, so ``Ŵ ~ 0`` and the weighted
    path is effectively INERT — a descent test on it would prove nothing about the ``Ŵ``
    gradient. A positive bias floors ``Ŵ`` above zero while the random kernel over the varying
    ``X⊗X`` makes it vary per pair, so ``omega_eff = omega·Ŵ`` genuinely reweights the
    descent. (The `token_proj` Dense keeps its glorot init — ``X'`` is already varying.)
    """
    rng = np.random.default_rng(seed)
    conv = block.adjacency_projector.conv
    conv.kernel.assign(
        (rng.standard_normal(tuple(conv.kernel.shape)) * kernel_scale).astype("float32")
    )
    conv.bias.assign(np.full(tuple(conv.bias.shape), bias, dtype="float32"))


class TestEnergyTransformer:
    """S7 (energy descent), S8b (mask), S2/S3 (the layer contract) for the ET block."""

    # -- S7: THE HEADLINE TEST ---------------------------------------

    @pytest.mark.parametrize("attn_self", [False, True])
    @pytest.mark.parametrize("hopfield_activation", ACTIVATIONS)
    def test_energy_is_non_increasing(self, attn_self, hopfield_activation, sample_input):
        """E(t) must be MONOTONE NON-INCREASING across the T descent steps (S7).

        TWO assertions, both mandatory:

        1. **Monotonicity** — per sample, `diff(E) <= tol`. This is the guarantee: the block
           adds `alpha * update` where `update == -dE/dg`, and `EnergyLayerNorm`'s PSD
           Hessian `dg/dx` makes `dE/dt = -(dE/dg)^T (dg/dx) (dE/dg) <= 0`.
        2. **ANTI-DEGENERACY** — `E[0] - E[-1] > 1e-3` per sample. WITHOUT THIS, S7 IS
           WORTHLESS: `diff(E) <= tol` is trivially satisfied by a DEAD layer (constant-zero
           energy, or an update of ~0 so `x` never moves). A green monotonicity check on a
           dead layer proves nothing. Do not delete this assertion to "fix" a failure — a
           failure here means the descent is not real.

        The `_excite_block` fixture's attention-heavy weight scaling is REQUIRED for this
        test to have teeth — see its docstring. With a Hopfield-dominated update this test
        passes even with `EnergyAttention.update()`'s `term_k` deleted.
        """
        block = _block(
            num_steps=20,
            step_size=0.01,          # small alpha: a large one may legitimately overshoot
            attn_self=attn_self,
            hopfield_activation=hopfield_activation,
            noise_std=0.0,           # descent is NOT claimed under Langevin noise (eq. 27)
            return_energy=True,
        )
        block.build((BATCH, TOKENS, DIM))
        _excite_block(block)

        x = sample_input * 2.0
        _, energies = block(x, training=False)
        e = keras.ops.convert_to_numpy(energies)          # (B, T + 1)

        assert e.shape == (BATCH, 21)
        assert np.all(np.isfinite(e))

        diffs = np.diff(e, axis=-1)                        # (B, T)
        drop = e[:, 0] - e[:, -1]
        print(
            f"\n[S7 descent] attn_self={attn_self} hopfield_activation={hopfield_activation}: "
            f"E[0]={np.array2string(e[:, 0], precision=3)} "
            f"E[-1]={np.array2string(e[:, -1], precision=3)} "
            f"drop={np.array2string(drop, precision=3)} "
            f"max_diff={diffs.max():.6e}"
        )

        # 1. monotone, PER SAMPLE (a batch mean would hide a per-sample violation).
        assert np.all(diffs <= 1e-5), (
            f"ENERGY WENT UP. max diff(E) = {diffs.max():.6e}. The block is performing "
            "energy ASCENT, not descent. Check the sign at `x = x + step_size * update` "
            "(update() returns -dE/dg, so the block must ADD), and check that both terms "
            "of EnergyAttention.update() are present."
        )

        # 2. anti-degeneracy (plan STOP-IF 2) — the energy actually MOVED.
        assert np.all(drop > 1e-3), (
            f"the energy barely moved (min drop = {drop.min():.6e}). Assertion 1 above is "
            "therefore VACUOUS — it would pass on a dead layer. The descent is not real."
        )

    # -- F-05a: the num_steps == 1 boundary --------------------------

    def test_single_step_block(self, sample_input):
        """`num_steps=1` still yields T + 1 == 2 energy readings (plan C7).

        This is the OFF-BY-ONE boundary of the whole trace protocol, and it was untested:
        `energies` gets one reading per descent step INSIDE the loop plus a final reading
        AFTER it (the state the caller actually receives), so `num_steps=1` must give
        `(B, 2)` — a pre-update and a post-update energy — NOT `(B, 1)`. Dropping the
        post-loop `energies.append(...)` makes every other trace test still pass on the
        first T entries while the LAST, most interesting reading silently disappears; this
        test is the only thing that sees it.

        The descent assertion is paired with the MANDATORY anti-degeneracy one, for the
        same reason S7 is (LESSONS [I:5]): `E[:, 1] <= E[:, 0]` alone is vacuously true of
        a dead block whose energy never moves. `_excite_block` (the S7 fixture — reused, not
        reinvented) is what gives the drop enough magnitude to clear the floor by orders of
        magnitude, and `noise_std=0` because descent is only claimed for the noise-free
        dynamics (eq. 27).
        """
        block = _block(
            num_steps=1,
            step_size=0.01,
            noise_std=0.0,
            return_energy=True,
        )
        block.build((BATCH, TOKENS, DIM))
        _excite_block(block)

        x = sample_input * 2.0
        out, energies = block(x, training=False)

        assert tuple(out.shape) == (BATCH, TOKENS, DIM)

        e = keras.ops.convert_to_numpy(energies)
        # THE off-by-one assertion: T = 1 -> T + 1 = 2 readings, not 1.
        assert e.shape == (BATCH, 2), (
            f"num_steps=1 produced an energy trace of shape {e.shape}, expected "
            f"{(BATCH, 2)}. A T-step block reports T + 1 energies: one per step, plus a "
            "final reading of the state it returns. Check the post-loop energies.append()."
        )
        assert np.all(np.isfinite(e))
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

        drop = e[:, 0] - e[:, 1]
        print(
            f"\n[C7 num_steps=1] E[:,0]={np.array2string(e[:, 0], precision=3)} "
            f"E[:,1]={np.array2string(e[:, 1], precision=3)} "
            f"drop={np.array2string(drop, precision=3)}"
        )

        # 1. descent across the single step.
        assert np.all(e[:, 1] <= e[:, 0] + 1e-5), (
            f"energy went UP on the single step (min drop = {drop.min():.6e}); the block "
            "is performing energy ASCENT."
        )
        # 2. anti-degeneracy — the one step actually MOVED the energy (LESSONS [I:5]).
        assert np.all(drop > 1e-3), (
            f"the energy barely moved on the single step (min drop = {drop.min():.6e}). "
            "Assertion 1 is therefore VACUOUS — it would pass on a dead block."
        )

    # -- S7 with the Branch-A weighted path active -------------------

    @pytest.mark.parametrize("hopfield_activation", ACTIVATIONS)
    def test_energy_is_non_increasing_with_weighted_adjacency(self, hopfield_activation):
        """S7 with ``use_weighted_adjacency=True`` and an EXCITED ``Ŵ`` at N=64 (SC3).

        The block-level proof that the hoisted-``Ŵ`` ``update()`` really is ``-dE/dg``: with
        an excited constant ``Ŵ`` fed to BOTH `energy()` and both `update()` calls, the T-step
        energy must still be MONOTONE NON-INCREASING (`diff(E) <= 1e-5`) AND ANTI-DEGENERATE
        (`E[:,0] - E[:,-1] > 1e-3`). Ascent here is falsification signal #3 (the `omega_eff`
        gradient is wrong, or ``Ŵ`` reached only one path) — STOP and report, do NOT loosen.
        """
        N = WEIGHTED_TOKENS
        block = _weighted_block(hopfield_activation=hopfield_activation)
        block.build((BATCH, N, DIM))
        _excite_block(block)
        _excite_projector(block)

        rng = np.random.default_rng(99)
        x = (rng.standard_normal((BATCH, N, DIM)) * 2.0).astype("float32")
        adjacency = np.ones((BATCH, N, N), dtype="float32")     # fully connected (+ attn_self)

        # The weighted path is genuinely ACTIVE and Ŵ is EXCITED (non-trivial + varying, not
        # inert). The real projector emits a SIGNED Ŵ here (X⊗X easily overwhelms the +1 bias);
        # descent holds for signed Ŵ exactly as for positive (the oracle at
        # test_energy_attention.py proved BOTH "positive" and "signed" kinds), and a signed,
        # large-magnitude Ŵ is a STRONGER reweighting than a positive-only one.
        w_hat = keras.ops.convert_to_numpy(block._weighted_adjacency(x, adjacency))
        assert w_hat.shape == (BATCH, HEADS, N, N)
        assert np.all(np.isfinite(w_hat))
        assert np.abs(w_hat).mean() > 1.0, (
            f"Ŵ is ~0 (mean|Ŵ|={np.abs(w_hat).mean():.3e}) — the weighted path is inert, the "
            "descent below would prove nothing about the Ŵ gradient"
        )
        assert w_hat.std() > 1e-3, f"Ŵ is ~constant (std={w_hat.std():.3e}) — path is inert"

        _, energies = block(x, attention_mask=adjacency, training=False)
        e = keras.ops.convert_to_numpy(energies)                 # (B, T + 1)

        assert e.shape == (BATCH, 21)
        assert np.all(np.isfinite(e))

        diffs = np.diff(e, axis=-1)
        drop = e[:, 0] - e[:, -1]
        print(
            f"\n[S7 weighted] hopfield_activation={hopfield_activation}: "
            f"E[0]={np.array2string(e[:, 0], precision=2)} "
            f"E[-1]={np.array2string(e[:, -1], precision=2)} "
            f"drop={np.array2string(drop, precision=2)} max_diff={diffs.max():.3e} "
            f"W_hat[min={w_hat.min():.2f} max={w_hat.max():.2f} std={w_hat.std():.2f}]"
        )

        # 1. monotone, PER SAMPLE, with the weighted path active.
        assert np.all(diffs <= 1e-5), (
            f"ENERGY WENT UP with Ŵ active (max diff(E) = {diffs.max():.3e}) — falsification "
            "signal #3. The block is ASCENDING, so `update() != -dE/dg` under the weighted "
            "path: check `omega_eff = omega·Ŵ` reaches BOTH term_q and term_k AND that the "
            "SAME Ŵ tensor reaches energy() and both update() calls (D-001/D-002)."
        )
        # 2. anti-degeneracy — the weighted-path energy actually MOVED.
        assert np.all(drop > 1e-3), (
            f"the weighted energy barely moved (min drop = {drop.min():.3e}). Assertion 1 is "
            "therefore VACUOUS — it would pass on a dead layer. The descent is not real."
        )

    def test_single_step_block_with_weighted_adjacency(self):
        """The ``num_steps=1`` boundary (T + 1 == 2 readings) with the weighted path active.

        Reuses the SC3 excitation, at ``num_steps=1``: `energies` must be `(B, 2)` (one
        pre-update + one post-update reading), the single step must DESCEND, and it must
        ANTI-DEGENERATELY move (LESSONS [I:5]) — with ``Ŵ`` hoisted into that single step.
        """
        N = WEIGHTED_TOKENS
        block = _weighted_block(num_steps=1)
        block.build((BATCH, N, DIM))
        _excite_block(block)
        _excite_projector(block)

        rng = np.random.default_rng(99)
        x = (rng.standard_normal((BATCH, N, DIM)) * 2.0).astype("float32")
        adjacency = np.ones((BATCH, N, N), dtype="float32")

        out, energies = block(x, attention_mask=adjacency, training=False)
        assert tuple(out.shape) == (BATCH, N, DIM)

        e = keras.ops.convert_to_numpy(energies)
        assert e.shape == (BATCH, 2), (
            f"num_steps=1 produced an energy trace of shape {e.shape}, expected {(BATCH, 2)}"
        )
        assert np.all(np.isfinite(e))
        drop = e[:, 0] - e[:, 1]
        assert np.all(e[:, 1] <= e[:, 0] + 1e-5), (
            f"energy went UP on the single weighted step (min drop = {drop.min():.3e})"
        )
        assert np.all(drop > 1e-3), (
            f"the weighted energy barely moved on the single step (min drop = {drop.min():.3e})"
        )

    # -- S8b: block-level mask ---------------------------------------

    def test_masked_token_has_no_influence(self, sample_input):
        """A masked-out token must not influence ANY other token's output (S8b).

        Per D-008, a rank-2 (B, N) mask is a SYMMETRIC per-token validity mask (key AND
        query axes) — a token masked only as a KEY would still reach every other token via
        `term_k`, which sums over the query axis.
        """
        j = 3
        block = _block(num_steps=3, step_size=0.05, noise_std=0.0)
        block.build((BATCH, TOKENS, DIM))
        _excite_block(block)

        mask = np.ones((BATCH, TOKENS), dtype="float32")
        mask[:, j] = 0.0

        x0 = sample_input.copy()
        x1 = sample_input.copy()
        rng = np.random.default_rng(99)
        x1[:, j, :] += rng.standard_normal((BATCH, DIM)).astype("float32") * 3.0

        y0 = keras.ops.convert_to_numpy(
            block(x0, attention_mask=mask, training=False)
        )
        y1 = keras.ops.convert_to_numpy(
            block(x1, attention_mask=mask, training=False)
        )

        others = [i for i in range(TOKENS) if i != j]
        np.testing.assert_allclose(
            y0[:, others, :], y1[:, others, :], atol=1e-6,
            err_msg=(
                "perturbing a MASKED token changed another token's output — the mask does "
                "not deliver zero influence. Is the rank-2 mask being applied to the key "
                "axis only (D-008)?"
            ),
        )

        # Anti-vacuity 1: the perturbation was real and IS visible at the masked row. Since
        # D-006 a rank-2 `attention_mask` IS a per-token validity mask, so the masked row is
        # FROZEN (both sub-updates are zero there) and comes out exactly as it went in — the
        # difference below is therefore the input perturbation itself, passed through.
        assert np.abs(y1[:, j, :] - y0[:, j, :]).max() > 1e-3, (
            "the perturbation changed nothing at all — the comparison above would pass "
            "vacuously"
        )

        # Anti-vacuity 2 (C15, iter-2 / R3): the assertion above measures the INPUT
        # perturbation, and the equality above is satisfied by a DEAD block for free. Assert
        # that the BLOCK moved the OTHER tokens' state, so `step_size -> 0` goes RED here.
        # This is the same copy-paste hole R3 found at the Keras-mask tests; found by the
        # iter-2 audit. See decisions.md D-005.
        assert np.abs(y0[:, others, :] - x0[:, others, :]).max() > 1e-3, (
            "the block did not move the unmasked tokens at all — a DEAD block satisfies "
            "'perturbing a masked token changes nothing' vacuously (decisions.md D-005)"
        )

    # -- return_energy / noise ---------------------------------------

    def test_return_energy_shape(self, sample_input):
        block = _block(num_steps=6, return_energy=True)
        y, energies = block(sample_input, training=False)
        assert tuple(y.shape) == (BATCH, TOKENS, DIM)
        assert tuple(energies.shape) == (BATCH, 7)     # (B, num_steps + 1)

    def test_noise_only_in_training(self, sample_input):
        """noise_std > 0: inference is DETERMINISTIC, training is STOCHASTIC (eq. 27)."""
        block = _block(noise_std=0.1, seed=42)

        a = keras.ops.convert_to_numpy(block(sample_input, training=False))
        b = keras.ops.convert_to_numpy(block(sample_input, training=False))
        np.testing.assert_allclose(
            a, b, rtol=1e-6, atol=1e-6,
            err_msg="noise leaked into inference — it must apply only when training",
        )

        c = keras.ops.convert_to_numpy(block(sample_input, training=True))
        d = keras.ops.convert_to_numpy(block(sample_input, training=True))
        assert np.abs(c - d).max() > 1e-4, (
            "two training-mode calls were identical — the noise_std branch never fired"
        )

    def test_no_noise_is_deterministic_in_training(self, sample_input):
        """noise_std == 0 (the default): training and inference agree exactly."""
        block = _block(noise_std=0.0)
        a = keras.ops.convert_to_numpy(block(sample_input, training=True))
        b = keras.ops.convert_to_numpy(block(sample_input, training=False))
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    # -- standard layer contract --------------------------------------

    def test_instantiation(self):
        default = EnergyTransformer(
            embed_dim=64, num_heads=8, head_dim=16, hopfield_dim=256,
        )
        assert default.embed_dim == 64
        assert default.num_steps == 12
        assert default.step_size == 0.1
        assert default.beta is None
        assert default.attn_self is False
        assert default.hopfield_activation == "relu"
        assert default.noise_std == 0.0
        assert default.return_energy is False
        assert default.hopfield_beta == 1.0
        assert default.norm_epsilon == 1e-5
        assert default.seed is None
        assert not default.built

        # Sub-layers exist BEFORE build (the Keras 3 golden pattern — never lazy).
        assert default.norm is not None
        assert default.attention is not None
        assert default.hopfield is not None

        custom = EnergyTransformer(
            embed_dim=DIM, num_heads=2, head_dim=4, hopfield_dim=8,
            num_steps=3, step_size=0.02, beta=0.5, attn_self=True,
            hopfield_activation="softmax", noise_std=0.3, return_energy=True,
            hopfield_beta=2.5, norm_epsilon=1e-3, seed=11,
        )
        assert custom.num_steps == 3
        assert custom.beta == 0.5
        assert custom.attn_self is True
        assert custom.hopfield_activation == "softmax"
        assert custom.return_energy is True
        # The three additive kwargs (D-007) reach the sub-layers they belong to.
        assert custom.hopfield.hopfield_beta == 2.5
        assert custom.norm.epsilon == 1e-3
        assert custom.attention.beta == 0.5

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"embed_dim": 0}, "embed_dim must be a positive integer"),
            ({"embed_dim": -8}, "embed_dim must be a positive integer"),
            ({"num_heads": 0}, "num_heads must be a positive integer"),
            ({"num_heads": -1}, "num_heads must be a positive integer"),
            ({"head_dim": 0}, "head_dim must be a positive integer"),
            ({"head_dim": -3}, "head_dim must be a positive integer"),
            ({"hopfield_dim": 0}, "hopfield_dim must be a positive integer"),
            ({"hopfield_dim": -2}, "hopfield_dim must be a positive integer"),
            ({"num_steps": 0}, "num_steps must be an integer >= 1"),
            ({"num_steps": -5}, "num_steps must be an integer >= 1"),
            ({"step_size": 0.0}, "step_size must be a positive number"),
            ({"step_size": -0.1}, "step_size must be a positive number"),
            ({"noise_std": -0.1}, "noise_std must be a non-negative number"),
            # delegated to the sub-layers (ONE source of truth per rule)
            ({"hopfield_activation": "power"}, "activation must be one of"),
            ({"beta": 0.0}, "beta must be a positive number"),
        ],
    )
    def test_invalid_config(self, kwargs, match):
        base = dict(embed_dim=DIM, num_heads=HEADS, head_dim=HEAD_DIM, hopfield_dim=MEM)
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            EnergyTransformer(**base)

    @pytest.mark.parametrize("attn_self", [False, True])
    @pytest.mark.parametrize("hopfield_activation", ACTIVATIONS)
    def test_forward_pass(self, attn_self, hopfield_activation, sample_input, input_shape):
        block = _block(
            attn_self=attn_self, hopfield_activation=hopfield_activation,
        )
        y = block(sample_input, training=False)
        assert tuple(y.shape) == input_shape
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(y)))

    def test_build_creates_sublayers_and_weights(self, input_shape):
        """All three sub-layers built; exactly 5 trainable weights, no more, no fewer.

        EnergyLayerNorm(gamma, delta) = 2, EnergyAttention(w_key, w_query) = 2,
        HopfieldNetwork(xi) = 1. A 6th weight means a bias crept in somewhere (all three
        sub-layers are bias-free by construction except the LayerNorm's `delta`, which IS
        the Lagrangian's linear term). A 3rd/4th weight missing means a sub-layer was
        lazily built and will DROP ITS WEIGHTS on a .keras round-trip.
        """
        block = _block()
        block.build(input_shape)

        assert block.built
        assert block.norm.built
        assert block.attention.built
        assert block.hopfield.built

        names = [w.name for w in block.trainable_weights]
        assert len(block.trainable_weights) == 5, (
            f"expected 5 trainable weights (2 norm + 2 attention + 1 hopfield), got "
            f"{len(block.trainable_weights)}: {names}"
        )
        assert len(block.non_trainable_weights) == 0

        assert tuple(block.norm.gamma.shape) == ()
        assert tuple(block.norm.delta.shape) == (DIM,)
        assert tuple(block.attention.w_key.shape) == (HEAD_DIM, HEADS, DIM)
        assert tuple(block.attention.w_query.shape) == (HEAD_DIM, HEADS, DIM)
        assert tuple(block.hopfield.xi.shape) == (MEM, DIM)

    def test_build_rejects_mismatched_feature_dim(self):
        block = _block()
        with pytest.raises(ValueError, match="does not match embed_dim"):
            block.build((BATCH, TOKENS, DIM + 1))

    @pytest.mark.parametrize("noise_std", [0.0, 0.1])
    def test_serialization_cycle(self, noise_std, sample_input, input_shape):
        """S2: full .keras round-trip through a keras.Model wrapper.

        `training=False` on BOTH forward calls is MANDATORY, not stylistic: with
        `noise_std > 0` this layer is stochastic, and a bare `model(x)` would compare two
        different noise draws and FLAKE (LESSONS [I:4]). Both noise settings are exercised
        because the `seed` / SeedGenerator machinery is exactly what a round-trip can drop.
        """
        inputs = keras.Input(shape=input_shape[1:])
        outputs = EnergyTransformer(
            embed_dim=DIM, num_heads=HEADS, head_dim=HEAD_DIM, hopfield_dim=MEM,
            num_steps=3, step_size=0.05, hopfield_activation="softmax",
            hopfield_beta=1.7, norm_epsilon=1e-4, noise_std=noise_std, seed=123,
        )(inputs)
        model = keras.Model(inputs, outputs)

        y_before = keras.ops.convert_to_numpy(model(sample_input, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "energy_transformer.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            y_after = keras.ops.convert_to_numpy(loaded(sample_input, training=False))

        np.testing.assert_allclose(
            y_before, y_after, rtol=1e-6, atol=1e-6,
            err_msg=(
                "EnergyTransformer outputs differ after a .keras round-trip. A sub-layer "
                "created lazily (in build()/call() instead of __init__) silently drops its "
                "weights on save/load."
            ),
        )

    def test_get_config_complete(self):
        block = EnergyTransformer(
            embed_dim=DIM, num_heads=2, head_dim=4, hopfield_dim=8,
            num_steps=3, step_size=0.02, beta=0.5, attn_self=True,
            hopfield_activation="softmax", noise_std=0.3, return_energy=True,
            hopfield_beta=2.5, norm_epsilon=1e-3, seed=11,
        )
        config = block.get_config()
        for key in ("embed_dim", "num_heads", "head_dim", "hopfield_dim", "num_steps",
                    "step_size", "beta", "attn_self", "hopfield_activation", "noise_std",
                    "return_energy", "hopfield_beta", "norm_epsilon", "seed"):
            assert key in config, f"{key} missing from get_config()"
        assert config["num_steps"] == 3
        assert config["step_size"] == 0.02
        assert config["beta"] == 0.5
        assert config["attn_self"] is True
        assert config["hopfield_activation"] == "softmax"
        assert config["noise_std"] == 0.3
        assert config["return_energy"] is True
        assert config["hopfield_beta"] == 2.5
        assert config["norm_epsilon"] == 1e-3
        assert config["seed"] == 11

    def test_from_config_reconstruction(self, sample_input, input_shape):
        original = _block(hopfield_activation="softmax", hopfield_beta=1.3, seed=5)
        rebuilt = EnergyTransformer.from_config(original.get_config())

        assert rebuilt.get_config() == original.get_config()

        original.build(input_shape)
        rebuilt.build(input_shape)
        rebuilt.set_weights(original.get_weights())

        y0 = keras.ops.convert_to_numpy(original(sample_input, training=False))
        y1 = keras.ops.convert_to_numpy(rebuilt(sample_input, training=False))
        np.testing.assert_allclose(y0, y1, rtol=1e-6, atol=1e-6)

    def test_compute_output_shape(self, input_shape):
        block = _block()
        block.build(input_shape)
        assert block.compute_output_shape(input_shape) == input_shape

    def test_compute_output_shape_before_build(self, input_shape):
        """compute_output_shape must work PRE-BUILD, from config ints only."""
        unbuilt = _block(num_steps=9)
        assert not unbuilt.built
        assert unbuilt.compute_output_shape(input_shape) == input_shape
        assert unbuilt.compute_output_shape((None, 5, DIM)) == (None, 5, DIM)
        assert not unbuilt.built

        with_energy = _block(num_steps=9, return_energy=True)
        assert not with_energy.built
        assert with_energy.compute_output_shape(input_shape) == (
            input_shape, (BATCH, 10),
        )
        assert with_energy.compute_output_shape((None, 5, DIM)) == (
            (None, 5, DIM), (None, 10),
        )
        assert not with_energy.built

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_variable_batch_size(self, batch_size):
        block = _block(num_steps=2, return_energy=True)
        rng = np.random.default_rng(batch_size)
        x = rng.standard_normal((batch_size, 5, DIM)).astype("float32")
        y, energies = block(x, training=False)
        assert tuple(y.shape) == (batch_size, 5, DIM)
        assert tuple(energies.shape) == (batch_size, 3)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(y)))

    @pytest.mark.parametrize("hopfield_activation", ACTIVATIONS)
    def test_gradient_flow(self, hopfield_activation, sample_input, input_shape):
        """Every one of the 5 trainable weights receives a non-None, finite gradient."""
        block = _block(hopfield_activation=hopfield_activation)
        block.build(input_shape)
        _excite_block(block)
        x = tf.convert_to_tensor(sample_input)

        with tf.GradientTape() as tape:
            y = block(x, training=True)
            loss = tf.reduce_mean(tf.square(y))

        grads = tape.gradient(loss, block.trainable_weights)
        assert len(grads) == 5
        for g, w in zip(grads, block.trainable_weights):
            assert g is not None, f"no gradient for {w.name}"
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(g))), (
                f"non-finite gradient for {w.name}"
            )


class TestDtypePolicies:
    """S13: the ET block is FINITE under every global dtype policy.

    Iteration 1's `EnergyAttention` applied a `-1e9` mask bias in the COMPUTE dtype. In
    fp16 that constant IS `-inf`, and the bias was applied unconditionally, so every
    UNMASKED position computed `0 * -inf = NaN`: `EnergyTransformer(...)(x)` returned
    **512/512 NaN under `mixed_float16` with no mask supplied**. Measured, not
    hypothetical, and no success criterion looked for it. This is that criterion, and the
    dtype parametrization is now PERMANENT (see `tests/test_layers/conftest.py`, which also
    guarantees the global policy is restored — a leaked policy corrupts the whole session).
    """

    @pytest.mark.parametrize("attn_self", [True, False])
    @pytest.mark.parametrize("use_mask", [False, True])
    @pytest.mark.parametrize("num_tokens", [1, 8])
    def test_no_nan_under_mixed_precision(
        self, dtype_policy, attn_self, use_mask, num_tokens
    ):
        block = _block(attn_self=attn_self, num_steps=3)
        x = keras.random.normal((BATCH, num_tokens, DIM))

        mask = None
        if use_mask:
            keep = np.ones((BATCH, num_tokens), dtype="float32")
            keep[:, -1] = 0.0            # drop the last token
            mask = keras.ops.convert_to_tensor(keep)

        out = keras.ops.convert_to_numpy(
            block(x, attention_mask=mask, training=False)
        )

        assert out.shape == (BATCH, num_tokens, DIM)
        assert np.isnan(out).sum() == 0, f"{np.isnan(out).sum()}/{out.size} NaN"
        assert np.isinf(out).sum() == 0, f"{np.isinf(out).sum()}/{out.size} Inf"
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("hopfield_activation", ACTIVATIONS)
    def test_hopfield_energy_and_update_callable_directly(
        self, dtype_policy, hopfield_activation
    ):
        """S14: `HopfieldNetwork.energy()` / `.update()` are safe OUTSIDE `__call__`.

        There is no autocast scope outside `__call__`, so before the fix a float32 `g` met
        a float16 `xi` here and the einsum raised `InvalidArgumentError`. These are the
        methods `EnergyTransformer` calls and the duck-typed convention advertises.
        """
        layer = HopfieldNetwork(
            dim=DIM, hopfield_dim=MEM, activation=hopfield_activation
        )
        g = keras.random.normal((BATCH, TOKENS, DIM))   # float32, NOT pre-cast by the caller
        layer.build(g.shape)

        e = keras.ops.convert_to_numpy(layer.energy(g))
        u = keras.ops.convert_to_numpy(layer.update(g))

        assert e.shape == (BATCH,)
        assert u.shape == (BATCH, TOKENS, DIM)
        assert np.all(np.isfinite(e)), f"non-finite energy: {e}"
        assert np.all(np.isfinite(u))

    # C13 (iter-2 / R1): see the note on `_REALISTIC_TOKENS` in `test_energy_attention.py`.
    # This guard was WRITTEN for the fp16-overflow failure mode (its docstring named it) and
    # STILL missed it, because it ran only at `TOKENS = 7` where the energy is O(-140) and an
    # overflow is arithmetically impossible. The sequence length IS the guard. Do NOT shrink
    # it. See decisions.md D-005.
    _REALISTIC_TOKENS = 1024
    _FP16_MAX = 65504.0

    @pytest.mark.parametrize("hopfield_activation", ACTIVATIONS)
    @pytest.mark.parametrize("num_tokens", [TOKENS, _REALISTIC_TOKENS])
    def test_reported_energy_is_finite_under_every_dtype(
        self, dtype_policy, hopfield_activation, num_tokens
    ):
        """C13: `return_energy=True` must survive fp16 AT A REALISTIC SEQUENCE LENGTH.

        The energy is a LARGE reduction (over heads x tokens for `E_ATT`, over tokens x
        memories for `E_HN`). RED at HEAD (before the D-005 fix): `mixed_float16`, N=1024 ->
        the whole trace is `-inf`, because the float32 reduction was cast back into fp16.
        """
        block = _block(
            hopfield_activation=hopfield_activation, num_steps=3, return_energy=True
        )
        x = keras.random.normal((BATCH, num_tokens, DIM), seed=4)

        out, energies = block(x, training=False)
        out = keras.ops.convert_to_numpy(out)
        energies = keras.ops.convert_to_numpy(energies)

        assert energies.shape == (BATCH, 4)
        assert np.all(np.isfinite(out)), f"{np.isnan(out).sum()}/{out.size} NaN in output"
        assert np.all(np.isfinite(energies)), (
            f"non-finite energies at N={num_tokens} under policy {dtype_policy!r}: "
            f"{energies}. The reported energy must stay in the reduce dtype (>= float32) "
            "and never be cast back to the compute dtype (decisions.md D-005)."
        )

        if num_tokens == self._REALISTIC_TOKENS:
            # Anti-vacuity: the cell only proves anything while |E| exceeds fp16's range.
            assert np.abs(energies).max() > self._FP16_MAX, (
                f"max|E| = {np.abs(energies).max():.1f} <= fp16 max ({self._FP16_MAX}) at "
                f"N={num_tokens} — this guard can no longer detect an fp16 cast-back."
            )

    @pytest.mark.parametrize(
        "hopfield_activation, hopfield_beta", [("relu", 1.0), ("softmax", 0.02)]
    )
    def test_hopfield_energy_is_finite_at_a_realistic_length(
        self, dtype_policy, hopfield_activation, hopfield_beta
    ):
        """C13: `HopfieldNetwork.energy()` had the IDENTICAL fp16 cast-back.

        `E_HN` sums over tokens x memories, so it clears fp16's 65504 range at N=1024 with
        excited memories (relu branch) or a small `hopfield_beta` (softmax branch: the energy
        carries a `1/beta` factor). Both cells are `-inf` at HEAD under `mixed_float16`.
        """
        hopfield = HopfieldNetwork(
            dim=DIM,
            hopfield_dim=MEM,
            activation=hopfield_activation,
            hopfield_beta=hopfield_beta,
        )
        hopfield.build((BATCH, self._REALISTIC_TOKENS, DIM))
        _excite(hopfield, scale=1.0)
        g = keras.random.normal((BATCH, self._REALISTIC_TOKENS, DIM), seed=3)

        e = keras.ops.convert_to_numpy(hopfield.energy(g))

        assert e.shape == (BATCH,)
        assert np.all(np.isfinite(e)), (
            f"non-finite E_HN under policy {dtype_policy!r}: {e} — the Hopfield energy must "
            "stay in the reduce dtype (>= float32); decisions.md D-005."
        )
        assert np.abs(e).max() > self._FP16_MAX, (
            f"max|E_HN| = {np.abs(e).max():.1f} <= fp16 max ({self._FP16_MAX}) — this guard "
            "can no longer detect an fp16 cast-back."
        )


@pytest.fixture
def global_mixed_float16():
    """Set the GLOBAL `mixed_float16` policy for one test, then ALWAYS restore it.

    Deliberately the GLOBAL policy and not a per-layer `dtype=` kwarg: this is the mainstream
    way to use mixed precision, and C16's bug (below) is only reachable that way — a leaked
    policy would poison the whole session, hence the `finally`.
    """
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield "mixed_float16"
    finally:
        keras.mixed_precision.set_global_policy(previous)


class TestEnergyDtypeInTheGraph:
    """C16 (iter-3 / R4): the energy's float32-ness must survive into the SYMBOLIC graph.

    Iter-2 (D-005) widened the energy at the LAYER boundary: `energy()` returns in the reduce
    dtype (>= float32) so an O(-1e5) trace is not `-inf` in fp16. But `EnergyTransformer`
    defined only `compute_output_shape`, so Keras' default `compute_output_spec` stamped BOTH
    outputs with `self.compute_dtype`: the symbolic `energies` KerasTensor claimed **float16**
    while `predict` returned **float32**.

    RED at HEAD (global `mixed_float16`, N=1024, `return_energy=True`):
        `model.outputs` dtypes -> `['float16', 'float16']`
        `predict`      dtypes -> `['float16', 'float32']`     <- the graph LIES

    That is the PUBLIC dtype contract — what an exporter, a dtype-aware consumer, and
    `model.summary()` read. `compute_output_spec` (decisions.md D-006) makes it true.

    NOT what this fixes (control-proven, and the reason the head below is `dtype='float32'`):
    a downstream layer autocasts its float inputs to its OWN `compute_dtype`. A default-policy
    `Dense(1)(energies)` overflows to nan/inf at N=1024 under `mixed_float16` REGARDLESS of the
    upstream symbolic dtype — reproduced with a plain float32 `keras.Input` and no ET in the
    graph at all. The energy head must be float32; see the `return_energy` docstring.
    """

    _N = 1024                    # |E| ~ 8e4 here — comfortably past fp16's 65504
    _FP16_MAX = 65504.0

    def test_symbolic_energy_dtype_is_float32(self, global_mixed_float16):
        """(a) THE assertion that bites: `model.outputs[1].dtype` — the SYMBOLIC dtype.

        (b) and the documented consumption path — a `float32` energy head — is FINITE at
        N=1024, on a trace whose magnitude provably exceeds the fp16 range.
        """
        block = _block(num_steps=2, step_size=0.05, return_energy=True)

        inp = keras.Input(shape=(self._N, DIM))
        state, energies = block(inp)
        # The DOCUMENTED way to consume the trace under a mixed policy (D-006): a float32
        # head. A default-policy head would autocast the O(-1e5) energy into fp16 by its OWN
        # policy and overflow — that is the consumer's dtype, not the block's output.
        head = keras.layers.Dense(1, dtype="float32")(energies)
        model = keras.Model(inp, [state, energies, head])

        # (a) The graph must not lie about the energy. RED at HEAD: 'float16'.
        assert model.outputs[1].dtype == "float32", (
            f"the SYMBOLIC energy dtype is {model.outputs[1].dtype!r}, but the tensor "
            "actually produced is float32 (decisions.md D-005). EnergyTransformer must "
            "override compute_output_spec — Keras' default stamps EVERY output with "
            "self.compute_dtype (decisions.md D-006)."
        )
        # ...while the STATE output stays in the compute dtype: this is mixed precision, and
        # widening the state would silently disable it.
        assert model.outputs[0].dtype == "float16", (
            f"the state output is {model.outputs[0].dtype!r}, not float16 — "
            "compute_output_spec must widen the ENERGY only, never the state"
        )

        x = np.random.default_rng(11).standard_normal(
            (1, self._N, DIM)
        ).astype("float32")
        out_state, out_energies, out_head = model.predict(x, verbose=0)

        # The concrete tensors must AGREE with the graph (that is the whole point).
        assert out_energies.dtype == np.float32
        assert out_state.dtype == np.float16

        # Anti-vacuity: without this the fp16 range is never exercised and (b) is free.
        assert np.abs(out_energies).max() > self._FP16_MAX, (
            f"max|E| = {np.abs(out_energies).max():.1f} <= fp16 max ({self._FP16_MAX}) at "
            f"N={self._N} — this guard can no longer detect an fp16 narrowing."
        )
        assert np.all(np.isfinite(out_energies)), (
            f"non-finite energy trace: {out_energies}"
        )
        # (b) the float32 energy head is FINITE on a trace that fp16 cannot represent.
        assert np.all(np.isfinite(out_head)), (
            f"a float32 head on the energy trace produced {out_head} — the trace reaching it "
            "was narrowed to fp16 somewhere (decisions.md D-006)"
        )

    def test_compute_output_spec_matches_compute_output_shape(
        self, global_mixed_float16
    ):
        """The two must never disagree: `compute_output_spec` takes its shapes FROM the shape
        method. Also covers `return_energy=False` (a single output, compute dtype).
        """
        with_energy = _block(num_steps=3, return_energy=True)
        shapes = with_energy.compute_output_shape((BATCH, TOKENS, DIM))
        specs = with_energy.compute_output_spec(
            keras.KerasTensor((BATCH, TOKENS, DIM), dtype="float16")
        )
        assert len(specs) == 2
        assert tuple(specs[0].shape) == tuple(shapes[0])
        assert tuple(specs[1].shape) == tuple(shapes[1]) == (BATCH, 4)
        assert specs[0].dtype == "float16"      # state: compute dtype
        assert specs[1].dtype == "float32"      # energy: reduce dtype (D-005/D-006)

        plain = _block(num_steps=3)
        spec = plain.compute_output_spec(
            keras.KerasTensor((BATCH, TOKENS, DIM), dtype="float16")
        )
        assert tuple(spec.shape) == (BATCH, TOKENS, DIM)
        assert spec.dtype == "float16"


class TestPerLayerDtypeKwarg:
    """C1/C2: a PER-LAYER ``dtype=`` kwarg must govern the layers the block OWNS.

    Distinct from :class:`TestDtypePolicies` above, and deliberately so. Those 9 cases set
    the GLOBAL policy — under which the sub-layers accidentally do the right thing, because
    they read that same global policy themselves. That is exactly why F-01 survived a green
    suite: `EnergyTransformer(..., dtype='float64')` built three sub-layers with NO `dtype=`
    kwarg, they silently took the global `float32`, and the block crashed at
    `x = x + self.step_size * update` with `InvalidArgumentError: cannot compute AddV2 as
    input #1 was expected to be a double tensor but is a float tensor`.

    These tests therefore NEVER call `keras.mixed_precision.set_global_policy` (see
    decisions.md D-004): the global policy stays `float32`, so `mixed_float16` here can only
    come from the layer kwarg, and there is no global state to leak into the rest of the
    session.
    """

    # `mixed_float16` -> compute float16 / variables float32; `float64` -> both float64.
    POLICIES = [("float64", "float64"), ("mixed_float16", "float16")]

    @pytest.mark.parametrize("policy_name, compute_dtype", POLICIES)
    def test_sublayers_inherit_the_block_policy(
        self, policy_name, compute_dtype, input_shape
    ):
        """(a) The 3 sub-layers' policies == the block's policy."""
        assert keras.mixed_precision.global_policy().name == "float32", (
            "this test is only meaningful with a float32 GLOBAL policy — the per-layer "
            "kwarg must be the ONLY source of the policy under test"
        )

        block = _block(dtype=policy_name, num_steps=2)
        block.build(input_shape)

        assert block.dtype_policy.name == policy_name
        assert block.compute_dtype == compute_dtype
        for name, sub in (
            ("norm", block.norm),
            ("attention", block.attention),
            ("hopfield", block.hopfield),
        ):
            assert sub.dtype_policy.name == block.dtype_policy.name, (
                f"sub-layer `{name}` has policy {sub.dtype_policy.name!r} but the block "
                f"has {block.dtype_policy.name!r} — the sub-layer ignored the block's "
                f"dtype and took the GLOBAL policy instead"
            )

    @pytest.mark.parametrize("policy_name, compute_dtype", POLICIES)
    def test_forward_pass_runs_in_the_requested_dtype(
        self, policy_name, compute_dtype, input_shape, sample_input
    ):
        """(b) The forward pass runs, returns the right dtype, and is finite."""
        block = _block(dtype=policy_name, num_steps=2)
        x = keras.ops.cast(keras.ops.convert_to_tensor(sample_input), compute_dtype)

        out = block(x, training=False)

        assert keras.backend.standardize_dtype(out.dtype) == compute_dtype
        out = keras.ops.convert_to_numpy(out)
        assert out.shape == input_shape
        assert np.all(np.isfinite(out)), f"{np.isnan(out).sum()}/{out.size} non-finite"

    @pytest.mark.parametrize("policy_name, compute_dtype", POLICIES)
    def test_gradient_step_under_per_layer_dtype(
        self, policy_name, compute_dtype, input_shape, sample_input
    ):
        """(c) A gradient step works: all 5 trainable weights get finite gradients."""
        block = _block(dtype=policy_name, num_steps=2)
        block.build(input_shape)
        x = keras.ops.cast(keras.ops.convert_to_tensor(sample_input), compute_dtype)

        with tf.GradientTape() as tape:
            y = block(x, training=True)
            loss = tf.reduce_mean(tf.square(tf.cast(y, tf.float32)))

        grads = tape.gradient(loss, block.trainable_weights)
        assert len(grads) == 5
        for g, w in zip(grads, block.trainable_weights):
            assert g is not None, f"no gradient for {w.name} under dtype={policy_name}"
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(g))), (
                f"non-finite gradient for {w.name} under dtype={policy_name}"
            )


# ---------------------------------------------------------------------
# F-02b — a KERAS-PROPAGATED mask must be honored AND FORWARDED by the BLOCK
# ---------------------------------------------------------------------

VOCAB = 10


def _padded_block_model(seed: int = 5, **block_overrides):
    """`Embedding(mask_zero=True) -> EnergyTransformer` — the standard Keras NLP idiom.

    The Embedding is the mask PRODUCER: no mask is ever hand-passed, so this exercises the
    real `_keras_mask` propagation path down to `EnergyAttention` (decisions.md D-004: a
    mask test that hand-passes an `attention_mask` cannot see F-02 at all).

    Returns `(model, block, emb_model)`, where `emb_model` maps the SAME token ids to the
    block's INPUT (the embedding output).

    C15 (iter-2 / R3): `emb_model` is not a convenience — it is the anti-vacuity instrument.
    The block's output is `embedding + sum(step_size * update)`, so `|y| > 1e-3` (the probe
    copy-pasted from the attention-level tests, where the layer's output IS its update) only
    proves the EMBEDDING is nonzero: a DEAD block passes it with pad-diff exactly 0.0. Every
    anti-vacuity check on this model must therefore measure `|y - emb(tokens)|` — what the
    BLOCK contributed. See decisions.md D-005.
    """
    emb = keras.layers.Embedding(VOCAB, DIM, mask_zero=True)
    block = _block(num_steps=3, step_size=0.05, noise_std=0.0, **block_overrides)

    inp = keras.Input(shape=(None,), dtype="int32")
    embedded = emb(inp)
    model = keras.Model(inp, block(embedded))
    emb_model = keras.Model(inp, embedded)

    # Excite BOTH the embedding and the block: with the default inits the block's output
    # displacement is O(1e-3) and a dropped mask would hide under atol=1e-6.
    rng = np.random.default_rng(seed)
    emb.embeddings.assign(rng.standard_normal((VOCAB, DIM)).astype("float32"))
    _excite_block(block)
    return model, block, emb_model


def _assert_block_moved_the_state(y, emb_out, floor: float = 1e-3) -> float:
    """C15: assert the BLOCK contributed something — never that the embedding is nonzero.

    Returns the displacement, so a caller can print it. A dead block (`step_size -> 0`, or a
    zeroed update) CANNOT satisfy this, which is the entire point (decisions.md D-005).
    """
    moved = float(np.abs(np.asarray(y) - np.asarray(emb_out)).max())
    assert moved > floor, (
        f"the block moved the state by only {moved:.3e} (<= {floor:.0e}) — it contributed "
        "nothing, so every equality assertion in this test would pass VACUOUSLY. This probe "
        "must measure |y - embedding|, NOT |y|: the block's output is `embedding + updates`, "
        "so |y| > floor merely proves the EMBEDDING is nonzero (decisions.md D-005)."
    )
    return moved


class TestKerasMaskIsHonored:
    """F-02b: PAD tokens must not influence the real tokens' outputs, at the BLOCK level.

    Step 2 fixed `EnergyAttention`. The block still has to (a) DECLARE `mask` in `call()` so
    Keras injects it, and (b) FORWARD it to `self.attention.energy/update`. Without (b) the
    fixed attention layer never sees the mask and this test stays RED.
    """

    def test_pads_do_not_shift_real_token_outputs(self):
        model, _, emb_model = _padded_block_model()

        padded = np.array([[1, 2, 3, 0, 0, 0]], dtype="int32")  # 3 real + 3 PAD
        short = np.array([[1, 2, 3]], dtype="int32")            # pads physically absent

        y_pad = keras.ops.convert_to_numpy(model(padded))[:, :3, :]
        y_short = keras.ops.convert_to_numpy(model(short))

        # C15 anti-vacuity: the BLOCK must have moved the state at the real positions. The
        # old probe here was `|y_short|.max() > 1e-3`, which a DEAD block (step_size=1e-12)
        # passes — it only proves the embedding is nonzero (decisions.md D-005).
        _assert_block_moved_the_state(
            y_short, keras.ops.convert_to_numpy(emb_model(short))
        )

        max_abs_diff = float(np.abs(y_pad - y_short).max())
        np.testing.assert_allclose(
            y_pad, y_short, atol=1e-6,
            err_msg=(
                f"PAD tokens shifted the real tokens' outputs by {max_abs_diff:.3e}. "
                "EnergyTransformer advertises supports_masking=True, but its call() must "
                "DECLARE `mask` (so Keras injects the propagated mask) AND FORWARD it to "
                "self.attention.energy/update (F-02b)."
            ),
        )

    def test_return_energy_with_a_propagated_mask(self):
        """`return_energy=True` + a Keras mask: shapes, finiteness, and NO mask on E.

        With `return_energy=True` the block outputs a TUPLE `(x, energies)`. A `(B, N)` token
        mask must NOT be attached to the `(B, T + 1)` energy tensor — Keras would then
        propagate a shape-incompatible mask downstream.
        """
        model, block, emb_model = _padded_block_model(**{"return_energy": True})

        padded = np.array([[1, 2, 3, 0, 0, 0]], dtype="int32")
        out, energies = model(padded)

        assert tuple(out.shape) == (1, 6, DIM)
        assert tuple(energies.shape) == (1, block.num_steps + 1)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(energies)))

        # The energy tensor must carry NO mask.
        assert getattr(energies, "_keras_mask", None) is None, (
            "a token mask was attached to the (B, T + 1) energy tensor — compute_mask must "
            "return [mask, None] when return_energy=True"
        )

        # ...and the mask must STILL be honored on the return_energy path (which reads the
        # energy through `self.energy()`, a SECOND forwarding site). Shapes + finiteness
        # alone would pass on a dropped mask — this assertion is what makes the guard bite.
        short = np.array([[1, 2, 3]], dtype="int32")
        y_short, _ = model(short)
        y_pad = keras.ops.convert_to_numpy(out)[:, :3, :]
        y_short = keras.ops.convert_to_numpy(y_short)

        # C15: ...and a DEAD block must not be able to satisfy that equality (it passed with
        # pad-diff exactly 0.000e+00 before this probe existed). See decisions.md D-005.
        _assert_block_moved_the_state(
            y_short, keras.ops.convert_to_numpy(emb_model(short))
        )

        max_abs_diff = float(np.abs(y_pad - y_short).max())
        np.testing.assert_allclose(
            y_pad, y_short, atol=1e-6,
            err_msg=(
                f"PAD tokens shifted the real tokens' outputs by {max_abs_diff:.3e} on the "
                "return_energy=True path — the mask is not reaching self.energy()/update()."
            ),
        )

    @pytest.mark.parametrize("num_pads", [3, 9])
    def test_block_energy_is_pad_invariant(self, num_pads):
        """C14 (iter-2 / R2): the PUBLIC energy trace must not count the PAD tokens.

        `E = E_ATT + E_HN`. `E_ATT` was already pad-invariant (the keep mask zeroes the
        masked key/query pairs), but `E_HN` summed the Hopfield energy over the PAD tokens
        too — so the trace a caller monitors drifted with the padding, and the rows of a
        variable-length batch became incomparable.

        RED at HEAD (before the D-005 fix): +74.9% with 3 pads, +224.6% with 9 pads.
        """
        model, block, _ = _padded_block_model(**{"return_energy": True})

        # `_excite_block` deliberately excites the ATTENTION hardest (that is what the F-02
        # tests need). Here the E_HN term is the one under test, so give it real weight in
        # the total energy — otherwise a pad-polluted E_HN hides inside a much larger E_ATT
        # and this guard barely bites.
        _excite(block.hopfield, scale=0.6)

        short = np.array([[1, 2, 3]], dtype="int32")
        padded = np.array([[1, 2, 3] + [0] * num_pads], dtype="int32")

        _, e_short = model(short)
        _, e_pad = model(padded)
        e_short = keras.ops.convert_to_numpy(e_short)
        e_pad = keras.ops.convert_to_numpy(e_pad)

        assert e_short.shape == (1, block.num_steps + 1)
        assert e_pad.shape == (1, block.num_steps + 1)

        # Anti-vacuity: an all-zero energy would make the comparison below meaningless...
        assert np.abs(e_short).min() > 1e-1, (
            f"the unpadded energy is ~0 ({e_short}) — the comparison would pass vacuously"
        )
        # ...and (iter-3, review NOTE 4) a nonzero-but-FLAT trace is not enough either: a DEAD
        # block (step_size -> 0) emits a constant, finite, large energy and satisfies every
        # equality below. The trace must actually MOVE. This is the same C15 hole, one level
        # down: assert the BLOCK did something, never merely that a tensor is nonzero.
        trace_motion = float(np.abs(np.diff(e_short, axis=-1)).max())
        assert trace_motion > 1e-3, (
            f"the energy trace is flat (max step = {trace_motion:.3e}) — a DEAD block would "
            "pass this test vacuously (decisions.md D-005)"
        )

        drift = float(np.abs(e_pad - e_short).max() / np.abs(e_short).max())
        print(
            f"\n[C14] pads={num_pads}: E(short)={e_short[0]}  E(padded)={e_pad[0]}  "
            f"relative drift={drift:.3%}"
        )
        np.testing.assert_allclose(
            e_pad, e_short, rtol=1e-4, atol=1e-3,
            err_msg=(
                f"{num_pads} PAD tokens shifted the reported energy by {drift:.1%}. The "
                "energy is a REDUCTION over the token axis, so BOTH terms must exclude the "
                "masked tokens — E_ATT via the keep mask and E_HN via its own token-sum mask "
                "(decisions.md D-005). Assumption A6 covers the UPDATE, not the energy."
            ),
        )

    def test_compute_mask_contract(self, sample_input):
        """`compute_mask` returns `[mask, None]` with return_energy, else the mask itself."""
        token_mask = keras.ops.convert_to_tensor(
            np.ones((BATCH, TOKENS), dtype="bool")
        )

        plain = _block(num_steps=2)
        assert plain.compute_mask(sample_input, mask=token_mask) is token_mask
        assert plain.compute_mask(sample_input, mask=None) is None

        with_energy = _block(num_steps=2, return_energy=True)
        out_mask = with_energy.compute_mask(sample_input, mask=token_mask)
        assert isinstance(out_mask, list) and len(out_mask) == 2
        assert out_mask[0] is token_mask
        assert out_mask[1] is None, (
            "the (B, T + 1) energy tensor must not carry the (B, N) token mask"
        )


class TestBlockMaskPrecedence:
    """D-003 at the BLOCK level: `mask` AND `attention_mask` compose as a LOGICAL AND.

    The AND is a claim about the ATTENTION keep mask (`_build_keep_mask`), and it is asserted
    below on the REAL tokens' outputs, where it holds EXACTLY (0.0).

    Since D-006 a **rank-2** `attention_mask` and the Keras `mask` are the SAME OBJECT — both
    are per-token VALIDITY masks — so a token hidden by EITHER is excluded from the energy sum
    and its state is not evolved: it must come out EXACTLY as it went in. Both rows are
    asserted to pass through below.

    HISTORY (do not re-break it): this test previously asserted the OPPOSITE for the
    `attention_mask` row — "an attention_mask is a key x query keep mask, not a 'this is not a
    token' mask" — which PINNED the iter-2 fork, 30 lines below a class docstring that said
    the two were identical. The per-token/pair-level distinction is real, but it lives at
    RANK 3/4, not at rank 2; it is pinned by
    `TestRank2AttentionMaskEqualsKerasMask.test_rank3_attention_mask_is_pair_level`.
    See decisions.md D-006.
    """

    def test_keras_mask_and_attention_mask_are_anded(self, sample_input):
        j_keras, j_explicit = 3, 4

        block = _block(num_steps=3, step_size=0.05, noise_std=0.0)
        block.build((BATCH, TOKENS, DIM))
        _excite_block(block)

        keras_mask = np.ones((BATCH, TOKENS), dtype="bool")
        keras_mask[:, j_keras] = False                     # Keras hides token 3

        explicit = np.ones((BATCH, TOKENS), dtype="float32")
        explicit[:, j_explicit] = 0.0                      # attention_mask hides token 4

        both = np.ones((BATCH, TOKENS), dtype="float32")
        both[:, [j_keras, j_explicit]] = 0.0               # reference: hides BOTH

        y_and = keras.ops.convert_to_numpy(
            block(sample_input, attention_mask=explicit, mask=keras_mask, training=False)
        )
        y_ref = keras.ops.convert_to_numpy(
            block(sample_input, attention_mask=both, training=False)
        )
        y_explicit_only = keras.ops.convert_to_numpy(
            block(sample_input, attention_mask=explicit, training=False)
        )

        # Anti-vacuity: hiding token 3 as well DOES change the answer, so the equality
        # below is measuring the AND and not a no-op.
        assert np.abs(y_ref - y_explicit_only).max() > 1e-3, (
            "hiding the extra token changed nothing — the AND check would pass vacuously"
        )

        # The AND, on the REAL tokens (every row but the two masked ones). Measured 0.0.
        real = [n for n in range(TOKENS) if n not in (j_keras, j_explicit)]
        max_abs_diff = float(np.abs(y_and[:, real] - y_ref[:, real]).max())
        np.testing.assert_allclose(
            y_and[:, real], y_ref[:, real], atol=1e-6,
            err_msg=(
                f"mask + attention_mask is not a logical AND (diff={max_abs_diff:.3e}). "
                "The block must forward BOTH masks to EnergyAttention, which ANDs them "
                "inside _build_keep_mask (decisions.md D-003)."
            ),
        )

        # D-005/D-006: a token hidden by EITHER mask is not a token. It contributes no energy,
        # so its gradient is zero and BOTH sub-updates are zero at its row — it must come out
        # EXACTLY as it went in. This is the block-level guard on the masked Hopfield update:
        # leave `HopfieldNetwork.update` unmasked (or forward the mask to only ONE of the two
        # mask sources) and one of these rows moves.
        for name, j in (("Keras `mask`", j_keras), ("rank-2 `attention_mask`", j_explicit)):
            passthrough = float(np.abs(y_and[:, j] - sample_input[:, j]).max())
            assert passthrough == 0.0, (
                f"the token hidden by the {name} MOVED by {passthrough:.3e}. A rank-2 mask of "
                "EITHER kind is a per-token VALIDITY mask (D-006): its energy contribution is "
                "zero, so its gradient is zero, so both EnergyAttention.update (via the keep "
                "mask) and HopfieldNetwork.update (via `_hopfield_token_mask`) must be exactly "
                "0 there (decisions.md D-005, D-006)."
            )


# ---------------------------------------------------------------------
# R5 / C17 — ONE per-token mask semantics: rank-2 `attention_mask` == Keras `mask`
# ---------------------------------------------------------------------

def _pad_probe(num_real: int = 5, num_pads: int = 3, seed: int = 3):
    """An excited block + a padded batch whose PAD rows are NONZERO.

    The nonzero PAD rows are the whole point: `Embedding(mask_zero=True)`'s id-0 row is a
    LEARNED, nonzero vector, so a PAD token has real Hopfield energy. A zero PAD row
    contributes `E_HN = 0` for free (h = xi @ 0 = 0) and would hide this bug completely.

    Returns `(block, x_padded, x_short, keep_2d, num_real)`.
    """
    rng = np.random.default_rng(seed)
    n = num_real + num_pads

    block = _block(num_steps=3, step_size=0.05, noise_std=0.0, return_energy=True)
    block.build((1, n, DIM))
    _excite_block(block)
    _excite(block.hopfield, scale=0.6)      # give E_HN real weight inside E = E_ATT + E_HN

    x_padded = rng.standard_normal((1, n, DIM)).astype("float32")
    x_short = x_padded[:, :num_real, :].copy()

    keep = np.ones((1, n), dtype="float32")
    keep[:, num_real:] = 0.0
    return block, x_padded, x_short, keep, num_real


class TestRank2AttentionMaskEqualsKerasMask:
    """C17 (iter-3 / R5): D-008's documented equivalence must be TRUE, not just documented.

    Iter-2 forwarded only the KERAS `mask` to `HopfieldNetwork.energy/update`, so the rank-2
    `attention_mask` path kept summing E_HN over the PAD tokens — while `energy_transformer`'s
    own `call()` docstring and the D-002 anchor in `energy_attention` both swore the two masks
    were the same operator. The caller most likely to hit it is exactly the one with no
    `Embedding(mask_zero=True)` upstream (an MIM model), which is FORCED to pass its validity
    mask as `attention_mask`.

    RED at HEAD (same weights, 8 tokens, 3 nonzero PAD, num_steps=3):
        E via Keras mask       -> pad-drift  0.0000%  (pad-invariant)
        E via rank-2 attn_mask -> pad-drift +27.2571% (NOT pad-invariant)
        PAD-row passthrough    -> 0.0 (Keras mask) vs 4.733 (attention_mask)

    Rank-3/rank-4 `attention_mask`s are PAIR-level (key x query) and correctly do NOT mask the
    Hopfield energy — that difference is real and is pinned by the last test below.
    """

    def test_energy_is_pad_invariant_under_a_rank2_attention_mask(self):
        """C17: the FULL energy (E_ATT + E_HN), not just E_ATT."""
        block, x_padded, x_short, keep, num_real = _pad_probe()

        _, e_ref = block(x_short, training=False)                     # pads ABSENT
        _, e_attn = block(
            x_padded, attention_mask=keras.ops.convert_to_tensor(keep), training=False
        )
        _, e_unmasked = block(x_padded, training=False)               # anti-vacuity control

        e_ref = keras.ops.convert_to_numpy(e_ref)
        e_attn = keras.ops.convert_to_numpy(e_attn)
        e_unmasked = keras.ops.convert_to_numpy(e_unmasked)

        # Anti-vacuity 1: the PAD tokens DO carry energy, so an unmasked run drifts. Without
        # this, a zero-energy PAD (or a dead E_HN) makes the equality below free.
        drift_unmasked = float(
            np.abs(e_unmasked - e_ref).max() / np.abs(e_ref).max()
        )
        assert drift_unmasked > 1e-2, (
            f"the PAD tokens contribute only {drift_unmasked:.3%} of energy — this guard "
            "cannot detect a pad-polluted E_HN. Are the PAD rows zero (h = xi @ 0 = 0)?"
        )
        # Anti-vacuity 2 (C15): the trace must actually MOVE, so a dead block goes RED here.
        assert np.abs(np.diff(e_ref, axis=-1)).max() > 1e-3, (
            f"the energy trace is flat ({e_ref}) — a DEAD block would satisfy every "
            "assertion below vacuously (decisions.md D-005)"
        )

        drift = float(np.abs(e_attn - e_ref).max() / np.abs(e_ref).max())
        print(
            f"\n[C17] E(short)={e_ref[0]}  E(padded, rank-2 attention_mask)={e_attn[0]}  "
            f"drift={drift:.4%}  (unmasked control drift={drift_unmasked:.4%})"
        )
        np.testing.assert_allclose(
            e_attn, e_ref, rtol=1e-4, atol=1e-3,
            err_msg=(
                f"a rank-2 attention_mask shifted the reported energy by {drift:.1%}. A rank-2 "
                "mask is a per-token VALIDITY mask (D-008) — identical to the Keras mask — so "
                "it must reach HopfieldNetwork.energy's token SUM too, via "
                "EnergyTransformer._hopfield_token_mask (decisions.md D-006). RED at HEAD: "
                "+27.3%."
            ),
        )

    def test_rank2_attention_mask_and_keras_mask_agree_exactly(self):
        """The two masks must be INTERCHANGEABLE: same energy, same output, same passthrough."""
        block, x_padded, _, keep, num_real = _pad_probe()

        y_attn, e_attn = block(
            x_padded, attention_mask=keras.ops.convert_to_tensor(keep), training=False
        )
        y_keras, e_keras = block(
            x_padded,
            mask=keras.ops.convert_to_tensor(keep.astype("bool")),
            training=False,
        )
        y_attn = keras.ops.convert_to_numpy(y_attn)
        y_keras = keras.ops.convert_to_numpy(y_keras)
        e_attn = keras.ops.convert_to_numpy(e_attn)
        e_keras = keras.ops.convert_to_numpy(e_keras)

        # Anti-vacuity: the block moved the REAL tokens (a dead block agrees with itself).
        assert np.abs(y_keras[:, :num_real] - x_padded[:, :num_real]).max() > 1e-3, (
            "the block did not move the real tokens — every equality here would be vacuous"
        )

        np.testing.assert_allclose(
            e_attn, e_keras, rtol=1e-5, atol=1e-4,
            err_msg=(
                "a rank-2 attention_mask and a Keras mask produced DIFFERENT energies. They "
                "are the same object (D-006/D-008); the code, the docstrings and this test "
                "must all say so."
            ),
        )
        np.testing.assert_allclose(
            y_attn, y_keras, atol=1e-6,
            err_msg="a rank-2 attention_mask and a Keras mask produced different outputs",
        )
        # ...and BOTH must freeze the PAD rows (energy contribution 0 => gradient 0).
        for name, y in (("attention_mask", y_attn), ("Keras mask", y_keras)):
            passthrough = float(
                np.abs(y[:, num_real:] - x_padded[:, num_real:]).max()
            )
            assert passthrough == 0.0, (
                f"the PAD rows MOVED by {passthrough:.3e} under the {name}. RED at HEAD for "
                "the attention_mask path: 4.733 (decisions.md D-006)."
            )

    @pytest.mark.parametrize("rank", [3, 4])
    def test_rank3_attention_mask_is_pair_level(self, rank):
        """A rank-3/rank-4 mask is a KEY x QUERY PAIR mask — it has NO per-token reading.

        It must NOT mask the Hopfield energy: it says "m may not attend to n", not "n is not a
        token" (a row can be half-masked). This is a genuine semantic difference, not a fork,
        and it is what the rank-2 equivalence above is NOT allowed to swallow.
        """
        block, x_padded, _, keep, num_real = _pad_probe()
        n = x_padded.shape[1]

        # The pair mask that a rank-2 keep EXPANDS to (D-008: symmetric, key AND query).
        pair = keep[:, :, None] * keep[:, None, :]                    # (1, N, N)
        if rank == 4:
            pair = np.repeat(pair[:, None, :, :], HEADS, axis=1)      # (1, H, N, N)

        y_pair, e_pair = block(
            x_padded, attention_mask=keras.ops.convert_to_tensor(pair), training=False
        )
        y_tok, e_tok = block(
            x_padded, attention_mask=keras.ops.convert_to_tensor(keep), training=False
        )
        e_pair = keras.ops.convert_to_numpy(e_pair)
        e_tok = keras.ops.convert_to_numpy(e_tok)

        # Same ATTENTION keep (the rank-2 mask expands to exactly this pair mask), so the two
        # differ ONLY in E_HN: the pair mask still counts the PAD tokens' Hopfield energy.
        gap = float(np.abs(e_pair - e_tok).max())
        assert gap > 1e-1, (
            f"a rank-{rank} pair mask produced the same energy as the rank-2 token mask "
            f"(gap={gap:.3e}). It must NOT mask the Hopfield energy — a pair mask has no "
            "per-token reading, and reducing one into a token keep would INVENT a semantics "
            "the caller never specified (decisions.md D-006)."
        )
        # ...and the PAD rows still evolve under a pair mask: they are still tokens.
        y_pair = keras.ops.convert_to_numpy(y_pair)
        moved = float(np.abs(y_pair[:, num_real:] - x_padded[:, num_real:]).max())
        assert moved > 1e-3, (
            f"a rank-{rank} pair mask froze the PAD rows (moved={moved:.3e}) — it was given a "
            "per-token reading it does not have (decisions.md D-006)"
        )


# ---------------------------------------------------------------------
# F-05b — the energy-descent update must be USEFUL FOR LEARNING, not merely finite
# ---------------------------------------------------------------------

class TestFitConvergence:
    """C8: `model.fit` on a token-MIXING task must actually reduce the loss.

    The gap this closes (decisions.md D-004): `test_gradient_flow` already proves the
    gradients are finite and non-None. NOTHING proved they are USEFUL — that the T-step
    energy descent computes anything a downstream task can learn from. A second finiteness
    test would be coverage theatre.

    **The design that makes this guard BITE.** A convergence test is worthless if the head
    can solve the task WITHOUT the block; it would then pass with a dead/identity block and
    certify nothing. So:

    1. **The head is mixing-free BY CONSTRUCTION.** `Dense(1)` applied to a `(B, N, D)`
       tensor acts INDEPENDENTLY on each token: token `n`'s prediction is a linear functional
       of row `n` alone. It can never see another token. (This is exactly why the head is NOT
       `GlobalAveragePooling1D` or `Flatten` — either of those performs the across-token
       mixing ITSELF and would hand a dead block the answer.)
    2. **The target REQUIRES mixing, with ZERO leakage to the own token.** The target for
       token `n` is the LEAVE-ONE-OUT mean of channel 0 over the OTHER tokens:
       `t[b, n] = scale * mean_{j != n} x[b, j, 0]`. Because the inputs are i.i.d. standard
       normal and `j != n`, `cov(t[b, n], x[b, n, :]) == 0` EXACTLY — the token's own content
       carries no linear information about its target. A dead block therefore leaves the head
       with nothing to regress on, and its loss floor is the target's own variance.
       (A plain global mean `mean_j x[b, j, 0]` is NOT good enough: it includes the own token
       with weight `1/N`, which is a real signal the head can exploit — measured live, the
       dead block then reaches ratio 0.427 and this test PASSES on a dead block.)
    3. **The batch is big enough that the head cannot memorize it.** `Dense(1)` over `D=32`
       is 33 parameters; with `B=16` it overfits the `16 * 7 = 112` fixed examples and again
       passes on a dead block. `B=64` (448 examples) starves that.

    Measured (guard-bites ledger, plan C10): with `x = x + step_size * update` the loss goes
    5.68 -> 0.028 (ratio 0.005). With the descent neutered to `x = x + 0.0 * update` the
    dead block's loss floor is **4.12** (ratio 0.726) — above the `0.5 * initial` threshold,
    so the guard goes RED. Do not "simplify" the target or shrink the batch: both changes
    make this test pass on a dead block.
    """

    FIT_BATCH = 64
    TARGET_SCALE = 5.0
    EPOCHS = 150

    @staticmethod
    def _leave_one_out_target(x: np.ndarray, scale: float) -> np.ndarray:
        """`t[b, n] = scale * mean_{j != n} x[b, j, 0]` — pure across-token structure."""
        c0 = x[:, :, 0]                                          # (B, N)
        total = c0.sum(axis=1, keepdims=True)                    # (B, 1)
        loo = (total - c0) / (c0.shape[1] - 1)                   # (B, N), excludes self
        return (scale * loo)[..., None].astype("float32")        # (B, N, 1)

    def test_fit_reduces_loss(self):
        keras.utils.set_random_seed(1234)

        rng = np.random.default_rng(0)
        x = rng.standard_normal((self.FIT_BATCH, TOKENS, DIM)).astype("float32")
        y = self._leave_one_out_target(x, self.TARGET_SCALE)

        block = _block(num_steps=3, step_size=0.05, noise_std=0.0)
        inputs = keras.Input(shape=(TOKENS, DIM))
        # Strictly per-token head — see the class docstring, point 1.
        model = keras.Model(inputs, keras.layers.Dense(1)(block(inputs)))
        model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")

        initial_loss = float(model.evaluate(x, y, verbose=0))
        model.fit(
            x, y,
            epochs=self.EPOCHS,
            batch_size=self.FIT_BATCH,
            verbose=0,
        )
        final_loss = float(model.evaluate(x, y, verbose=0))

        print(
            f"\n[C8 fit] initial={initial_loss:.4f} final={final_loss:.4f} "
            f"ratio={final_loss / initial_loss:.4f} "
            f"(dead-block floor, measured: 4.12 / ratio 0.726)"
        )

        assert np.isfinite(initial_loss) and np.isfinite(final_loss)

        # ANTI-DEGENERACY: a task whose initial loss is already ~0 proves nothing.
        assert initial_loss > 1e-1, (
            f"the task is trivial at init (loss = {initial_loss:.6e}); the convergence "
            "assertion below would pass vacuously."
        )

        assert final_loss <= 0.5 * initial_loss, (
            f"the loss did not halve ({initial_loss:.4f} -> {final_loss:.4f}). The block's "
            "energy-descent update is not usefully learnable: the head is per-token and the "
            "target is a LEAVE-ONE-OUT across-token mean, so ONLY the block can supply the "
            "token mixing this task needs. A dead block (x = x + 0.0 * update) floors at "
            "~4.12 here — if you are seeing a number near that, the descent is not moving x."
        )


class TestGraphSafeTrainingFlag:
    """G-02: `training` as a SYMBOLIC tf bool tensor inside a `tf.function`.

    At HEAD `call()` did `add_noise = self.noise_std > 0.0 and training`. Python `and`
    calls `Tensor.__bool__()`, which is illegal in graph mode -> the layer raised
    `OperatorNotAllowedInGraphError` the moment a caller wrapped it in a `tf.function`
    and passed a traced `training`. `fit`/`predict`/`jit_compile` never hit it (Keras
    resolves `training` to a Python bool for those), which is exactly why 247 tests
    missed it. See decisions.md D-003.

    THREE guards, because a "graph-safe" branch that simply never adds the noise would
    pass a compile-only guard BOTH WAYS -- the bug fixed, the FEATURE dead, the suite
    green. This feature has already shipped four guards that passed both ways.
    """

    # Realistic N (LESSONS [I:5]): the reductions in `energy()` are exercised at N=256,
    # never the N=7 fixture default.
    N_REAL = 256
    NOISE = 0.15

    def _fn(self, block):
        """The layer inside a `tf.function`, with `training` traced as a bool tensor."""
        @tf.function
        def run(x, training):
            return block(x, training=training)
        return run

    def _x(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        return tf.constant(
            rng.standard_normal((2, self.N_REAL, DIM)).astype("float32")
        )

    # -- guard 1: it must COMPILE -------------------------------------

    def test_tensor_training_in_tf_function_does_not_raise(self):
        """RED at HEAD: OperatorNotAllowedInGraphError. GREEN after D-003."""
        block = _block(noise_std=self.NOISE, seed=11)
        run = self._fn(block)
        y = run(self._x(), tf.constant(True))
        assert y.shape == (2, self.N_REAL, DIM)
        assert np.all(np.isfinite(y.numpy()))

    # -- guard 2: THE ONE THAT MATTERS (the feature must stay ALIVE) ---

    def test_tensor_training_noise_semantics(self):
        """`noise_std > 0`: a True tensor CHANGES the output, a False tensor does NOT.

        DEAD-COMPONENT injection for this guard: force `noise_std` to 0 inside the
        noise branch of `call()` (keeping the code shape). This assertion MUST go RED.
        If it does not, the guard is worthless -- rewrite it, do not ship it.
        """
        block = _block(noise_std=self.NOISE, seed=11)
        run = self._fn(block)
        x = self._x()

        y_true = run(x, tf.constant(True)).numpy()
        y_false = run(x, tf.constant(False)).numpy()
        y_py_false = block(x, training=False).numpy()

        # A False TENSOR must behave EXACTLY like a Python False: bit-identical.
        np.testing.assert_allclose(y_false, y_py_false, rtol=0, atol=0)

        # A True TENSOR must actually inject the eq. 27 noise. The scale is not
        # incidental: with num_steps=4 at noise_std=0.15 the accumulated perturbation is
        # ~1e-1, i.e. far above any fp32 reassociation noise.
        delta = float(np.max(np.abs(y_true - y_false)))
        assert delta > 1e-3, (
            f"training=tf.constant(True) did not change the output (max|diff| = {delta:.3e}) "
            f"at noise_std={self.NOISE}. The graph-safe gate compiles but the eq. 27 noise "
            "is DEAD."
        )

    # -- guard 2b: the same semantics under `mixed_float16` --------------

    def test_tensor_training_noise_semantics_mixed_float16(self, global_mixed_float16):
        """The gate must stay ALIVE and FINITE under `mixed_float16`.

        The G-02 gate was the only new runtime code this pass added and it shipped with
        zero fp16 coverage -- on a feature where FOUR consecutive defects lived at the
        dtype boundary (the `-1e9` mask bias, the `logsumexp`, the energy dtype, the
        `gamma` floor). `ops.where(cast(training, bool), x + noise, x)` runs in the
        compute dtype, i.e. float16 here, where `noise_std * sqrt(step_size)` is small
        enough to be a plausible flush-to-zero candidate. It is not -- but that must be
        GUARDED, not merely believed.
        """
        block = _block(noise_std=self.NOISE, seed=11)
        run = self._fn(block)
        x = tf.cast(self._x(), "float16")

        y_true = run(x, tf.constant(True))
        y_false = run(x, tf.constant(False))

        assert y_true.dtype == tf.float16
        assert np.all(np.isfinite(y_true.numpy())), "fp16 noise gate produced NaN/Inf"
        assert np.all(np.isfinite(y_false.numpy()))

        # A False TENSOR is still bit-identical to a Python False.
        np.testing.assert_allclose(
            y_false.numpy(), block(x, training=False).numpy(), rtol=0, atol=0
        )

        # And the noise is still ALIVE: it did not flush to zero in float16.
        delta = float(np.max(np.abs(
            y_true.numpy().astype("float32") - y_false.numpy().astype("float32")
        )))
        assert delta > 1e-3, (
            f"under mixed_float16 the eq. 27 noise is DEAD (max|True-False| = {delta:.3e} "
            f"at noise_std={self.NOISE})."
        )

    # -- guard 3: I7 -- the DEFAULT path stays trace-time-eliminated ---

    def _op_types(self, block, x, training):
        graph = self._fn(block).get_concrete_function(
            tf.TensorSpec(x.shape, x.dtype), tf.TensorSpec([], tf.bool)
        ).graph
        return [op.type for op in graph.get_operations()]

    def test_default_path_builds_no_noise_and_no_gate(self):
        """`noise_std = 0.0` (the DEFAULT): identical output AND no noise/cond in the graph.

        The Python `if self.noise_std > 0.0:` runs FIRST, so the default path must never
        construct the RNG op -- and therefore never a gate over it either (I7). The
        default-path assertion is on RANDOM ops, deliberately: a random op appears IF AND
        ONLY IF the noise branch was traced, whereas `_build_keep_mask` emits `SelectV2`
        unconditionally. `SelectV2` is only ever asserted on as a COUNT DELTA below (a
        membership check on it passes both ways and can never fail).
        """
        x = self._x()

        default = _block(noise_std=0.0, seed=11)          # the DEFAULT
        run = self._fn(default)
        y_t = run(x, tf.constant(True)).numpy()
        y_f = run(x, tf.constant(False)).numpy()
        np.testing.assert_allclose(y_t, y_f, rtol=0, atol=0)

        default_ops = self._op_types(_block(noise_std=0.0, seed=11), x, True)
        rand_default = [t for t in default_ops if "Random" in t]
        assert rand_default == [], (
            f"the DEFAULT (noise_std=0) path traced random ops {rand_default} -- the noise "
            "branch was NOT eliminated at trace time (I7)."
        )

        # POSITIVE CONTROL: the same probe MUST see the ops when noise IS enabled, or the
        # assertion above passes for the wrong reason.
        noisy_ops = self._op_types(_block(noise_std=self.NOISE, seed=11), x, True)
        rand_noisy = [t for t in noisy_ops if "Random" in t]
        assert rand_noisy, (
            "the probe found no random op even at noise_std>0 -- it cannot see what it "
            "claims the default path lacks."
        )

        # The tensor GATE itself, as a COUNT DELTA -- never a membership check.
        # `"SelectV2" in noisy_ops` was shipped here once and is VACUOUS: `_build_keep_mask`
        # emits `SelectV2` unconditionally (one per descent step), so membership is true at
        # noise_std=0 too and the assertion can NEVER fail. The delta discriminates, and it
        # is a structural law rather than a magic number -- the gate is exactly ONE
        # `ops.where` per descent step, so it adds exactly `num_steps` of them. Measured on
        # this machine at num_steps=1/2/4/8/12: default = T, noisy = 2T, delta = +T, exactly.
        n_steps = default.num_steps
        sel_default = default_ops.count("SelectV2")
        sel_noisy = noisy_ops.count("SelectV2")
        assert sel_noisy == sel_default + n_steps, (
            f"the noise gate is not in the graph: SelectV2 went {sel_default} -> {sel_noisy} "
            f"(expected +{n_steps}, one `ops.where` per descent step) when noise_std went "
            f"0 -> {self.NOISE}."
        )


# ---------------------------------------------------------------------
# Branch A: WeightedAdjacencyProjector + use_weighted_adjacency
# ---------------------------------------------------------------------

# The step-2 gate is BYTE-IDENTICAL default-off + serialization of the trainable projector.
# A realistic graph length is used (N >= 64) rather than the toy N=7 of the shared fixtures:
# LESSONS records an fp16 reduction bug that a toy N hid, and the projector's D^2-channel conv
# is precisely the kind of size-sensitive op that must be exercised at a real length.
_WA_N = 64
_WA_DIM = 24
_WA_HEADS = 4
_WA_HEAD_DIM = 6
_WA_MEM = 16


def _wa_inputs(dim: int = _WA_DIM, n: int = _WA_N, batch: int = 2, seed: int = 3):
    """Random tokens ``(B, N, dim)`` and a binary rank-3 adjacency ``(B, N, N)``."""
    x = np.asarray(keras.random.normal((batch, n, dim), seed=seed)).astype("float32")
    adj = (
        np.asarray(keras.random.uniform((batch, n, n), seed=seed + 1)) > 0.5
    ).astype("float32")
    return x, adj


def _wa_block(**overrides) -> EnergyTransformer:
    kwargs = dict(
        embed_dim=_WA_DIM, num_heads=_WA_HEADS, head_dim=_WA_HEAD_DIM,
        hopfield_dim=_WA_MEM, num_steps=3, step_size=0.05,
    )
    kwargs.update(overrides)
    return EnergyTransformer(**kwargs)


class TestWeightedAdjacencyProjector:
    """The projector in isolation: shape, finiteness, the `⊙ A'` zeroing, serialization."""

    def test_output_shape_and_finiteness(self):
        proj = WeightedAdjacencyProjector(num_heads=_WA_HEADS, embed_dim=_WA_DIM)
        x, adj = _wa_inputs()
        w_hat = proj((x, adj))
        assert tuple(w_hat.shape) == (2, _WA_HEADS, _WA_N, _WA_N)
        w = keras.ops.convert_to_numpy(w_hat)
        assert np.all(np.isfinite(w)), "Ŵ must be FINITE (the -inf lives in the keep-bias)."

    def test_non_edges_are_zeroed(self):
        """`⊙ A'`: every non-edge pair carries EXACTLY zero weight in every head."""
        proj = WeightedAdjacencyProjector(num_heads=_WA_HEADS, embed_dim=_WA_DIM)
        x, adj = _wa_inputs()
        w = keras.ops.convert_to_numpy(proj((x, adj)))       # (B, H, N, N)
        non_edge = adj[:, None, :, :] == 0.0                 # (B, 1, N, N) broadcast
        non_edge = np.broadcast_to(non_edge, w.shape)
        assert np.all(w[non_edge] == 0.0), "Ŵ is non-zero on a non-edge (A' not applied)."

    def test_proj_dim_shrinks_the_conv_input_channels(self):
        """`proj_dim` cuts the conv's input channels from D^2 to proj_dim^2 (OOM hatch)."""
        full = WeightedAdjacencyProjector(num_heads=_WA_HEADS, embed_dim=_WA_DIM)
        reduced = WeightedAdjacencyProjector(
            num_heads=_WA_HEADS, embed_dim=_WA_DIM, proj_dim=8
        )
        x, adj = _wa_inputs()
        _ = full((x, adj)); _ = reduced((x, adj))
        # conv kernel is (kh, kw, in_channels, filters)
        assert full.conv.kernel.shape[2] == _WA_DIM * _WA_DIM
        assert reduced.conv.kernel.shape[2] == 8 * 8
        assert reduced.token_proj is not None and full.token_proj is None

    def test_serialization_cycle(self):
        proj = WeightedAdjacencyProjector(
            num_heads=_WA_HEADS, embed_dim=_WA_DIM, kernel_size=3, proj_dim=8
        )
        cfg = proj.get_config()
        for key in ("num_heads", "embed_dim", "kernel_size", "proj_dim"):
            assert key in cfg, f"{key} missing from WeightedAdjacencyProjector.get_config()"
        rebuilt = WeightedAdjacencyProjector.from_config(cfg)
        assert rebuilt.get_config() == cfg


class TestUseWeightedAdjacencyFlag:
    """Falsification #2: default-off is BYTE-IDENTICAL; flag-on is a real, serializable path."""

    def test_default_off_is_byte_identical_to_no_flag(self):
        """S C2: `use_weighted_adjacency=False` == omitting the kwarg, to the last bit."""
        x, _ = _wa_inputs()
        no_flag = _wa_block()
        off = _wa_block(use_weighted_adjacency=False)
        off.build((None, _WA_N, _WA_DIM))
        no_flag.build((None, _WA_N, _WA_DIM))
        off.set_weights(no_flag.get_weights())

        y0 = keras.ops.convert_to_numpy(no_flag(x, training=False))
        y1 = keras.ops.convert_to_numpy(off(x, training=False))
        np.testing.assert_array_equal(
            y0, y1,
            err_msg="the flag-OFF path is not byte-identical to omitting the kwarg.",
        )
        # No projector, no extra weights.
        assert no_flag.adjacency_projector is None
        assert off.adjacency_projector is None
        assert len(off.weights) == len(no_flag.weights)

    def test_flag_off_leaves_the_existing_default_untouched(self):
        """The shared fixture path (no adjacency) is unchanged with the flag off."""
        block = _wa_block(use_weighted_adjacency=False)
        x, _ = _wa_inputs()
        y = keras.ops.convert_to_numpy(block(x, training=False))
        assert np.all(np.isfinite(y))

    def test_flag_on_smoke_projector_is_trainable_and_reaches_attention(self):
        """Flag-on: finite output, trainable projector weights, and Ŵ actually moves the state.

        The `Ŵ reaches attention` proof is a CONTROLLED comparison: a flag-on and a flag-off
        block that share IDENTICAL norm/attention/hopfield weights differ ONLY in whether Ŵ
        is threaded into the attention score. A non-trivial output difference is therefore
        attributable to Ŵ alone (a dropped forwarding site would make them equal).
        """
        x, adj = _wa_inputs()
        on = _wa_block(use_weighted_adjacency=True)
        on.build((None, _WA_N, _WA_DIM))

        y_on = keras.ops.convert_to_numpy(on(x, attention_mask=adj, training=False))
        assert np.all(np.isfinite(y_on)), "flag-on output is not finite."

        # Trainable projector weights exist.
        assert on.adjacency_projector is not None
        tw = on.adjacency_projector.trainable_weights
        assert len(tw) >= 2 and all(w.trainable for w in tw), (
            "the projector has no trainable weights."
        )

        # Controlled Ŵ-reaches-attention: share the non-projector weights with a flag-off twin.
        off = _wa_block(use_weighted_adjacency=False)
        off.build((None, _WA_N, _WA_DIM))
        off.norm.set_weights(on.norm.get_weights())
        off.attention.set_weights(on.attention.get_weights())
        off.hopfield.set_weights(on.hopfield.get_weights())
        y_off = keras.ops.convert_to_numpy(off(x, attention_mask=adj, training=False))

        assert np.max(np.abs(y_on - y_off)) > 1e-5, (
            "flag-on output equals the flag-off output on the same tokens+adjacency: Ŵ never "
            "reached the attention (a forwarding site was missed)."
        )

    def test_rank2_and_missing_adjacency_leave_the_weighted_path_inert(self):
        """With no pair-level adjacency there is no A', so Ŵ is None (path inert, not crashing)."""
        on = _wa_block(use_weighted_adjacency=True)
        assert on._binary_adjacency(None) is None
        rank2 = np.ones((2, _WA_N), dtype="float32")
        assert on._binary_adjacency(rank2) is None
        # A rank-4 mask reduces to (B, N, N).
        rank4 = np.ones((2, _WA_HEADS, _WA_N, _WA_N), dtype="float32")
        assert tuple(on._binary_adjacency(rank4).shape) == (2, _WA_N, _WA_N)

    def test_keras_round_trip_preserves_projector_weights(self):
        """The lazy-serialization trap: the trainable projector must survive .keras save/load."""
        x, adj = _wa_inputs()
        inp = keras.Input(shape=(_WA_N, _WA_DIM))
        amask = keras.Input(shape=(_WA_N, _WA_N))
        out = _wa_block(use_weighted_adjacency=True, adjacency_proj_dim=8)(
            inp, attention_mask=amask
        )
        model = keras.Model([inp, amask], out)
        y_before = keras.ops.convert_to_numpy(model([x, adj], training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weighted_et.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            y_after = keras.ops.convert_to_numpy(loaded([x, adj], training=False))

        assert len(loaded.weights) == len(model.weights), (
            "projector weights were dropped on the .keras round-trip (lazy build)."
        )
        np.testing.assert_allclose(y_before, y_after, rtol=1e-6, atol=1e-6)

    def test_get_config_carries_the_three_new_keys(self):
        block = _wa_block(
            use_weighted_adjacency=True, adjacency_kernel_size=3, adjacency_proj_dim=8
        )
        cfg = block.get_config()
        assert cfg["use_weighted_adjacency"] is True
        assert cfg["adjacency_kernel_size"] == 3
        assert cfg["adjacency_proj_dim"] == 8
        rebuilt = EnergyTransformer.from_config(cfg)
        assert rebuilt.get_config() == cfg

    @pytest.mark.parametrize("kwargs,match", [
        (dict(adjacency_kernel_size=0), "adjacency_kernel_size"),
        (dict(adjacency_kernel_size=-1), "adjacency_kernel_size"),
        (dict(adjacency_proj_dim=0), "adjacency_proj_dim"),
        (dict(adjacency_proj_dim=-4), "adjacency_proj_dim"),
    ])
    def test_invalid_adjacency_config(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _wa_block(use_weighted_adjacency=True, **kwargs)


if __name__ == "__main__":
    pytest.main([__file__])
