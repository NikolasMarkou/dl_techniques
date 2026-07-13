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

        # Anti-vacuity: the perturbation was real and DID move the masked row itself
        # (a masked token still receives its own per-token Hopfield update).
        assert np.abs(y1[:, j, :] - y0[:, j, :]).max() > 1e-3, (
            "the perturbation changed nothing at all — the comparison above would pass "
            "vacuously"
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

    @pytest.mark.parametrize("hopfield_activation", ACTIVATIONS)
    def test_reported_energy_is_finite_under_every_dtype(
        self, dtype_policy, hopfield_activation
    ):
        """`return_energy=True` must also survive fp16 — the energy is a LARGE reduction
        (a sum over tokens x memories) and would overflow an fp16 accumulator."""
        block = _block(
            hopfield_activation=hopfield_activation, num_steps=3, return_energy=True
        )
        x = keras.random.normal((BATCH, TOKENS, DIM))

        out, energies = block(x, training=False)
        out = keras.ops.convert_to_numpy(out)
        energies = keras.ops.convert_to_numpy(energies)

        assert energies.shape == (BATCH, 4)
        assert np.all(np.isfinite(out)), f"{np.isnan(out).sum()}/{out.size} NaN in output"
        assert np.all(np.isfinite(energies)), f"non-finite energies: {energies}"


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

    Returns `(model, block)`.
    """
    emb = keras.layers.Embedding(VOCAB, DIM, mask_zero=True)
    block = _block(num_steps=3, step_size=0.05, noise_std=0.0, **block_overrides)

    inp = keras.Input(shape=(None,), dtype="int32")
    model = keras.Model(inp, block(emb(inp)))

    # Excite BOTH the embedding and the block: with the default inits the block's output
    # displacement is O(1e-3) and a dropped mask would hide under atol=1e-6.
    rng = np.random.default_rng(seed)
    emb.embeddings.assign(rng.standard_normal((VOCAB, DIM)).astype("float32"))
    _excite_block(block)
    return model, block


class TestKerasMaskIsHonored:
    """F-02b: PAD tokens must not influence the real tokens' outputs, at the BLOCK level.

    Step 2 fixed `EnergyAttention`. The block still has to (a) DECLARE `mask` in `call()` so
    Keras injects it, and (b) FORWARD it to `self.attention.energy/update`. Without (b) the
    fixed attention layer never sees the mask and this test stays RED.
    """

    def test_pads_do_not_shift_real_token_outputs(self):
        model, _ = _padded_block_model()

        padded = np.array([[1, 2, 3, 0, 0, 0]], dtype="int32")  # 3 real + 3 PAD
        short = np.array([[1, 2, 3]], dtype="int32")            # pads physically absent

        y_pad = keras.ops.convert_to_numpy(model(padded))[:, :3, :]
        y_short = keras.ops.convert_to_numpy(model(short))

        # Anti-vacuity: the block must actually be doing something at the real positions.
        assert np.abs(y_short).max() > 1e-3, (
            "the un-padded output is ~0 — the comparison below would pass vacuously"
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
        model, block = _padded_block_model(**{"return_energy": True})

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
        y_short, _ = model(np.array([[1, 2, 3]], dtype="int32"))
        y_pad = keras.ops.convert_to_numpy(out)[:, :3, :]
        y_short = keras.ops.convert_to_numpy(y_short)
        max_abs_diff = float(np.abs(y_pad - y_short).max())
        np.testing.assert_allclose(
            y_pad, y_short, atol=1e-6,
            err_msg=(
                f"PAD tokens shifted the real tokens' outputs by {max_abs_diff:.3e} on the "
                "return_energy=True path — the mask is not reaching self.energy()/update()."
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
    """D-003 at the BLOCK level: `mask` AND `attention_mask` compose as a LOGICAL AND."""

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

        max_abs_diff = float(np.abs(y_and - y_ref).max())
        np.testing.assert_allclose(
            y_and, y_ref, atol=1e-6,
            err_msg=(
                f"mask + attention_mask is not a logical AND (diff={max_abs_diff:.3e}). "
                "The block must forward BOTH masks to EnergyAttention, which ANDs them "
                "inside _build_keep_mask (decisions.md D-003)."
            ),
        )


if __name__ == "__main__":
    pytest.main([__file__])
