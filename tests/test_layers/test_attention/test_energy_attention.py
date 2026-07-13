"""
Test suite for the EnergyAttention layer (layers/attention/energy_attention.py).

EnergyAttention is the Energy Transformer's attention (arXiv:2302.07253 eq. 3-4). It has
**no value matrix**: it defines a scalar energy ``E_ATT(g)`` and its ``update()`` returns
the hand-coded closed-form ``-dE_ATT/dg``.

``test_gradient_oracle`` is THE headline test (plan plan_2026-07-13_57c9833e, success
criterion S6a): it is the ONLY thing that proves the hand-coded two-term gradient is
actually the negative gradient of the reported energy. Deleting the second term
(``term_k``) leaves a layer that runs, trains, and is wrong; this test is what catches it.
``tf.GradientTape`` is the oracle and is legitimate HERE — it is forbidden only in
``src/`` (decisions.md D-001).

Also covers S9 (attn_self=False really excludes the diagonal) and S8a (a masked-out token
has zero influence on other tokens), plus the standard layer contract.

Run:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \\
        tests/test_layers/test_attention/test_energy_attention.py -q
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

from dl_techniques.layers.attention.energy_attention import EnergyAttention
from dl_techniques.layers.attention.factory import create_attention_layer


# ---------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------

BATCH, TOKENS, DIM, HEADS = 3, 7, 32, 4


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
    return EnergyAttention(dim=DIM, num_heads=HEADS)


def _excite(layer: EnergyAttention, scale: float = 0.5, seed: int = 7) -> None:
    """Replace the N(0, 0.02) init with visibly-sized weights.

    With the paper's default init the update is O(1e-3) and a broken gradient could match
    a correct one to within `atol`. The oracle asserts a non-trivial magnitude, so the
    weights must actually excite the layer.
    """
    rng = np.random.default_rng(seed)
    shape = tuple(layer.w_key.shape)
    layer.w_key.assign((rng.standard_normal(shape) * scale).astype("float32"))
    layer.w_query.assign((rng.standard_normal(shape) * scale).astype("float32"))


def _keep_mask(kind, seed=11):
    """Build a KEEP mask (1 = attend, 0 = masked) of the requested shape, or None."""
    if kind is None:
        return None
    rng = np.random.default_rng(seed)
    if kind == "BN":
        m = (rng.random((BATCH, TOKENS)) > 0.3).astype("float32")
        m[:, 0] = 1.0  # guarantee at least one valid key per sample
        return m
    if kind == "BNN":
        m = (rng.random((BATCH, TOKENS, TOKENS)) > 0.3).astype("float32")
        m[:, 0, :] = 1.0  # key 0 is always valid -> no fully-masked query column
        return m
    raise AssertionError(f"unknown mask kind {kind}")


def _bool_token_mask(seed=21):
    """A BOOLEAN (B, N) per-token validity mask — the shape/dtype Keras propagates."""
    rng = np.random.default_rng(seed)
    m = rng.random((BATCH, TOKENS)) > 0.3
    m[:, 0] = True  # guarantee at least one valid key per sample
    return m


def _oracle_masks(kind):
    """Resolve an oracle `mask_kind` into `(attention_mask, keras_mask)`.

    The `KERAS` / `BN+KERAS` cells exist to satisfy plan STOP-IF 2: the Keras mask enters
    BOTH `energy()`'s logsumexp/col_valid path AND `update()`'s `omega`. A merge landing in
    only one of them leaves `update() != -dE/dg` — the layer still runs, still trains, and
    the descent guarantee silently evaporates. Only the autodiff oracle can see that, and
    only if it is RUN with a Keras mask present.
    """
    if kind is None:
        return None, None
    if kind in ("BN", "BNN"):
        return tf.convert_to_tensor(_keep_mask(kind)), None
    if kind == "KERAS":
        return None, tf.convert_to_tensor(_bool_token_mask())
    if kind == "BN+KERAS":
        return (
            tf.convert_to_tensor(_keep_mask("BN")),
            tf.convert_to_tensor(_bool_token_mask()),
        )
    raise AssertionError(f"unknown mask kind {kind}")


# ---------------------------------------------------------------------
# S6a — THE GRADIENT ORACLE (the go/no-go for the whole plan)
# ---------------------------------------------------------------------

class TestGradientOracle:
    """update(g) MUST equal -d/dg [ sum_b energy(g) ] (S6a)."""

    @pytest.mark.parametrize("attn_self", [True, False])
    @pytest.mark.parametrize("mask_kind", [None, "BN", "BNN", "KERAS", "BN+KERAS"])
    def test_gradient_oracle(self, attn_self, mask_kind, sample_input):
        layer = EnergyAttention(dim=DIM, num_heads=HEADS, attn_self=attn_self)
        layer.build((BATCH, TOKENS, DIM))
        _excite(layer)

        mask, keras_mask = _oracle_masks(mask_kind)

        g = tf.Variable(sample_input, dtype=tf.float32)

        with tf.GradientTape() as tape:
            energy = tf.reduce_sum(
                layer.energy(g, attention_mask=mask, mask=keras_mask)
            )
        grad = tape.gradient(energy, g)

        assert grad is not None, "energy() has no gradient path to g"

        update = keras.ops.convert_to_numpy(
            layer.update(g, attention_mask=mask, mask=keras_mask)
        )
        neg_grad = grad.numpy()  # tape.gradient(E, g) == +dE/dg; update must be its negation

        # Anti-vacuity: a zero-vs-zero comparison must not be able to pass.
        assert np.abs(update).max() > 1e-3, (
            f"update is ~0 (max |update| = {np.abs(update).max():.3e}) — the oracle would "
            "pass vacuously. Excite the weights."
        )
        assert np.all(np.isfinite(update))
        assert np.all(np.isfinite(neg_grad))

        max_abs_err = float(np.abs(update - (-neg_grad)).max())
        np.testing.assert_allclose(
            update, -neg_grad, rtol=1e-4, atol=1e-5,
            err_msg=(
                f"update() != -dE/dg (attn_self={attn_self}, mask={mask_kind}); "
                f"max-abs-error={max_abs_err:.3e}. The closed-form gradient in "
                "EnergyAttention.update() is WRONG — do NOT 'fix' energy() to match it."
            ),
        )


# ---------------------------------------------------------------------
# S9 — attn_self=False actually excludes the diagonal
# ---------------------------------------------------------------------

class TestSelfAttentionExclusion:

    def test_attn_self_excludes_diagonal(self):
        """N == 1 with attn_self=False: every entry of A is masked -> energy and update
        are EXACTLY 0 and finite (no NaN from a -inf logsumexp)."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal((BATCH, 1, DIM)).astype("float32")

        excluded = EnergyAttention(dim=DIM, num_heads=HEADS, attn_self=False)
        excluded.build((BATCH, 1, DIM))
        _excite(excluded)

        e = keras.ops.convert_to_numpy(excluded.energy(x))
        u = keras.ops.convert_to_numpy(excluded.update(x))

        assert np.all(np.isfinite(e)), f"energy is not finite: {e}"
        assert np.all(np.isfinite(u)), "update is not finite (NaN from a -inf logsumexp?)"
        np.testing.assert_array_equal(e, np.zeros((BATCH,), dtype="float32"))
        np.testing.assert_array_equal(u, np.zeros((BATCH, 1, DIM), dtype="float32"))

        # Contrast: with attn_self=True the single token attends to itself.
        included = EnergyAttention(dim=DIM, num_heads=HEADS, attn_self=True)
        included.build((BATCH, 1, DIM))
        _excite(included)

        e_in = keras.ops.convert_to_numpy(included.energy(x))
        u_in = keras.ops.convert_to_numpy(included.update(x))
        assert np.all(np.isfinite(e_in)) and np.all(np.isfinite(u_in))
        assert np.abs(u_in).max() > 1e-3, (
            "attn_self=True on a single token must produce a NON-zero update"
        )

    def test_no_nan_on_fully_masked_input(self, sample_input):
        """An all-zero KEEP mask -> zero energy, zero update, no NaN."""
        layer = EnergyAttention(dim=DIM, num_heads=HEADS, attn_self=True)
        layer.build((BATCH, TOKENS, DIM))
        _excite(layer)

        mask = np.zeros((BATCH, TOKENS), dtype="float32")
        e = keras.ops.convert_to_numpy(layer.energy(sample_input, attention_mask=mask))
        u = keras.ops.convert_to_numpy(layer.update(sample_input, attention_mask=mask))

        assert np.all(np.isfinite(e)) and np.all(np.isfinite(u))
        np.testing.assert_array_equal(e, np.zeros((BATCH,), dtype="float32"))
        np.testing.assert_array_equal(u, np.zeros((BATCH, TOKENS, DIM), dtype="float32"))


# ---------------------------------------------------------------------
# S8a — a masked-out token has zero influence on the other tokens
# ---------------------------------------------------------------------

class TestMasking:

    def test_masked_token_has_no_influence(self, sample_input):
        """With a (B, N) KEEP mask zeroing token j, perturbing g[:, j, :] must leave
        update[:, i, :] bit-stable for every i != j (S8a)."""
        j = 2
        layer = EnergyAttention(dim=DIM, num_heads=HEADS, attn_self=False)
        layer.build((BATCH, TOKENS, DIM))
        _excite(layer)

        mask = np.ones((BATCH, TOKENS), dtype="float32")
        mask[:, j] = 0.0  # token j is invalid -> masked in BOTH roles (D-008)

        x0 = sample_input.copy()
        x1 = sample_input.copy()
        rng = np.random.default_rng(99)
        x1[:, j, :] += rng.standard_normal((BATCH, DIM)).astype("float32") * 3.0

        u0 = keras.ops.convert_to_numpy(layer.update(x0, attention_mask=mask))
        u1 = keras.ops.convert_to_numpy(layer.update(x1, attention_mask=mask))

        others = [i for i in range(TOKENS) if i != j]
        np.testing.assert_allclose(
            u0[:, others, :], u1[:, others, :], atol=1e-6,
            err_msg=(
                "a masked-out token influenced the update of another token. Note the "
                "ET-specific trap: a KEY-ONLY mask is NOT enough, because term_k sums "
                "over query columns m and would carry g[:, j, :] into every other row "
                "(decisions.md D-008)."
            ),
        )

        # A token masked in both roles receives no update of its own either.
        np.testing.assert_allclose(
            u0[:, j, :], np.zeros((BATCH, DIM), dtype="float32"), atol=1e-6,
        )

        # Anti-vacuity: the perturbation is real and DOES move the other tokens when the
        # mask is removed — so the test above is measuring the mask, not a no-op.
        v0 = keras.ops.convert_to_numpy(layer.update(x0))
        v1 = keras.ops.convert_to_numpy(layer.update(x1))
        assert np.abs(v1[:, others, :] - v0[:, others, :]).max() > 1e-3, (
            "without the mask the perturbation changed nothing — the masked comparison "
            "would pass vacuously"
        )


# ---------------------------------------------------------------------
# F-02a — a KERAS-PROPAGATED mask must be honored, not silently dropped
# ---------------------------------------------------------------------

def _padded_mask_model(seed=5):
    """`Embedding(mask_zero=True) -> EnergyAttention` — the standard Keras NLP idiom.

    The Embedding is the mask PRODUCER: no mask is ever hand-passed, so this exercises the
    real `_keras_mask` propagation path (decisions.md D-004: a mask test that hand-passes
    an `attention_mask` cannot see F-02 at all).
    """
    vocab = 10
    emb = keras.layers.Embedding(vocab, DIM, mask_zero=True)
    attn = EnergyAttention(dim=DIM, num_heads=HEADS)

    inp = keras.Input(shape=(None,), dtype="int32")
    model = keras.Model(inp, attn(emb(inp)))

    # Excite BOTH layers: with the default inits the output is O(1e-3) and a dropped mask
    # would be indistinguishable from an honored one at atol=1e-6.
    rng = np.random.default_rng(seed)
    emb.embeddings.assign(rng.standard_normal((vocab, DIM)).astype("float32"))
    _excite(attn)
    return model


class TestKerasMaskIsHonored:
    """F-02a: PAD tokens must not influence the real tokens' outputs."""

    def test_pads_do_not_shift_real_token_outputs(self):
        model = _padded_mask_model()

        padded = np.array([[1, 2, 3, 0, 0, 0]], dtype="int32")  # 3 real + 3 PAD
        short = np.array([[1, 2, 3]], dtype="int32")            # pads physically absent

        y_pad = keras.ops.convert_to_numpy(model(padded))[:, :3, :]
        y_short = keras.ops.convert_to_numpy(model(short))

        # Anti-vacuity: the layer must actually be doing something at the real positions.
        assert np.abs(y_short).max() > 1e-3, (
            "the un-padded output is ~0 — the comparison below would pass vacuously"
        )

        max_abs_diff = float(np.abs(y_pad - y_short).max())
        np.testing.assert_allclose(
            y_pad, y_short, atol=1e-6,
            err_msg=(
                f"PAD tokens shifted the real tokens' outputs by {max_abs_diff:.3e}. "
                "EnergyAttention advertises supports_masking=True but its call() must "
                "declare `mask` for Keras to inject the propagated mask (F-02)."
            ),
        )


class TestMaskPrecedence:
    """D-003: `mask` AND `attention_mask` compose as a LOGICAL AND."""

    def test_keras_mask_and_attention_mask_are_anded(self, sample_input):
        j_keras, j_explicit = 3, 4

        layer = EnergyAttention(dim=DIM, num_heads=HEADS)
        layer.build((BATCH, TOKENS, DIM))
        _excite(layer)

        keras_mask = np.ones((BATCH, TOKENS), dtype="bool")
        keras_mask[:, j_keras] = False                     # Keras hides token 3

        explicit = np.ones((BATCH, TOKENS), dtype="float32")
        explicit[:, j_explicit] = 0.0                      # attention_mask hides token 4

        both = np.ones((BATCH, TOKENS), dtype="float32")
        both[:, [j_keras, j_explicit]] = 0.0               # reference: hides BOTH

        y_and = keras.ops.convert_to_numpy(
            layer(sample_input, attention_mask=explicit, mask=keras_mask)
        )
        y_ref = keras.ops.convert_to_numpy(layer(sample_input, attention_mask=both))
        y_explicit_only = keras.ops.convert_to_numpy(
            layer(sample_input, attention_mask=explicit)
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
                "The Keras mask must be merged as an EXTRA multiplicative keep factor "
                "(decisions.md D-003) — an explicit mask must never un-mask a PAD token."
            ),
        )


# ---------------------------------------------------------------------
# Standard layer contract
# ---------------------------------------------------------------------

class TestEnergyAttention:

    def test_instantiation(self):
        default = EnergyAttention(dim=64)
        assert default.dim == 64
        assert default.num_heads == 8
        assert default.head_dim is None          # config value stays None
        assert default._head_dim == 8            # resolved 64 // 8
        assert default.beta is None
        assert default._beta == pytest.approx(1.0 / np.sqrt(8.0))
        assert default.attn_self is False
        assert default.w_key is None and default.w_query is None
        assert not default.built

        custom = EnergyAttention(
            dim=48, num_heads=3, head_dim=16, beta=0.25, attn_self=True,
            kernel_initializer="glorot_uniform",
        )
        assert custom.head_dim == 16
        assert custom._head_dim == 16
        assert custom.beta == 0.25 and custom._beta == 0.25
        assert custom.attn_self is True
        assert isinstance(custom.kernel_initializer, keras.initializers.GlorotUniform)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"dim": 0}, "dim must be a positive integer"),
            ({"dim": -4}, "dim must be a positive integer"),
            ({"dim": 64, "num_heads": 0}, "num_heads must be a positive integer"),
            ({"dim": 64, "num_heads": -2}, "num_heads must be a positive integer"),
            # head_dim resolves to 0 via floor division (4 // 8 == 0)
            ({"dim": 4, "num_heads": 8}, "head_dim must resolve to a positive integer"),
            ({"dim": 64, "head_dim": 0}, "head_dim must resolve to a positive integer"),
            ({"dim": 64, "beta": 0.0}, "beta must be a positive number"),
            ({"dim": 64, "beta": -1.0}, "beta must be a positive number"),
        ],
    )
    def test_invalid_config(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            EnergyAttention(**kwargs)

    def test_forward_pass(self, layer, sample_input, input_shape):
        y = layer(sample_input, training=False)
        assert tuple(y.shape) == input_shape
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(y)))

    def test_call_equals_update(self, layer, sample_input):
        y = keras.ops.convert_to_numpy(layer(sample_input, training=False))
        u = keras.ops.convert_to_numpy(layer.update(sample_input))
        np.testing.assert_allclose(y, u, rtol=1e-6, atol=1e-6)

    def test_energy_shape(self, layer, sample_input):
        e = layer.energy(sample_input)
        assert tuple(e.shape) == (BATCH,)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(e)))

    def test_build_creates_weights(self, input_shape):
        """Exactly 2 trainable weights, both (Y, H, D), and NO bias variable."""
        layer = EnergyAttention(dim=DIM, num_heads=HEADS)
        layer.build(input_shape)

        expected = (DIM // HEADS, HEADS, DIM)  # (Y, H, D)
        assert tuple(layer.w_key.shape) == expected
        assert tuple(layer.w_query.shape) == expected

        assert len(layer.trainable_weights) == 2, (
            f"expected exactly 2 trainable weights (w_key, w_query), got "
            f"{[w.name for w in layer.trainable_weights]}"
        )
        assert len(layer.non_trainable_weights) == 0
        assert not any("bias" in w.name for w in layer.weights), (
            "EnergyAttention MUST be bias-free — the paper's energy is defined without a "
            "bias and the closed-form gradient cannot express one"
        )

    def test_serialization_cycle(self, sample_input, input_shape):
        inputs = keras.Input(shape=input_shape[1:])
        outputs = EnergyAttention(
            dim=DIM, num_heads=HEADS, beta=0.3, attn_self=True
        )(inputs)
        model = keras.Model(inputs, outputs)

        y_before = keras.ops.convert_to_numpy(model(sample_input, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "energy_attention.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            y_after = keras.ops.convert_to_numpy(loaded(sample_input, training=False))

        np.testing.assert_allclose(
            y_before, y_after, rtol=1e-6, atol=1e-6,
            err_msg="EnergyAttention outputs differ after a .keras round-trip",
        )

    def test_get_config_complete(self):
        layer = EnergyAttention(
            dim=48, num_heads=3, head_dim=16, beta=0.25, attn_self=True,
        )
        config = layer.get_config()
        for key in ("dim", "num_heads", "head_dim", "beta", "attn_self",
                    "kernel_initializer"):
            assert key in config, f"{key} missing from get_config()"
        assert config["dim"] == 48
        assert config["num_heads"] == 3
        assert config["head_dim"] == 16
        assert config["beta"] == 0.25
        assert config["attn_self"] is True

    def test_from_config_reconstruction(self, sample_input):
        original = EnergyAttention(dim=DIM, num_heads=HEADS, beta=0.3, attn_self=True)
        rebuilt = EnergyAttention.from_config(original.get_config())

        assert rebuilt.dim == original.dim
        assert rebuilt.num_heads == original.num_heads
        assert rebuilt.beta == original.beta
        assert rebuilt.attn_self == original.attn_self

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
        unbuilt = EnergyAttention(dim=DIM, num_heads=HEADS)
        assert not unbuilt.built
        assert unbuilt.compute_output_shape(input_shape) == input_shape
        assert unbuilt.compute_output_shape((None, 5, DIM)) == (None, 5, DIM)
        assert not unbuilt.built

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_variable_batch_size(self, batch_size):
        layer = EnergyAttention(dim=DIM, num_heads=HEADS)
        rng = np.random.default_rng(batch_size)
        x = rng.standard_normal((batch_size, 5, DIM)).astype("float32")
        y = layer(x, training=False)
        assert tuple(y.shape) == (batch_size, 5, DIM)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(y)))
        assert tuple(layer.energy(x).shape) == (batch_size,)

    def test_gradient_flow(self, sample_input, input_shape):
        layer = EnergyAttention(dim=DIM, num_heads=HEADS)
        layer.build(input_shape)
        x = tf.convert_to_tensor(sample_input)

        with tf.GradientTape() as tape:
            y = layer(x, training=True)
            loss = tf.reduce_mean(tf.square(y))

        grads = tape.gradient(loss, layer.trainable_weights)
        assert len(grads) == 2
        for g, w in zip(grads, layer.trainable_weights):
            assert g is not None, f"no gradient for {w.name}"
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(g))), (
                f"non-finite gradient for {w.name}"
            )


class TestDtypePolicies:
    """S13/S14: EnergyAttention is FINITE under every global dtype policy.

    Iteration 1 shipped `_MASK_BIAS_VALUE = -1e9` applied as `(1 - keep) * _MASK_BIAS_VALUE`
    in the COMPUTE dtype. Under `mixed_float16` that constant is `-inf`, and `keep` is
    ALWAYS constructed, so every UNMASKED position computed `0 * -inf = NaN`:
    **512/512 NaN, with no mask supplied** — measured, not hypothetical. No success
    criterion looked for it. This class is that missing criterion, and it is parametrized
    over the dtype PERMANENTLY (via the `dtype_policy` fixture in
    `tests/test_layers/conftest.py`, which also guarantees the global policy is restored).

    The `N == 1, attn_self=False` cell is the fully-masked degenerate case — every entry of
    the attention matrix is masked out — and is the one most likely to NaN.
    """

    @pytest.mark.parametrize("attn_self", [True, False])
    @pytest.mark.parametrize("use_mask", [False, True])
    @pytest.mark.parametrize("num_tokens", [1, 8])
    def test_no_nan_under_mixed_precision(
        self, dtype_policy, attn_self, use_mask, num_tokens
    ):
        layer = EnergyAttention(dim=DIM, num_heads=HEADS, attn_self=attn_self)
        x = keras.random.normal((BATCH, num_tokens, DIM))

        mask = None
        if use_mask:
            keep = np.ones((BATCH, num_tokens), dtype="float32")
            keep[:, -1] = 0.0            # drop the last token
            mask = keras.ops.convert_to_tensor(keep)

        out = keras.ops.convert_to_numpy(layer(x, attention_mask=mask))

        assert out.shape == (BATCH, num_tokens, DIM)
        assert np.isnan(out).sum() == 0, f"{np.isnan(out).sum()}/{out.size} NaN"
        assert np.isinf(out).sum() == 0, f"{np.isinf(out).sum()}/{out.size} Inf"
        assert np.all(np.isfinite(out))

    def test_energy_and_update_callable_directly(self, dtype_policy):
        """S14: `energy()` / `update()` are safe OUTSIDE `__call__`.

        There is no autocast scope outside `__call__`, so before the fix a float32 `g` met
        float16 weights here and the einsum raised `InvalidArgumentError`. These are the
        methods the attention factory registry ADVERTISES.
        """
        layer = EnergyAttention(dim=DIM, num_heads=HEADS)
        g = keras.random.normal((BATCH, 8, DIM))     # float32, NOT pre-cast by the caller
        layer.build(g.shape)

        e = keras.ops.convert_to_numpy(layer.energy(g))
        u = keras.ops.convert_to_numpy(layer.update(g))

        assert e.shape == (BATCH,)
        assert u.shape == (BATCH, 8, DIM)
        assert np.all(np.isfinite(e))
        assert np.all(np.isfinite(u))

    # C13 (iter-2 / R1): the energy at a REALISTIC sequence length ---------
    #
    # `_REALISTIC_TOKENS` is not decoration. `E_ATT` grows super-linearly in N, and the
    # ONLY reason this failure mode survived a green suite is that every dtype test above
    # runs at `TOKENS = 7`, where the energy is O(-140) and an fp16 overflow is arithmetically
    # impossible. At N = 1024 the float32 energy is O(-8e4) — PAST fp16's 65504 max — so a
    # cast of the reduced energy back into the compute dtype becomes `-inf`. Do NOT shrink
    # this constant "to make the suite faster": below ~1024 (at these dims) the guard cannot
    # bite, and the anti-vacuity assertion below exists to fail loudly if it ever stops
    # biting. See decisions.md D-005.
    _REALISTIC_TOKENS = 1024
    _FP16_MAX = 65504.0

    @pytest.mark.parametrize("num_tokens", [TOKENS, _REALISTIC_TOKENS])
    def test_reported_energy_is_finite_at_a_realistic_length(
        self, dtype_policy, num_tokens
    ):
        """C13: `energy()` must be FINITE at a realistic N under EVERY dtype policy.

        RED at HEAD (before the D-005 fix): `mixed_float16`, N=1024 -> `E_ATT = -inf`,
        because the float32 reduction was cast back into fp16 on the very last op.
        """
        layer = EnergyAttention(dim=DIM, num_heads=HEADS)
        g = keras.random.normal((BATCH, num_tokens, DIM), seed=3)
        layer.build(g.shape)

        e = keras.ops.convert_to_numpy(layer.energy(g))

        assert e.shape == (BATCH,)
        assert np.all(np.isfinite(e)), (
            f"non-finite E_ATT at N={num_tokens} under policy {dtype_policy!r}: {e}. "
            "The energy is a LARGE reduction; it must be returned in the reduce dtype "
            "(>= float32) and NEVER cast back to the compute dtype (decisions.md D-005)."
        )

        if num_tokens == self._REALISTIC_TOKENS:
            # Anti-vacuity: this cell only PROVES anything while |E| exceeds fp16's range.
            # If a future dim/N change drops it below, the guard has stopped biting and the
            # test must say so rather than pass quietly.
            assert np.abs(e).max() > self._FP16_MAX, (
                f"max|E_ATT| = {np.abs(e).max():.1f} <= fp16 max ({self._FP16_MAX}) at "
                f"N={num_tokens} — this guard can no longer detect an fp16 cast-back. "
                "Raise the sequence length or the dims until it can."
            )

    def test_degenerate_case_exactly_zero_in_every_dtype(self, dtype_policy):
        """S9, re-proven per dtype: N == 1 + attn_self=False -> EXACTLY 0, never NaN."""
        layer = EnergyAttention(dim=DIM, num_heads=HEADS, attn_self=False)
        x = keras.random.normal((BATCH, 1, DIM))

        e = keras.ops.convert_to_numpy(layer.energy(x))
        u = keras.ops.convert_to_numpy(layer(x))

        assert np.all(np.isfinite(e)) and np.all(np.isfinite(u))
        assert np.all(e == 0.0), f"energy must be EXACTLY 0.0, got {e}"
        assert np.all(u == 0.0), f"update must be EXACTLY 0.0, got max|u|={np.abs(u).max()}"


class TestFactoryIntegration:
    """S4: EnergyAttention is reachable through the attention factory."""

    def test_factory_integration(self):
        layer = create_attention_layer('energy', dim=DIM)
        assert isinstance(layer, EnergyAttention)

        x = keras.random.normal((2, 8, DIM))
        y = layer(x)
        assert y.shape == (2, 8, DIM)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(y)))


if __name__ == "__main__":
    pytest.main([__file__])
