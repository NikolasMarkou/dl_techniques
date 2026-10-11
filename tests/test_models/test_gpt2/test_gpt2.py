"""Tests for the GPT-2 model implementation.

Covers initialization, parameter validation, forward pass, weight tying,
causal masking, serialization, variant creation, gradient flow, and the
in-model task head (``include_top``).
"""

from typing import Any, Dict

import pytest
import numpy as np
import keras
import tensorflow as tf

import dl_techniques.models.language.gpt2.gpt2 as gpt2_module
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.models.language.gpt2 import create_gpt2
from dl_techniques.models.language.gpt2.gpt2 import GPT2


def _random_ids(shape, vocab_size):
    """Generate random integer token IDs."""
    return np.random.randint(0, vocab_size, shape).astype(np.int32)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def tiny_config():
    """Minimal GPT-2 config for fast testing."""
    return {
        "vocab_size": 256,
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 4,
        "max_seq_len": 32,
        "dropout_rate": 0.0,
        "attention_dropout_rate": 0.0,
    }


@pytest.fixture
def tiny_model(tiny_config):
    """Pre-built tiny GPT-2 model."""
    model = GPT2(**tiny_config)
    dummy = _random_ids((1, tiny_config["max_seq_len"]), tiny_config["vocab_size"])
    model(dummy, training=False)
    return model


# ---------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------


class TestGPT2Initialization:
    """Test GPT-2 model initialization and parameter validation."""

    def test_basic_initialization(self, tiny_config):
        model = GPT2(**tiny_config)
        assert model.vocab_size == 256
        assert model.embed_dim == 64
        assert model.depth == 2
        assert model.num_heads == 4
        assert model.max_seq_len == 32
        assert model.decoder is not None

    def test_default_initialization(self):
        model = GPT2()
        assert model.vocab_size == 100277
        assert model.embed_dim == 768
        assert model.depth == 12
        assert model.num_heads == 12
        assert model.max_seq_len == 1024

    def test_parameter_validation_divisibility(self):
        with pytest.raises(ValueError, match="embed_dim.*must be divisible by num_heads"):
            GPT2(vocab_size=256, embed_dim=100, num_heads=12, depth=2)

    def test_parameter_validation_negative_vocab(self):
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            GPT2(vocab_size=-1, embed_dim=64, depth=2, num_heads=4)

    def test_parameter_validation_dropout(self):
        with pytest.raises(ValueError, match="dropout_rate must be between"):
            GPT2(vocab_size=256, embed_dim=64, depth=2, num_heads=4, dropout_rate=1.5)

    def test_parameter_validation_attention_dropout(self):
        with pytest.raises(ValueError, match="attention_dropout_rate must be between"):
            GPT2(vocab_size=256, embed_dim=64, depth=2, num_heads=4, attention_dropout_rate=-0.1)

    def test_pre_norm_configuration(self, tiny_model):
        """Verify the decoder uses pre-layer normalization."""
        assert tiny_model.decoder.normalization_position == "pre"


# ---------------------------------------------------------------------
# Forward Pass Tests
# ---------------------------------------------------------------------


class TestGPT2ForwardPass:
    """Test GPT-2 forward pass and output shapes."""

    def test_forward_pass_tensor_input(self, tiny_config):
        model = GPT2(**tiny_config)
        batch_size, seq_len = 2, 16
        input_ids = _random_ids((batch_size, seq_len), tiny_config["vocab_size"])
        outputs = model(input_ids, training=False)

        assert "logits" in outputs
        assert "last_hidden_state" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, tiny_config["vocab_size"])
        assert outputs["last_hidden_state"].shape == (batch_size, seq_len, tiny_config["embed_dim"])

    def test_forward_pass_dict_input(self, tiny_config):
        model = GPT2(**tiny_config)
        batch_size, seq_len = 2, 16
        inputs = {
            "input_ids": _random_ids((batch_size, seq_len), tiny_config["vocab_size"]),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int32),
        }
        outputs = model(inputs, training=False)

        assert outputs["logits"].shape == (batch_size, seq_len, tiny_config["vocab_size"])

    def test_forward_pass_dict_missing_input_ids(self, tiny_config):
        model = GPT2(**tiny_config)
        with pytest.raises(ValueError, match="input_ids"):
            model({"attention_mask": np.ones((2, 16), dtype=np.int32)})

    def test_forward_pass_batch_size_one(self, tiny_config):
        model = GPT2(**tiny_config)
        input_ids = _random_ids((1, 8), tiny_config["vocab_size"])
        outputs = model(input_ids, training=False)
        assert outputs["logits"].shape == (1, 8, tiny_config["vocab_size"])

    def test_forward_pass_full_sequence(self, tiny_config):
        model = GPT2(**tiny_config)
        seq_len = tiny_config["max_seq_len"]
        input_ids = _random_ids((1, seq_len), tiny_config["vocab_size"])
        outputs = model(input_ids, training=False)
        assert outputs["logits"].shape == (1, seq_len, tiny_config["vocab_size"])


# ---------------------------------------------------------------------
# Weight Tying Tests
# ---------------------------------------------------------------------


class TestGPT2WeightTying:
    """Test that LM head reuses token embedding weights."""

    def test_weight_tying(self, tiny_config):
        model = GPT2(**tiny_config)
        input_ids = _random_ids((1, 8), tiny_config["vocab_size"])
        model(input_ids, training=False)

        embedding_weights = model.decoder.word_embeddings.embeddings
        assert embedding_weights.shape == (tiny_config["vocab_size"], tiny_config["embed_dim"])

    def test_logits_use_embedding_weights(self, tiny_config):
        """Verify logits are computed via matmul with embedding weights."""
        model = GPT2(**tiny_config)
        input_ids = _random_ids((1, 8), tiny_config["vocab_size"])
        outputs = model(input_ids, training=False)

        hidden_states = outputs["last_hidden_state"]
        embedding_weights = model.decoder.word_embeddings.embeddings
        expected_logits = keras.ops.matmul(
            hidden_states, keras.ops.transpose(embedding_weights)
        )

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(outputs["logits"]),
            keras.ops.convert_to_numpy(expected_logits),
            atol=1e-5,
        )


# ---------------------------------------------------------------------
# Causal Masking Tests
# ---------------------------------------------------------------------


class TestGPT2CausalMasking:
    """Test that causal masking prevents attending to future tokens."""

    def test_causal_masking_future_does_not_affect_past(self, tiny_config):
        """Changing a future token must not change any earlier position's logits."""
        tiny_config = {**tiny_config, "dropout_rate": 0.0, "attention_dropout_rate": 0.0}
        model = GPT2(**tiny_config)

        # Only the last token differs between sequences
        seq1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
        seq2 = np.array([[1, 2, 3, 4, 5, 6, 7, 99]], dtype=np.int32)

        out1 = model(seq1, training=False)
        out2 = model(seq2, training=False)

        logits1 = keras.ops.convert_to_numpy(out1["logits"])
        logits2 = keras.ops.convert_to_numpy(out2["logits"])

        # Positions 0-6 must be identical (future cannot affect past)
        for pos in range(7):
            np.testing.assert_allclose(
                logits1[0, pos], logits2[0, pos], atol=1e-6,
                err_msg=f"Position {pos} logits changed when only position 7 changed "
                        f"(causality violation: future leaking into past)",
            )

        # Position 7 should differ (it sees different input at its own position)
        assert not np.allclose(logits1[0, 7], logits2[0, 7], atol=1e-3)

    def test_causal_masking_later_tokens_differ(self, tiny_config):
        """Later tokens should differ when context changes."""
        tiny_config = {**tiny_config, "dropout_rate": 0.0, "attention_dropout_rate": 0.0}
        model = GPT2(**tiny_config)

        # Same prefix up to position 3, then diverge
        seq1 = np.array([[1, 2, 3, 10, 20, 30, 40, 50]], dtype=np.int32)
        seq2 = np.array([[1, 2, 3, 99, 99, 99, 99, 99]], dtype=np.int32)

        out1 = model(seq1, training=False)
        out2 = model(seq2, training=False)

        logits1 = keras.ops.convert_to_numpy(out1["logits"])
        logits2 = keras.ops.convert_to_numpy(out2["logits"])

        # Positions 0-2 must be identical (same prefix, causal masking)
        for pos in range(3):
            np.testing.assert_allclose(
                logits1[0, pos], logits2[0, pos], atol=1e-6,
                err_msg=f"Position {pos} logits differ despite identical prefix",
            )

        # Tokens at position 4+ should differ (different preceding tokens)
        assert not np.allclose(logits1[0, 4], logits2[0, 4], atol=1e-3)


# ---------------------------------------------------------------------
# Serialization Tests
# ---------------------------------------------------------------------


class TestGPT2Serialization:
    """Test GPT-2 model serialization and deserialization."""

    def test_get_config(self, tiny_config):
        model = GPT2(**tiny_config)
        config = model.get_config()

        assert config["vocab_size"] == tiny_config["vocab_size"]
        assert config["embed_dim"] == tiny_config["embed_dim"]
        assert config["depth"] == tiny_config["depth"]
        assert config["num_heads"] == tiny_config["num_heads"]
        assert config["max_seq_len"] == tiny_config["max_seq_len"]

    def test_from_config_roundtrip(self, tiny_config):
        model = GPT2(**tiny_config)
        config = model.get_config()
        model2 = GPT2.from_config(config)

        assert model2.vocab_size == model.vocab_size
        assert model2.embed_dim == model.embed_dim
        assert model2.depth == model.depth
        assert model2.num_heads == model.num_heads

    def test_compute_output_shape(self, tiny_config):
        model = GPT2(**tiny_config)
        shapes = model.compute_output_shape((None, 16))
        assert shapes["logits"] == (None, 16, tiny_config["vocab_size"])
        assert shapes["last_hidden_state"] == (None, 16, tiny_config["embed_dim"])


# ---------------------------------------------------------------------
# Variant Tests
# ---------------------------------------------------------------------


class TestGPT2Variants:
    """Test GPT-2 model variant creation."""

    def test_from_variant_tiny(self):
        model = GPT2.from_variant("tiny")
        assert model.embed_dim == 256
        assert model.depth == 4
        assert model.num_heads == 4

    def test_from_variant_small(self):
        model = GPT2.from_variant("small")
        assert model.embed_dim == 768
        assert model.depth == 12
        assert model.num_heads == 12

    def test_from_variant_with_overrides(self):
        model = GPT2.from_variant("tiny", dropout_rate=0.2)
        assert model.dropout_rate == 0.2
        assert model.embed_dim == 256

    def test_from_variant_unknown(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            GPT2.from_variant("nonexistent")

    def test_from_variant_pretrained_missing_file(self):
        with pytest.raises(FileNotFoundError):
            GPT2.from_variant("tiny", pretrained="/nonexistent/path.keras")

    def test_all_variants_have_required_keys(self):
        required_keys = {"embed_dim", "depth", "num_heads", "max_seq_len"}
        for name, config in GPT2.MODEL_VARIANTS.items():
            for key in required_keys:
                assert key in config, f"Variant '{name}' missing key '{key}'"


# ---------------------------------------------------------------------
# Gradient Flow Tests
# ---------------------------------------------------------------------


class TestGPT2GradientFlow:
    """Test gradient flow through the model."""

    def test_gradient_flow(self, tiny_config):
        model = GPT2(**tiny_config)
        batch_size, seq_len = 2, 8
        input_ids = _random_ids((batch_size, seq_len), tiny_config["vocab_size"])
        labels = _random_ids((batch_size, seq_len), tiny_config["vocab_size"])

        with tf.GradientTape() as tape:
            outputs = model(input_ids, training=True)
            logits = outputs["logits"]
            loss = keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=True
            )
            loss = keras.ops.mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)

        for var, grad in zip(model.trainable_variables, gradients):
            assert grad is not None, f"No gradient for {var.name}"

    def test_training_step(self, tiny_config):
        """Test that model can do a training step with GradientTape."""
        model = GPT2(**tiny_config)
        batch_size, seq_len = 4, 8
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        input_ids = _random_ids((batch_size, seq_len), tiny_config["vocab_size"])
        labels = _random_ids((batch_size, seq_len), tiny_config["vocab_size"])

        # Get initial weights snapshot
        _ = model(input_ids, training=False)
        initial_weights = [w.numpy().copy() for w in model.trainable_weights[:2]]

        with tf.GradientTape() as tape:
            outputs = model(input_ids, training=True)
            loss = loss_fn(labels, outputs["logits"])

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Weights should have changed
        updated_weights = [w.numpy() for w in model.trainable_weights[:2]]
        for w_init, w_updated in zip(initial_weights, updated_weights):
            assert not np.allclose(w_init, w_updated), "Weights did not update"


# ---------------------------------------------------------------------
# Iter-1 Refactor Lock-In Tests (plan_2026-05-11_a9e8e6f6)
# ---------------------------------------------------------------------


class TestGPT2Iter1Refactor:
    """Lock-in tests for the iter-1 refactor: factory, NotImplementedError, API surface."""

    def test_create_gpt2_factory_returns_gpt2_instance(self):
        """`create_gpt2` is importable from the package root and returns a
        ``GPT2`` instance with variant params matching ``MODEL_VARIANTS``."""
        from dl_techniques.models.language.gpt2 import GPT2 as PkgGPT2, create_gpt2

        model = create_gpt2("tiny")
        assert isinstance(model, PkgGPT2)
        tiny_spec = PkgGPT2.MODEL_VARIANTS["tiny"]
        assert model.embed_dim == tiny_spec["embed_dim"]
        assert model.depth == tiny_spec["depth"]
        assert model.num_heads == tiny_spec["num_heads"]

    def test_from_variant_pretrained_true_raises_not_implemented(self):
        """`GPT2.from_variant(..., pretrained=True)` must raise
        ``NotImplementedError`` (no silent random-init fallback). Locks in the
        I-01 fix."""
        with pytest.raises(NotImplementedError):
            GPT2.from_variant("tiny", pretrained=True)

    def test_public_api_surface(self):
        """The package's declared public API (`__all__`) is exactly
        ``{GPT2, create_gpt2}``. Locks in the 2-name surface."""
        import dl_techniques.models.language.gpt2 as pkg

        assert sorted(pkg.__all__) == ["GPT2", "create_gpt2"]


# ---------------------------------------------------------------------
# Untied LM Head Tests (plan-2026-08-13-230c101d / F-05)
# ---------------------------------------------------------------------


class TestGPT2UntiedLMHead:
    """Cover ``tie_word_embeddings=False`` — the only hand-rolled layer in the
    package (``keras.layers.Dense(vocab_size, use_bias=False, name='lm_head')``,
    ``gpt2.py`` ``GPT2._build_architecture``), which had ZERO tests.

    Every bar below is a measured number, not a hope. Measured on GPU:1 and
    reproduced across ``keras.utils.set_random_seed`` in {7, 11, 23} with the
    SHIPPED initializers, ``tiny_config`` and ``ids`` of shape ``(2, 8)``:

    - ``|untied logits - tied recomputation|`` max: 0.8657 / 0.8186 / 0.8936
      (logits themselves are O(0.5)), so the ``> 1e-2`` bar has ~80x margin.
    - L2 norm of ``d(mean(logits^2))/d(lm_head.kernel)``: 0.04633 / 0.04539 /
      0.05494, so the ``> 1e-6`` bar has ~4.5e4x margin.
    """

    @staticmethod
    def _untied_model(tiny_config, seed=1234):
        keras.utils.set_random_seed(seed)
        return GPT2(**tiny_config, tie_word_embeddings=False)

    def test_untied_head_is_a_distinct_variable(self, tiny_config):
        """The untied ``lm_head`` kernel must be its OWN variable, not an alias
        of (or a view onto) the token-embedding table."""
        model = self._untied_model(tiny_config)
        model(_random_ids((2, 8), tiny_config["vocab_size"]), training=False)

        assert model.lm_head is not None
        embeddings = model.decoder.word_embeddings.embeddings
        kernel = model.lm_head.kernel

        # Shapes are transposes of each other, so a shape check alone would
        # not distinguish them; pin the orientation too. (The `is not` identity
        # check lives in the square-config test below: at this config the two
        # shapes differ, so aliasing is structurally impossible and an identity
        # assertion here would be a tautology that no injection can turn RED.)
        assert tuple(kernel.shape) == (
            tiny_config["embed_dim"], tiny_config["vocab_size"],
        )
        assert tuple(embeddings.shape) == (
            tiny_config["vocab_size"], tiny_config["embed_dim"],
        )
        # ``use_bias=False`` — the head is a bare linear projection.
        assert model.lm_head.bias is None

    # DECISION plan-2026-08-13T091555-230c101d/D-009
    # Do NOT move this identity assertion back into the rectangular-config
    # test above and do NOT "simplify" this square config away. At
    # vocab_size != embed_dim the kernel (D, V) and the embedding table (V, D)
    # have transposed shapes, so no injection can make them the same object
    # while the head stays live: three separate attempts were defeated by an
    # EARLIER guard (Keras' post-build state lock, a BatchMatMul shape error,
    # and "You must build the layer before accessing `kernel`") rather than by
    # the assertion. An assertion no injection can turn RED is a tautology, not
    # a guard. See decisions.md D-009.
    def test_untied_head_kernel_is_not_the_embedding_variable(self):
        """The untied head's kernel must be its OWN variable, not an alias of
        the token-embedding table.

        Deliberately run at ``vocab_size == embed_dim`` (64/64). At the normal
        rectangular config the two variables have transposed shapes, so an
        alias is structurally impossible and this assertion could not be made
        to fail by ANY injection — a tautology, not a guard. At a square config
        the alias IS shape-legal (it is the classic transposed-tie defect), so
        the assertion becomes falsifiable and was RED-proven by injecting
        exactly that alias.
        """
        square_config = {
            "vocab_size": 64, "embed_dim": 64, "depth": 2, "num_heads": 4,
            "max_seq_len": 32, "dropout_rate": 0.0,
            "attention_dropout_rate": 0.0,
        }
        model = self._untied_model(square_config)
        model(_random_ids((2, 8), 64), training=False)

        embeddings = model.decoder.word_embeddings.embeddings
        assert model.lm_head.kernel is not embeddings
        assert all(w is not embeddings for w in model.lm_head.weights)

    def test_untied_logits_are_not_the_tied_computation(self, tiny_config):
        """BEHAVIORAL untiedness: the logits must NOT equal
        ``hidden @ embeddings.T``. An identity-only check would pass even if
        ``call`` silently took the tied branch with a fresh copy of the table.

        Measured max |delta| across seeds {7, 11, 23}: 0.8657 / 0.8186 / 0.8936
        against logits of magnitude ~0.5. Bar: > 1e-2.
        """
        model = self._untied_model(tiny_config)
        outputs = model(
            _random_ids((2, 8), tiny_config["vocab_size"]), training=False,
        )

        tied_logits = keras.ops.matmul(
            outputs["last_hidden_state"],
            keras.ops.transpose(model.decoder.word_embeddings.embeddings),
        )
        delta = np.abs(
            keras.ops.convert_to_numpy(outputs["logits"])
            - keras.ops.convert_to_numpy(tied_logits)
        ).max()
        assert delta > 1e-2, (
            f"untied logits are indistinguishable from the tied computation "
            f"(max |delta| = {delta:.4e}); the lm_head is not in the path"
        )

    def test_gradients_reach_the_untied_lm_head_kernel(self, tiny_config):
        """Gradients must actually reach the hand-rolled head. A dead head
        (``stop_gradient`` around the ``lm_head`` call) yields ``None`` here.

        Measured grad L2 norm across seeds {7, 11, 23}: 0.04633 / 0.04539 /
        0.05494. Bar: > 1e-6.
        """
        model = self._untied_model(tiny_config)
        input_ids = _random_ids((2, 8), tiny_config["vocab_size"])
        model(input_ids, training=False)

        kernel = model.lm_head.kernel
        embeddings = model.decoder.word_embeddings.embeddings
        with tf.GradientTape() as tape:
            outputs = model(input_ids, training=True)
            loss = keras.ops.mean(keras.ops.square(outputs["logits"]))
        grad, emb_grad = tape.gradient(loss, [kernel, embeddings])

        assert grad is not None, "no gradient reaches lm_head.kernel"
        grad_norm = float(
            keras.ops.sqrt(keras.ops.sum(keras.ops.square(grad)))
        )
        assert grad_norm > 1e-6, (
            f"lm_head.kernel gradient is dead (L2 = {grad_norm:.4e})"
        )
        # The untied head must not starve the embedding table either: with an
        # untied head the table is still reached through the input embedding.
        assert emb_grad is not None

    def test_tied_model_has_no_lm_head(self, tiny_config):
        """Control for the three tests above: the default (tied) path builds
        NO ``lm_head`` at all, so the untied assertions are exercising a
        genuinely different branch."""
        keras.utils.set_random_seed(1234)
        model = GPT2(**tiny_config, tie_word_embeddings=True)
        model(_random_ids((2, 8), tiny_config["vocab_size"]), training=False)
        assert model.lm_head is None


# ---------------------------------------------------------------------
# attention_mask Tests (plan-2026-08-13-230c101d / F-05)
# ---------------------------------------------------------------------


class TestGPT2AttentionMask:
    """``attention_mask`` must actually mask, not merely be accepted.

    ``test_forward_pass_dict_input`` passes an all-ones mask and checks only the
    output SHAPE, so nothing pinned padding behaviour. GPT-2 is causal, so a
    RIGHT-padded probe is vacuous (causality alone hides the future). These
    tests use LEFT padding — positions ``0..P-1`` masked, ``P..N-1`` real — and
    perturb the token IDs sitting at the masked positions.

    Measured (``vocab_size=256, embed_dim=64, depth=2, num_heads=4``, B=2, N=12,
    P=5, seed pinned), on BOTH the RTX 4070 (``CUDA_VISIBLE_DEVICES=1``) and CPU
    (``CUDA_VISIBLE_DEVICES=""``), i.e. in both the TF32 and non-TF32 regimes:

    - max |delta| at UNMASKED positions: **exactly 0.0** in both regimes.
    - max |delta| at MASKED positions:   1.0547 (the perturbation is real).
    - same perturbation with NO mask:    3.944e-03 at the unmasked positions
      (i.e. the isolation is due to the mask, not to causality).

    Logits are O(0.96), so the ``< 1e-6`` bar is a two-sided margin, not a
    bit-identity claim.

    **Re-measured 2026-08-18** (plan-2026-08-18T140459-7991552f, step 8, F-70).
    The previous numbers were 0.8897 / 0.1973 against logits of O(0.75). They
    moved because ``TextDecoder`` now initialises its transformer blocks from
    ``initializer_range`` (0.02 truncated normal) instead of falling through to
    ``TransformerLayer``'s ``glorot_uniform`` (~0.036 for these widths) -- the
    blocks' weights are ~1.8x smaller, so the *cross-position* signal that has to
    traverse two attention blocks fell ~50x while the *same-position* signal,
    dominated by the untouched embedding path, rose. The no-mask control's bar
    moved 1e-2 -> 1e-3 for that reason and only that reason: it is still 3.9x
    above the new measurement and 3900x above the ``1e-6`` isolation bar it
    exists to make non-vacuous, and the isolation measurement itself is still
    exactly 0.0.
    """

    P = 5  # number of left-padded positions

    @staticmethod
    def _probe_inputs(vocab_size, batch=2, seq_len=12, pad=5):
        rng = np.random.default_rng(0)
        ids = rng.integers(0, vocab_size, (batch, seq_len)).astype(np.int32)
        mask = np.ones((batch, seq_len), dtype=np.int32)
        mask[:, :pad] = 0
        perturbed = ids.copy()
        perturbed[:, :pad] = rng.integers(
            0, vocab_size, (batch, pad),
        ).astype(np.int32)
        assert not np.array_equal(ids[:, :pad], perturbed[:, :pad])
        return ids, perturbed, mask

    def test_masked_positions_do_not_reach_unmasked_outputs(self, tiny_config):
        """Perturbing the tokens at LEFT-padded (masked) positions must leave
        every unmasked position's logits unchanged. Measured: 0.0 (GPU and
        CPU). Bar: < 1e-6."""
        keras.utils.set_random_seed(1234)
        model = GPT2(**tiny_config)
        ids, perturbed, mask = self._probe_inputs(tiny_config["vocab_size"])

        base = keras.ops.convert_to_numpy(
            model({"input_ids": ids, "attention_mask": mask},
                  training=False)["logits"]
        )
        moved = keras.ops.convert_to_numpy(
            model({"input_ids": perturbed, "attention_mask": mask},
                  training=False)["logits"]
        )

        leak = np.abs(base[:, self.P:] - moved[:, self.P:]).max()
        assert leak < 1e-6, (
            f"masked (padded) tokens leaked into unmasked positions: "
            f"max |delta| = {leak:.4e} (measured 0.0 on GPU and CPU)"
        )

    def test_the_perturbation_is_real(self, tiny_config):
        """Non-vacuity guard for the test above: the perturbed tokens DO change
        the model's output somewhere — at the masked positions themselves.
        Measured 1.0547 against logits of magnitude ~0.96. Bar: > 1e-2."""
        keras.utils.set_random_seed(1234)
        model = GPT2(**tiny_config)
        ids, perturbed, mask = self._probe_inputs(tiny_config["vocab_size"])

        base = keras.ops.convert_to_numpy(
            model({"input_ids": ids, "attention_mask": mask},
                  training=False)["logits"]
        )
        moved = keras.ops.convert_to_numpy(
            model({"input_ids": perturbed, "attention_mask": mask},
                  training=False)["logits"]
        )

        moved_delta = np.abs(base[:, :self.P] - moved[:, :self.P]).max()
        assert moved_delta > 1e-2, (
            f"the probe perturbation changed nothing anywhere "
            f"(max |delta| at masked positions = {moved_delta:.4e}); the "
            f"isolation assertion would be vacuous"
        )

    def test_without_the_mask_the_same_tokens_do_leak(self, tiny_config):
        """The isolation is attributable to the MASK, not to causality: drop
        the mask and the identical perturbation reaches the same positions.
        Measured 3.944e-03 (GPU and CPU). Bar: > 1e-3 -- see the class docstring
        for why this bar moved from 1e-2 on 2026-08-18."""
        keras.utils.set_random_seed(1234)
        model = GPT2(**tiny_config)
        ids, perturbed, _mask = self._probe_inputs(tiny_config["vocab_size"])

        base = keras.ops.convert_to_numpy(
            model(ids, training=False)["logits"]
        )
        moved = keras.ops.convert_to_numpy(
            model(perturbed, training=False)["logits"]
        )

        leak = np.abs(base[:, self.P:] - moved[:, self.P:]).max()
        assert leak > 1e-3, (
            f"unmasked run shows no influence from the prefix tokens "
            f"(max |delta| = {leak:.4e}); the masked run's 0.0 would then be "
            f"explained by causality alone and prove nothing about masking"
        )


# ---------------------------------------------------------------------
# pretrained="<path>" -- the local-path door (C-4b, D-070)
# ---------------------------------------------------------------------


class TestPretrainedLocalPathIsVerified:
    """The route the `pretrained=True` error message points callers at.

    Before D-070 a local path was restored with ``skip_mismatch=True`` and a
    checkpoint whose variables did not line up restored NOTHING while the next
    line logged "Loaded weights from ..." -- the same "silently untrained
    model" the `pretrained=True` contract exists to prevent, one door over.
    Since D-024 (plan-2026-10-11T004932-b83935e1) the load is by layer name and
    complete-or-raise, so such a checkpoint is refused before anything loads.
    """

    def test_mismatched_checkpoint_raises_naming_the_cause(self, tmp_path):
        other = keras.Sequential(
            [keras.layers.Input(shape=(4,)), keras.layers.Dense(3, name="d")]
        )
        ckpt = str(tmp_path / "not_a_gpt2.keras")
        other.save(ckpt)

        with pytest.raises(ValueError, match="No overlapping layers"):
            GPT2.from_variant("tiny", vocab_size=64, pretrained=ckpt)

    def test_a_checkpoint_that_fills_only_the_lm_head_raises(self, tmp_path):
        """An ``lm_head`` that fits does not excuse a missing decoder (D-024).

        ``lm_head`` is the one layer allowed to stay fresh; the decoder is not,
        so a checkpoint whose only overlap is an equal-shape ``lm_head`` is
        refused rather than returning a model with a fresh backbone.
        """
        other = keras.Sequential([
            keras.layers.Input(shape=(32,)),
            keras.layers.Dense(64, use_bias=False, name="lm_head"),
        ])
        ckpt = str(tmp_path / "only_an_lm_head.keras")
        other.save(ckpt)

        with pytest.raises(ValueError, match=r"unloaded: \['decoder'\]"):
            GPT2.from_variant("tiny", vocab_size=64, embed_dim=32, depth=2,
                              num_heads=2, max_seq_len=16,
                              tie_word_embeddings=False, pretrained=ckpt)

    def test_matching_checkpoint_still_loads(self, tmp_path):
        """Anti-vacuity: the real warm-start path must still work."""
        src = GPT2.from_variant("tiny", vocab_size=64)
        src(_random_ids((1, 8), 64), training=False)
        ckpt = str(tmp_path / "gpt2_tiny.keras")
        src.save(ckpt)

        loaded = GPT2.from_variant("tiny", vocab_size=64, pretrained=ckpt)
        x = _random_ids((1, 8), 64)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(src(x, training=False)["logits"]),
            keras.ops.convert_to_numpy(loaded(x, training=False)["logits"]),
            rtol=1e-6,
            atol=1e-6,
        )


# ---------------------------------------------------------------------
# In-model task head: include_top / task_config / head_config_overrides
# (plan-2026-10-11T004932-b83935e1, D-012)
# ---------------------------------------------------------------------

#: ``max_seq_len`` 16 sits below the 32-token dummy forward the pretrained
#: path used to run, so every ``pretrained=<path>`` arm here also guards that
#: fix (it raised "seq_len=32 exceeds max_seq_len=16" before).
_HEAD_KW: Dict[str, Any] = {
    "vocab_size": 64, "embed_dim": 32, "depth": 2, "num_heads": 2,
    "max_seq_len": 16,
}
_SEQ = 8
_CLF = NLPTaskConfig(
    name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3
)
_NER = NLPTaskConfig(
    name="ner", task_type=NLPTaskType.TOKEN_CLASSIFICATION, num_classes=9
)
#: Same head class and weight shapes as ``_CLF``, a different task.
_SPAM = NLPTaskConfig(
    name="spam", task_type=NLPTaskType.SPAM_DETECTION, num_classes=3
)


def _np(tensor) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(tensor))


def _head_batch() -> Dict[str, np.ndarray]:
    """Row 1 is right-padded: 5 real tokens, then 3 pad ids under a 0 mask."""
    ids = np.random.default_rng(0).integers(1, 64, (2, _SEQ)).astype("int32")
    mask = np.ones_like(ids)
    ids[1, 5:] = 0
    mask[1, 5:] = 0
    return {"input_ids": ids, "attention_mask": mask}


def _relative_paths(model: keras.Model) -> list:
    """Weight paths without the model-root segment (guide 8.3)."""
    return sorted(w.path.split("/", 1)[-1] for w in model.weights)


def _backbone_hidden(model: GPT2, batch: Dict[str, np.ndarray]) -> np.ndarray:
    """``last_hidden_state`` of ``model``'s decoder, whatever its top.

    Read through a headless twin rebuilt from the model's own config and given
    the model's own decoder weights; the twin's LM head does not feed
    ``last_hidden_state``.
    """
    config = model.get_config()
    config.update(include_top=False, task_config=None, head_config_overrides=None)
    config.pop("name", None)
    twin = GPT2.from_config(config)
    twin(batch, training=False)
    twin.decoder.set_weights(model.decoder.get_weights())
    return _np(twin(batch, training=False)["last_hidden_state"])


def _randomized(model: GPT2, seed: int) -> GPT2:
    """Overwrite every weight, so a restored value cannot match by init luck."""
    assert model.weights, "build the model before randomizing it"
    rng = np.random.default_rng(seed)
    for w in model.weights:
        w.assign(rng.normal(0.0, 0.1, size=tuple(w.shape)).astype(w.dtype))
    return model


class TestGPT2IncludeTop:
    """The ``include_top`` / ``task_config`` / ``head_config_overrides`` knobs."""

    def test_head_output_shapes(self):
        """Bare logits for single-output heads, a dict for QA; nothing else."""
        batch = _head_batch()
        out = GPT2(**_HEAD_KW, include_top=True, task_config=_CLF)(
            batch, training=False)
        assert not isinstance(out, dict) and tuple(out.shape) == (2, 3)

        out = GPT2(**_HEAD_KW, include_top=True, task_config=_NER)(
            batch, training=False)
        assert not isinstance(out, dict) and tuple(out.shape) == (2, _SEQ, 9)

        qa = GPT2(**_HEAD_KW, include_top=True, task_config=NLPTaskConfig(
            name="qa", task_type=NLPTaskType.QUESTION_ANSWERING, num_classes=2))
        out = qa(batch, training=False)
        # No `logits` / `last_hidden_state` beside the head's keys.
        assert set(out) == {"start_logits", "end_logits"}
        assert tuple(out["start_logits"].shape) == (2, _SEQ)

    def test_a_task_head_replaces_the_untied_lm_head(self):
        """``build()`` must not create the LM head ``call()`` no longer runs."""
        headed = GPT2(**_HEAD_KW, tie_word_embeddings=False,
                      include_top=True, task_config=_CLF)
        headed(_head_batch(), training=False)
        assert headed.lm_head is None
        assert not [w.path for w in headed.weights if "lm_head" in w.path]

        headless = GPT2(**_HEAD_KW, tie_word_embeddings=False)
        headless(_head_batch(), training=False)
        assert [w.path.split("/", 1)[-1] for w in headless.weights
                if "lm_head" in w.path] == ["lm_head/kernel"]

    @pytest.mark.parametrize("head_type", ["linear", "harmonic"])
    def test_no_lm_logits_are_computed_under_a_head(self, monkeypatch, head_type):
        """Neither the tied nor the harmonic projection runs with a head.

        Both are replaced by functions that raise; the headed forward pass must
        not reach them, the headless one must (anti-vacuity).
        """
        def _forbidden(*args, **kwargs):
            raise AssertionError("LM logits computed")

        monkeypatch.setattr(gpt2_module, "tied_embedding_logits", _forbidden)
        monkeypatch.setattr(gpt2_module, "harmonic_logits", _forbidden)
        headed = GPT2(**_HEAD_KW, head_type=head_type,
                      include_top=True, task_config=_CLF)
        assert tuple(headed(_head_batch(), training=False).shape) == (2, 3)
        with pytest.raises(AssertionError, match="LM logits computed"):
            GPT2(**_HEAD_KW, head_type=head_type)(_head_batch(), training=False)

    def test_symbolic_output_matches_the_eager_one(self):
        """A symbolic call reports what ``call`` returns, headless or headed."""
        model = GPT2(**_HEAD_KW, include_top=True, task_config=_NER)
        symbolic = model(keras.Input(shape=(None,), dtype="int32"))
        assert not isinstance(symbolic, dict)
        assert tuple(symbolic.shape) == (None, None, 9)
        headless = GPT2(**_HEAD_KW)
        symbolic = headless(keras.Input(shape=(None,), dtype="int32"))
        assert set(symbolic) == {"logits", "last_hidden_state"}

    def test_the_default_pooling_is_last_and_an_override_wins(self):
        """Causal decoder: pool the last real token, not position 0 (D-012)."""
        assert GPT2(**_HEAD_KW, include_top=True,
                    task_config=_CLF).task_head.pooling_type == "last"
        assert GPT2(**_HEAD_KW, include_top=True, task_config=_CLF,
                    head_config_overrides={"pooling_type": "mean"}
                    ).task_head.pooling_type == "mean"

    def test_the_head_reads_the_last_real_token(self):
        """Changing the last real token changes the head output.

        Under ``'cls'`` pooling the head reads position 0, which in a causal
        decoder has seen only itself, so this change would be invisible.
        """
        model = GPT2(**_HEAD_KW, include_top=True, task_config=_CLF)
        batch = _head_batch()
        base = _np(model(batch, training=False))
        edited = {k: v.copy() for k, v in batch.items()}
        edited["input_ids"][1, 4] = (edited["input_ids"][1, 4] % 63) + 1
        moved = _np(model(edited, training=False))
        np.testing.assert_array_equal(moved[0], base[0])
        assert not np.allclose(moved[1], base[1], atol=1e-6, rtol=0)

    @pytest.mark.parametrize("with_mask", [True, False],
                             ids=["attention_mask", "pad_derived"])
    def test_appended_padding_does_not_change_the_head_output(self, with_mask):
        """Padding after the last real token is invisible to the head.

        The decoder is causal, so the real positions' states do not depend on
        what follows them; the head must not pool the padding either. Without
        an ``attention_mask`` the head's mask is ``input_ids != pad_token_id``
        (here 0, the default).
        """
        model = GPT2(**_HEAD_KW, include_top=True, task_config=_CLF)
        ids = np.random.default_rng(3).integers(1, 64, (2, 5)).astype("int32")
        padded = np.concatenate([ids, np.zeros((2, 4), "int32")], axis=1)

        def run(x):
            if with_mask:
                return _np(model({"input_ids": x,
                                  "attention_mask": (x != 0).astype("int32")},
                                 training=False))
            return _np(model(x, training=False))

        np.testing.assert_allclose(run(padded), run(ids), atol=1e-5, rtol=0)

    def test_a_custom_pad_token_id_drives_the_derived_mask(self):
        """``pad_token_id`` is the padding id when no mask is given."""
        model = GPT2(**_HEAD_KW, pad_token_id=7, include_top=True,
                     task_config=_CLF)
        ids = np.random.default_rng(4).integers(8, 64, (2, 5)).astype("int32")
        padded = np.concatenate([ids, np.full((2, 3), 7, "int32")], axis=1)
        np.testing.assert_allclose(_np(model(padded, training=False)),
                                   _np(model(ids, training=False)),
                                   atol=1e-5, rtol=0)

    def test_config_round_trip_with_a_head(self):
        """Every knob is serialized; ``task_config`` as a plain JSON dict."""
        import json

        overrides = {"pooling_type": "mean"}
        model = GPT2(**_HEAD_KW, pad_token_id=5, include_top=True,
                     task_config=_CLF, head_config_overrides=overrides)
        config = model.get_config()
        assert config["include_top"] is True
        assert config["pad_token_id"] == 5
        assert config["head_config_overrides"] == overrides
        assert isinstance(config["task_config"], dict)
        assert config["task_config"]["task_type"] == "text_classification"
        json.dumps(config)  # plain data, no enum or dataclass inside

        from_dict = GPT2(**_HEAD_KW, include_top=True,
                         task_config=dict(config["task_config"]),
                         head_config_overrides=overrides)
        assert from_dict.get_config()["task_config"] == config["task_config"]

        rebuilt = GPT2.from_config(config)
        assert rebuilt.get_config() == config
        assert rebuilt.task_head.pooling_type == "mean"

        batch = _head_batch()
        model(batch, training=False)
        rebuilt(batch, training=False)
        assert _relative_paths(rebuilt) == _relative_paths(model)
        rebuilt.set_weights(model.get_weights())
        np.testing.assert_array_equal(
            _np(rebuilt(batch, training=False)), _np(model(batch, training=False)))

    def test_a_config_without_the_new_keys_still_loads(self):
        """A config written before the knobs existed rebuilds the headless model."""
        config = GPT2(**_HEAD_KW).get_config()
        for key in ("include_top", "task_config", "head_config_overrides",
                    "pad_token_id"):
            config.pop(key)
        model = GPT2.from_config(config)
        assert model.include_top is False and model.task_head is None
        assert model.pad_token_id == 0

    def test_keras_round_trip_restores_the_head_by_value(self, tmp_path):
        """Weights compared BEFORE the first call on the loaded model (guide 8.4)."""
        model = GPT2(**_HEAD_KW, include_top=True, task_config=_CLF)
        batch = _head_batch()
        before = _np(model(batch, training=False))
        saved = [_np(w).copy() for w in model.weights]
        head_saved = [_np(w).copy() for w in model.task_head.weights]
        assert head_saved, "the head has no weights to compare"

        path = str(tmp_path / "gpt2_head.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        assert len(loaded.weights) == len(saved)
        for s, w in zip(saved, loaded.weights):
            np.testing.assert_allclose(s, _np(w), atol=0.0, rtol=0.0,
                                       err_msg=f"{w.path} not restored")
        for s, w in zip(head_saved, loaded.task_head.weights):
            np.testing.assert_allclose(s, _np(w), atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(_np(loaded(batch, training=False)), before)

    @pytest.mark.parametrize("include_top", [False, True])
    def test_explicit_build_matches_lazy_build(self, include_top):
        """``build()`` materializes exactly the tree ``call()`` runs (guide 8.1)."""
        kw = dict(_HEAD_KW, tie_word_embeddings=False, include_top=include_top,
                  task_config=_CLF if include_top else None)
        explicit = GPT2(**kw)
        explicit.build((None, _SEQ))
        lazy = GPT2(**kw)
        lazy(_head_batch(), training=False)
        assert _relative_paths(explicit) == _relative_paths(lazy)
        assert explicit.count_params() == lazy.count_params() > 0

    def test_a_disabled_head_builds_no_head_weights(self):
        """Anti-vacuity sibling of the parity arm (guide 8.3)."""
        headless = GPT2(**_HEAD_KW)
        headless(_head_batch(), training=False)
        assert headless.task_head is None
        assert not [w.path for w in headless.weights if "task_head" in w.path]

        headed = GPT2(**_HEAD_KW, include_top=True, task_config=_CLF)
        headed(_head_batch(), training=False)
        assert [w.path for w in headed.weights if "/task_head/" in w.path]

    @pytest.mark.parametrize("tie", [True, False], ids=["tied", "untied"])
    def test_the_encoder_weight_paths_do_not_depend_on_the_head(self, tie):
        """The decoder's paths are identical across tops (D-005 of the plan).

        The backbone is every weight under ``decoder/``. Measured on this
        config: 30 decoder paths; untied headless adds ``lm_head/kernel`` (31);
        a TEXT_CLASSIFICATION head adds 8 under ``task_head/`` (38).
        """
        def weights(**kw):
            m = GPT2(**_HEAD_KW, tie_word_embeddings=tie, name="gpt", **kw)
            m(_head_batch(), training=False)
            return [w.path for w in m.weights]

        headless = weights()
        backbone = [p for p in headless if p.startswith("gpt/decoder/")]
        assert (len(backbone), len(headless)) == (30, 30 if tie else 31)
        assert headless[:len(backbone)] == backbone
        for task in (_CLF, _NER):
            headed = weights(include_top=True, task_config=task)
            assert headed[:len(backbone)] == backbone, task.name
            assert all(p.startswith("gpt/task_head/")
                       for p in headed[len(backbone):]), task.name
            assert len(headed) > len(backbone)

    def test_backbone_weights_transfer_across_heads(self, tmp_path):
        """A backbone trained under one head is reused, same weights, under another.

        The supported route is a local ``pretrained`` path, which copies the
        decoder by layer name and never the checkpoint's ``task_head``
        (D-007, D-012 of plan-2026-10-11T004932-b83935e1). Exercised into a
        TOKEN_CLASSIFICATION model and into a headless one.
        """
        source = create_gpt2("tiny", include_top=True, task_config=_CLF,
                             **_HEAD_KW)
        batch = _head_batch()
        source(batch, training=False)
        _randomized(source, 1)
        path = str(tmp_path / "trained_under_clf.keras")
        source.save(path)
        reference = _backbone_hidden(source, batch)

        token = create_gpt2("tiny", include_top=True, task_config=_NER,
                            pretrained=path, **_HEAD_KW)
        headless = create_gpt2("tiny", pretrained=path, **_HEAD_KW)

        np.testing.assert_array_equal(_backbone_hidden(token, batch), reference)
        np.testing.assert_array_equal(
            _np(headless(batch, training=False)["last_hidden_state"]), reference)
        assert tuple(token(batch, training=False).shape) == (2, _SEQ, 9)

        # Anti-vacuity: a fresh backbone does NOT reproduce the reference.
        fresh = create_gpt2("tiny", **_HEAD_KW)
        assert not np.array_equal(
            _np(fresh(batch, training=False)["last_hidden_state"]), reference)

    def test_an_equal_shape_head_of_another_task_is_not_copied(self, tmp_path):
        """A SPAM head is not filled from a CLF checkpoint's head of equal shape.

        Whole-file loading (the headless route) matches the head by object
        path and would copy it silently, because the shapes agree
        (plan-2026-10-11T004932-b83935e1 D-007, D-012).
        """
        source = create_gpt2("tiny", include_top=True, task_config=_CLF,
                             **_HEAD_KW)
        batch = _head_batch()
        source(batch, training=False)
        _randomized(source, 2)
        path = str(tmp_path / "clf.keras")
        source.save(path)

        spam = create_gpt2("tiny", include_top=True, task_config=_SPAM,
                           pretrained=path, **_HEAD_KW)
        src_head = [_np(w) for w in source.task_head.weights]
        dst_head = [_np(w) for w in spam.task_head.weights]
        assert [a.shape for a in src_head] == [b.shape for b in dst_head], (
            "fixture invalid: the two heads must have equal weight shapes")
        assert not any(np.array_equal(a, b) for a, b in zip(src_head, dst_head))
        np.testing.assert_array_equal(
            _backbone_hidden(spam, batch), _backbone_hidden(source, batch))

    def test_pretrained_headless_checkpoint_into_a_headed_model(self, tmp_path):
        """An untied headless (LM) checkpoint restores a headed model's decoder.

        The checkpoint's ``lm_head`` has no target there and is left out.
        """
        lm = create_gpt2("tiny", tie_word_embeddings=False, **_HEAD_KW)
        batch = _head_batch()
        lm(batch, training=False)
        _randomized(lm, 3)
        expected = _np(lm(batch, training=False)["last_hidden_state"])
        path = str(tmp_path / "lm.keras")
        lm.save(path)

        headed = create_gpt2("tiny", tie_word_embeddings=False, include_top=True,
                             task_config=_CLF, pretrained=path, **_HEAD_KW)
        assert headed.lm_head is None
        np.testing.assert_array_equal(_backbone_hidden(headed, batch), expected)

    def test_a_decoder_that_does_not_fit_raises_for_a_headed_model(self, tmp_path):
        """A vocab mismatch fails loudly instead of loading nothing (D-012, D-024).

        The decoder is ONE layer to the by-name transfer, so a non-strict
        transfer would skip the whole backbone and return normally; leaving
        just the embedding fresh is not possible, so the change is refused.
        """
        lm = create_gpt2("tiny", **_HEAD_KW)
        lm(_head_batch(), training=False)
        path = str(tmp_path / "lm_v64.keras")
        lm.save(path)
        with pytest.raises(ValueError, match="another vocab_size"):
            create_gpt2("tiny", include_top=True, task_config=_CLF,
                        pretrained=path, **dict(_HEAD_KW, vocab_size=80))

    def test_include_top_without_task_config_raises(self):
        with pytest.raises(ValueError, match="include_top=True needs a task_config"):
            GPT2(**_HEAD_KW, include_top=True)

    @pytest.mark.parametrize("dead", [
        {"task_config": _CLF},
        {"head_config_overrides": {"pooling_type": "mean"}},
    ], ids=["task_config", "head_config_overrides"])
    def test_head_knobs_without_include_top_raise(self, dead):
        with pytest.raises(ValueError, match="only take effect with include_top=True"):
            GPT2(**_HEAD_KW, **dead)

    def test_pretrained_true_still_raises_with_a_head(self):
        with pytest.raises(NotImplementedError):
            create_gpt2("tiny", pretrained=True, include_top=True,
                        task_config=_CLF)
