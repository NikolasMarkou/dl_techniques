"""Tests for the GPT-2 model implementation.

Covers initialization, parameter validation, forward pass, weight tying,
causal masking, serialization, variant creation, and gradient flow.
"""

import pytest
import numpy as np
import keras
import tensorflow as tf

from dl_techniques.models.gpt2.gpt2 import GPT2


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
        from dl_techniques.models.gpt2 import GPT2 as PkgGPT2, create_gpt2

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
        import dl_techniques.models.gpt2 as pkg

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
    - max |delta| at MASKED positions:   0.8897 (the perturbation is real).
    - same perturbation with NO mask:    0.1973 at the unmasked positions
      (i.e. the isolation is due to the mask, not to causality).

    Logits are O(0.75), so the ``< 1e-6`` bar is a two-sided margin, not a
    bit-identity claim.
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
        Measured 0.8897 against logits of magnitude ~0.75. Bar: > 1e-2."""
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
        Measured 0.1973. Bar: > 1e-2."""
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
        assert leak > 1e-2, (
            f"unmasked run shows no influence from the prefix tokens "
            f"(max |delta| = {leak:.4e}); the masked run's 0.0 would then be "
            f"explained by causality alone and prove nothing about masking"
        )
