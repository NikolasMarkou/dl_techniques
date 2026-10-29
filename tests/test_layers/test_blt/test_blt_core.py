"""Tests for ``ByteLatentReasoningCore`` (``dl_techniques.layers.blt.blt_core``).

This is the FIRST test for this layer. The layer was dead-on-forward until the
3-bug-chain fix (entropy 3-D tile, ``stop_gradient`` is a function not a context
manager, HRM list-call) landed; these tests pin the resurrected behaviour.

Key constraint (see plan findings [NEW iter-1]): the ``.keras`` round-trip MUST
build the core EAGERLY (construct + call on concrete tensors inside the wrapper
``__init__``). A symbolic / functional trace lazily builds the core's attention
sub-layers inside a scratch graph and trips an out-of-scope graph-tensor leak in
``MultiHeadCrossAttention.__init__`` — a pre-existing attention-constructor bug,
unrelated to this layer and out of scope here.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from typing import Any, Dict

from dl_techniques.layers.blt.blt_core import ByteLatentReasoningCore
from tests.layer_naming import PublicAutoName


class TestByteLatentReasoningCore:
    """Comprehensive test suite for ByteLatentReasoningCore."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------
    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Smallest config that exercises every forward path (puzzle disabled)."""
        return {
            "vocab_size": 260,
            "seq_len": 32,
            "embed_dim": 16,
            "local_dim": 16,
            "global_dim": 16,
            "max_patches": 4,
            "h_layers": 1,
            "l_layers": 1,
            "h_cycles": 1,
            "l_cycles": 1,
            "num_heads": 2,
            "batch_size": 2,
            "puzzle_emb_dim": 0,
        }

    @pytest.fixture
    def batch_size(self) -> int:
        return 2

    @pytest.fixture
    def sample_inputs(self, layer_config, batch_size) -> Dict[str, Any]:
        """Sample byte-token input dict ``{"byte_tokens": (B, seq_len) int32}``."""
        return {
            "byte_tokens": keras.ops.convert_to_tensor(
                np.random.randint(
                    0, layer_config["vocab_size"],
                    (batch_size, layer_config["seq_len"]),
                ).astype(np.int32)
            ),
        }

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def test_initialization(self, layer_config):
        """Test layer initialization stores config and defers sub-layer build."""
        layer = ByteLatentReasoningCore(**layer_config)

        assert layer.vocab_size == 260
        assert layer.seq_len == 32
        assert layer.embed_dim == 16
        assert layer.max_patches == 4
        assert layer.num_heads == 2
        assert not layer.built

        # Sub-layers are created in build(), not __init__.
        assert layer.h_reasoning is None
        assert layer.l_reasoning is None
        assert layer.entropy_model is None

    def test_empty_carry(self, layer_config, batch_size):
        """Test empty carry has the documented keys + shapes."""
        layer = ByteLatentReasoningCore(**layer_config)
        carry = layer.empty_carry(batch_size)

        assert set(carry.keys()) == {"z_h", "z_l", "patch_context", "entropy_state"}
        assert carry["z_h"].shape == (batch_size, 4, 16)
        assert carry["z_l"].shape == (batch_size, 4, 16)
        assert carry["patch_context"].shape == (batch_size, 4, 16)
        assert carry["entropy_state"].shape == (batch_size, 32)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def test_forward(self, layer_config, sample_inputs, batch_size):
        """Forward returns (new_carry, outputs) with correct keys/shapes/finite."""
        layer = ByteLatentReasoningCore(**layer_config)
        carry = layer.empty_carry(batch_size)

        new_carry, outputs = layer(carry, sample_inputs, training=False)

        # Carry structure preserved.
        assert set(new_carry.keys()) == {
            "z_h", "z_l", "patch_context", "entropy_state"
        }
        assert new_carry["z_h"].shape == carry["z_h"].shape
        assert new_carry["z_l"].shape == carry["z_l"].shape

        # Output structure + shapes.
        assert set(outputs.keys()) == {
            "logits", "q_halt_logits", "q_continue_logits",
            "entropy", "patch_lengths",
        }
        assert outputs["logits"].shape == (batch_size, 32, 260)
        assert outputs["q_halt_logits"].shape == (batch_size,)
        assert outputs["q_continue_logits"].shape == (batch_size,)
        assert outputs["entropy"].shape == (batch_size, 32)
        assert outputs["patch_lengths"].shape == (batch_size, 4)

        # Finiteness of the differentiable outputs.
        for key in ("logits", "q_halt_logits", "q_continue_logits", "entropy"):
            arr = keras.ops.convert_to_numpy(outputs[key])
            assert np.all(np.isfinite(arr)), f"{key} contains non-finite values"

    def test_learned_positions_run_a_forward_pass(
        self, layer_config, sample_inputs, batch_size
    ):
        """``pos_encodings='learned'`` used to read a table the layer does not
        have (``position_embeddings``) and raise at call time."""
        layer = ByteLatentReasoningCore(**layer_config, pos_encodings="learned")
        carry = layer.empty_carry(batch_size)
        _, outputs = layer(carry, sample_inputs, training=False)

        assert outputs["logits"].shape == (batch_size, 32, 260)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(outputs["logits"])))
        assert layer.position_embedding.pos_embedding.shape == (1, 4, 16)

    def test_the_learned_table_changes_the_output(
        self, layer_config, sample_inputs, batch_size
    ):
        layer = ByteLatentReasoningCore(**layer_config, pos_encodings="learned")
        carry = layer.empty_carry(batch_size)
        _, before = layer(carry, sample_inputs, training=False)

        layer.position_embedding.pos_embedding.assign(
            keras.ops.ones_like(layer.position_embedding.pos_embedding)
        )
        _, after = layer(carry, sample_inputs, training=False)
        difference = np.abs(
            keras.ops.convert_to_numpy(before["logits"])
            - keras.ops.convert_to_numpy(after["logits"])
        ).max()
        assert difference > 0.0

    @pytest.mark.parametrize("pos_encodings,attention_type", [
        ("rope", "group_query"),
        ("learned", "multi_head"),
    ])
    def test_reasoning_modules_get_the_positional_settings(
        self, layer_config, sample_inputs, batch_size, pos_encodings, attention_type
    ):
        """RoPE lives inside attention: ``rope_theta`` and the patch count must
        reach both reasoning modules, and a learned table gets plain attention."""
        layer = ByteLatentReasoningCore(
            **layer_config, pos_encodings=pos_encodings, rope_theta=500.0
        )
        layer(layer.empty_carry(batch_size), sample_inputs, training=False)

        for module in (layer.h_reasoning, layer.l_reasoning):
            assert module.attention_type == attention_type
            assert module.rope_theta == 500.0
            assert module.max_seq_len == layer_config["max_patches"]

    def test_no_sublayer_is_built_and_then_never_called(self, layer_config):
        """``tokenizer``, ``lm_head`` and ``rope`` were created in ``build`` and
        never used: logits come from ``local_decoder`` and RoPE from attention."""
        layer = ByteLatentReasoningCore(**layer_config)
        for name in ("tokenizer", "lm_head", "rope"):
            assert not hasattr(layer, name), name

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_inputs, batch_size, training):
        """Forward runs under all training-flag values."""
        layer = ByteLatentReasoningCore(**layer_config)
        carry = layer.empty_carry(batch_size)
        _, outputs = layer(carry, sample_inputs, training=training)
        assert outputs["logits"].shape == (batch_size, 32, 260)

    # ------------------------------------------------------------------
    # .keras round-trip (EAGER pre-build — see module docstring)
    # ------------------------------------------------------------------
    def test_keras_roundtrip(self, layer_config, sample_inputs, batch_size):
        """Save/load a Model wrapping the EAGERLY pre-built core; outputs match."""

        @keras.saving.register_keras_serializable(package="test_blt_core")
        class _CoreWrapper(PublicAutoName, keras.Model):
            """Thin wrapper that builds the core EAGERLY in __init__.

            NOT a functional/symbolic trace: a symbolic trace lazily constructs
            the core's MultiHeadCrossAttention inside a scratch graph and leaks an
            out-of-scope graph tensor (findings [NEW iter-1]). Calling the core on
            concrete tensors here forces an eager build with real weights.
            """

            def __init__(self, cfg, batch, **kw):
                super().__init__(**kw)
                self._cfg = dict(cfg)
                self._batch = batch
                self.core = ByteLatentReasoningCore(**cfg)
                dummy_carry = self.core.empty_carry(batch)
                dummy_inputs = {
                    "byte_tokens": keras.ops.convert_to_tensor(
                        np.zeros((batch, cfg["seq_len"]), dtype=np.int32)
                    )
                }
                self.core(dummy_carry, dummy_inputs, training=False)

            def call(self, inputs, training=None):
                carry = self.core.empty_carry(self._batch)
                _, outputs = self.core(carry, inputs, training=training)
                return outputs

            def get_config(self):
                return {"cfg": self._cfg, "batch": self._batch}

            @classmethod
            def from_config(cls, config):
                return cls(config["cfg"], config["batch"])

        model = _CoreWrapper(layer_config, batch_size)
        out_before = model(sample_inputs, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "blt_core_model.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)

        out_after = reloaded(sample_inputs, training=False)

        # The entropy_model / patcher are deterministic at eval; with weights
        # round-tripped the reload must reproduce the pre-save outputs.
        logits_before = keras.ops.convert_to_numpy(out_before["logits"])
        logits_after = keras.ops.convert_to_numpy(out_after["logits"])
        assert logits_after.shape == logits_before.shape
        assert np.all(np.isfinite(logits_after))
        np.testing.assert_allclose(
            logits_after, logits_before, atol=1e-5, rtol=1e-5,
            err_msg=".keras round-trip changed the logits (weight-restore drift)",
        )

    # ------------------------------------------------------------------
    # Gradient flow
    # ------------------------------------------------------------------
    def test_gradient_flow(self, layer_config, sample_inputs, batch_size):
        """Gradients are computable, non-None for most vars, and finite.

        The reasoning cycle loop is intentionally DETACHED (truncated BPTT) — only
        the final reasoning step carries gradient. A handful of vars legitimately
        receive None gradient: ``h_init``/``l_init`` are used only by
        ``reset_carry`` (never in ``call``), and one Dense kernel/bias pair sits
        off the scalar-loss path. So we assert the gradient computation SUCCEEDS,
        that the vast majority of trainable vars get a gradient, and that every
        non-None gradient is finite — we do NOT assert every var is non-None.
        """
        layer = ByteLatentReasoningCore(**layer_config)
        carry = layer.empty_carry(batch_size)

        with tf.GradientTape() as tape:
            _, outputs = layer(carry, sample_inputs, training=True)
            loss = keras.ops.mean(outputs["logits"])

        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) > 0

        non_none = [g for g in grads if g is not None]
        # The overwhelming majority must receive a gradient (detached design + a
        # few off-path vars account for the rest).
        assert len(non_none) > 0
        assert len(non_none) >= len(grads) - 4

        for g in non_none:
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(g))), \
                "a trainable variable received a non-finite gradient"

    def test_carry_detachment(self, layer_config, sample_inputs, batch_size):
        """A loss on the returned carry yields no gradients (carry is detached)."""
        layer = ByteLatentReasoningCore(**layer_config)
        carry = layer.empty_carry(batch_size)

        with tf.GradientTape() as tape:
            new_carry, _ = layer(carry, sample_inputs, training=True)
            loss = keras.ops.mean(new_carry["z_h"])

        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is None for g in grads)

    # ------------------------------------------------------------------
    # Config round-trip
    # ------------------------------------------------------------------
    def test_config_completeness(self, layer_config):
        """get_config() contains every __init__ parameter."""
        layer = ByteLatentReasoningCore(**layer_config)
        config = layer.get_config()

        expected_keys = {
            "vocab_size", "seq_len", "embed_dim", "local_dim", "global_dim",
            "max_patches", "num_puzzle_identifiers", "puzzle_emb_dim",
            "batch_size", "h_layers", "l_layers", "h_cycles", "l_cycles",
            "num_heads", "entropy_threshold", "pos_encodings", "rope_theta",
            "dropout_rate", "use_bias", "embeddings_initializer",
            "kernel_initializer", "embeddings_regularizer", "kernel_regularizer",
        }
        for key in expected_keys:
            assert key in config, f"Missing {key} in get_config()"

    def test_config_roundtrip(self, layer_config):
        """from_config(get_config()) rebuilds an equivalent layer."""
        layer = ByteLatentReasoningCore(**layer_config)
        config = layer.get_config()

        reconstructed = ByteLatentReasoningCore.from_config(config)

        assert reconstructed.vocab_size == layer.vocab_size
        assert reconstructed.seq_len == layer.seq_len
        assert reconstructed.embed_dim == layer.embed_dim
        assert reconstructed.local_dim == layer.local_dim
        assert reconstructed.global_dim == layer.global_dim
        assert reconstructed.max_patches == layer.max_patches
        assert reconstructed.num_heads == layer.num_heads
        assert reconstructed.h_cycles == layer.h_cycles
        assert reconstructed.l_cycles == layer.l_cycles
        assert reconstructed.puzzle_emb_dim == layer.puzzle_emb_dim
