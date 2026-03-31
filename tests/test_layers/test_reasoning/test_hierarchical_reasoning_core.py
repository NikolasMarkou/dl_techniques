import pytest
import numpy as np
import keras
from keras import ops
from typing import Any, Dict

from dl_techniques.layers.reasoning.hrm_reasoning_core import HierarchicalReasoningCore


class TestHierarchicalReasoningCore:
    """Comprehensive test suite for HierarchicalReasoningCore."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing (small dims for speed)."""
        return {
            'vocab_size': 50,
            'seq_len': 8,
            'embed_dim': 32,
            'num_puzzle_identifiers': 10,
            'puzzle_emb_dim': 0,
            'batch_size': 2,
            'h_layers': 1,
            'l_layers': 1,
            'h_cycles': 1,
            'l_cycles': 1,
            'num_heads': 4,
            'ffn_expansion_factor': 2,
            'pos_encodings': 'rope',
        }

    @pytest.fixture
    def config_with_puzzle(self) -> Dict[str, Any]:
        """Configuration with puzzle embeddings enabled."""
        return {
            'vocab_size': 50,
            'seq_len': 8,
            'embed_dim': 32,
            'num_puzzle_identifiers': 10,
            'puzzle_emb_dim': 16,
            'batch_size': 2,
            'h_layers': 1,
            'l_layers': 1,
            'h_cycles': 1,
            'l_cycles': 1,
            'num_heads': 4,
            'ffn_expansion_factor': 2,
            'pos_encodings': 'rope',
        }

    @pytest.fixture
    def config_learned_pos(self) -> Dict[str, Any]:
        """Configuration with learned positional embeddings."""
        return {
            'vocab_size': 50,
            'seq_len': 8,
            'embed_dim': 32,
            'num_puzzle_identifiers': 10,
            'puzzle_emb_dim': 0,
            'batch_size': 2,
            'h_layers': 1,
            'l_layers': 1,
            'h_cycles': 1,
            'l_cycles': 1,
            'num_heads': 4,
            'ffn_expansion_factor': 2,
            'pos_encodings': 'learned',
        }

    @pytest.fixture
    def sample_carry(self, layer_config) -> Dict[str, Any]:
        """Sample carry state."""
        batch_size = layer_config['batch_size']
        seq_len = layer_config['seq_len']
        embed_dim = layer_config['embed_dim']
        return {
            "z_h": ops.zeros((batch_size, seq_len, embed_dim)),
            "z_l": ops.zeros((batch_size, seq_len, embed_dim)),
        }

    @pytest.fixture
    def sample_inputs(self, layer_config) -> Dict[str, Any]:
        """Sample input dictionary."""
        batch_size = layer_config['batch_size']
        seq_len = layer_config['seq_len']
        vocab_size = layer_config['vocab_size']
        return {
            "token_ids": ops.convert_to_tensor(
                np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32)
            ),
        }

    def test_initialization(self, layer_config):
        """Test layer initialization."""
        layer = HierarchicalReasoningCore(**layer_config)

        assert layer.vocab_size == 50
        assert layer.seq_len == 8
        assert layer.embed_dim == 32
        assert layer.num_heads == 4
        assert layer.pos_encodings == 'rope'
        assert not layer.built

        # Sub-layers created in __init__
        assert layer.token_embedding is not None
        assert layer.h_reasoning is not None
        assert layer.l_reasoning is not None
        assert layer.lm_head is not None
        assert layer.q_head is not None

    def test_initialization_with_puzzle(self, config_with_puzzle):
        """Test initialization with puzzle embeddings."""
        layer = HierarchicalReasoningCore(**config_with_puzzle)

        assert layer.puzzle_emb_dim == 16
        assert layer.puzzle_embedding is not None
        assert layer.puzzle_emb_len > 0
        assert layer.total_seq_len == layer.seq_len + layer.puzzle_emb_len

    def test_initialization_learned_pos(self, config_learned_pos):
        """Test initialization with learned positional embeddings."""
        layer = HierarchicalReasoningCore(**config_learned_pos)

        assert layer.pos_encodings == 'learned'
        assert layer.position_embedding is not None
        assert layer.rope is None

    def test_initialization_rope_pos(self, layer_config):
        """Test initialization with RoPE positional embeddings."""
        layer = HierarchicalReasoningCore(**layer_config)

        assert layer.pos_encodings == 'rope'
        assert layer.rope is not None
        assert layer.position_embedding is None

    def test_empty_carry(self, layer_config):
        """Test empty carry generation."""
        layer = HierarchicalReasoningCore(**layer_config)

        carry = layer.empty_carry(batch_size=2)

        assert "z_h" in carry
        assert "z_l" in carry
        assert carry["z_h"].shape == (2, layer.total_seq_len, 32)
        assert carry["z_l"].shape == (2, layer.total_seq_len, 32)

    def test_forward_pass_rope(self, layer_config):
        """Test forward pass with RoPE positional encodings."""
        layer = HierarchicalReasoningCore(**layer_config)

        carry = layer.empty_carry(batch_size=2)
        inputs = {
            "token_ids": ops.convert_to_tensor(
                np.random.randint(0, 50, (2, 8)).astype(np.int32)
            ),
        }

        new_carry, outputs = layer(carry, inputs, training=False)

        # Check carry shapes
        assert new_carry["z_h"].shape == carry["z_h"].shape
        assert new_carry["z_l"].shape == carry["z_l"].shape

        # Check output shapes
        assert outputs["logits"].shape == (2, 8, 50)
        assert outputs["q_halt_logits"].shape == (2,)
        assert outputs["q_continue_logits"].shape == (2,)

    def test_forward_pass_learned_pos(self, config_learned_pos):
        """Test forward pass with learned positional encodings."""
        layer = HierarchicalReasoningCore(**config_learned_pos)

        carry = layer.empty_carry(batch_size=2)
        inputs = {
            "token_ids": ops.convert_to_tensor(
                np.random.randint(0, 50, (2, 8)).astype(np.int32)
            ),
        }

        new_carry, outputs = layer(carry, inputs, training=False)

        assert outputs["logits"].shape == (2, 8, 50)
        assert outputs["q_halt_logits"].shape == (2,)

    def test_forward_pass_with_puzzle(self, config_with_puzzle):
        """Test forward pass with puzzle embeddings."""
        layer = HierarchicalReasoningCore(**config_with_puzzle)

        carry = layer.empty_carry(batch_size=2)
        inputs = {
            "token_ids": ops.convert_to_tensor(
                np.random.randint(0, 50, (2, 8)).astype(np.int32)
            ),
            "puzzle_ids": ops.convert_to_tensor(
                np.array([0, 5], dtype=np.int32)
            ),
        }

        new_carry, outputs = layer(carry, inputs, training=False)

        assert outputs["logits"].shape == (2, 8, 50)
        assert outputs["q_halt_logits"].shape == (2,)

    def test_multiple_reasoning_cycles(self):
        """Test with multiple reasoning cycles."""
        layer = HierarchicalReasoningCore(
            vocab_size=50,
            seq_len=4,
            embed_dim=16,
            num_puzzle_identifiers=10,
            batch_size=2,
            h_layers=1,
            l_layers=1,
            h_cycles=2,
            l_cycles=2,
            num_heads=2,
            ffn_expansion_factor=2,
        )

        carry = layer.empty_carry(batch_size=2)
        inputs = {
            "token_ids": ops.convert_to_tensor(
                np.random.randint(0, 50, (2, 4)).astype(np.int32)
            ),
        }

        new_carry, outputs = layer(carry, inputs, training=False)

        assert outputs["logits"].shape == (2, 4, 50)

    def test_reset_carry(self, layer_config):
        """Test carry state resetting."""
        layer = HierarchicalReasoningCore(**layer_config)

        # Build the layer first by doing a forward pass
        carry = layer.empty_carry(batch_size=2)
        inputs = {
            "token_ids": ops.convert_to_tensor(
                np.random.randint(0, 50, (2, 8)).astype(np.int32)
            ),
        }
        new_carry, _ = layer(carry, inputs, training=False)

        # Reset first sequence, keep second
        reset_flag = ops.convert_to_tensor(np.array([True, False]))
        reset_carry = layer.reset_carry(reset_flag, new_carry)

        assert reset_carry["z_h"].shape == new_carry["z_h"].shape
        assert reset_carry["z_l"].shape == new_carry["z_l"].shape

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = HierarchicalReasoningCore(**layer_config)
        config = layer.get_config()

        expected_keys = {
            'vocab_size', 'seq_len', 'embed_dim', 'num_puzzle_identifiers',
            'puzzle_emb_dim', 'batch_size', 'h_layers', 'l_layers',
            'h_cycles', 'l_cycles', 'num_heads', 'ffn_expansion_factor',
            'pos_encodings', 'rope_theta', 'dropout_rate', 'use_bias',
            'embeddings_initializer', 'kernel_initializer',
            'embeddings_regularizer', 'kernel_regularizer',
        }

        for key in expected_keys:
            assert key in config, f"Missing {key} in get_config()"

        assert config['vocab_size'] == layer_config['vocab_size']
        assert config['seq_len'] == layer_config['seq_len']
        assert config['embed_dim'] == layer_config['embed_dim']

    def test_config_roundtrip(self, layer_config):
        """Test config-based reconstruction."""
        layer = HierarchicalReasoningCore(**layer_config)
        config = layer.get_config()

        reconstructed = HierarchicalReasoningCore.from_config(config)

        assert reconstructed.vocab_size == layer.vocab_size
        assert reconstructed.seq_len == layer.seq_len
        assert reconstructed.embed_dim == layer.embed_dim
        assert reconstructed.num_heads == layer.num_heads
        assert reconstructed.pos_encodings == layer.pos_encodings

    def test_compute_output_shape(self, layer_config):
        """Test output shape computation."""
        layer = HierarchicalReasoningCore(**layer_config)

        input_shape = {"token_ids": (2, 8)}
        carry_shape, output_shape = layer.compute_output_shape(input_shape)

        assert "z_h" in carry_shape
        assert "z_l" in carry_shape
        assert "logits" in output_shape
        assert "q_halt_logits" in output_shape
        assert "q_continue_logits" in output_shape
        assert output_shape["logits"][-1] == 50

    def test_gradients_flow(self, layer_config):
        """Test gradient computation."""
        import tensorflow as tf

        layer = HierarchicalReasoningCore(**layer_config)

        carry = layer.empty_carry(batch_size=2)
        inputs = {
            "token_ids": ops.convert_to_tensor(
                np.random.randint(0, 50, (2, 8)).astype(np.int32)
            ),
        }

        with tf.GradientTape() as tape:
            new_carry, outputs = layer(carry, inputs, training=True)
            loss = ops.mean(outputs["logits"])

        gradients = tape.gradient(loss, layer.trainable_variables)
        assert len(gradients) > 0
        # At least some gradients should be non-None
        non_none = [g for g in gradients if g is not None]
        assert len(non_none) > 0

    def test_edge_cases(self):
        """Test error conditions."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            HierarchicalReasoningCore(
                vocab_size=0, seq_len=8, embed_dim=32,
                num_puzzle_identifiers=10, batch_size=2
            )

        with pytest.raises(ValueError, match="must be divisible"):
            HierarchicalReasoningCore(
                vocab_size=50, seq_len=8, embed_dim=33,
                num_puzzle_identifiers=10, batch_size=2, num_heads=4
            )

        with pytest.raises(ValueError, match="pos_encodings must be"):
            HierarchicalReasoningCore(
                vocab_size=50, seq_len=8, embed_dim=32,
                num_puzzle_identifiers=10, batch_size=2,
                pos_encodings='invalid'
            )

        with pytest.raises(ValueError, match="dropout_rate must be between"):
            HierarchicalReasoningCore(
                vocab_size=50, seq_len=8, embed_dim=32,
                num_puzzle_identifiers=10, batch_size=2,
                dropout_rate=1.5
            )

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, training):
        """Test behavior in different training modes."""
        layer = HierarchicalReasoningCore(**layer_config)

        carry = layer.empty_carry(batch_size=2)
        inputs = {
            "token_ids": ops.convert_to_tensor(
                np.random.randint(0, 50, (2, 8)).astype(np.int32)
            ),
        }

        new_carry, outputs = layer(carry, inputs, training=training)
        assert outputs["logits"].shape == (2, 8, 50)

    def test_stateful_behavior(self, layer_config):
        """Test that carry state evolves across steps."""
        layer = HierarchicalReasoningCore(**layer_config)

        carry = layer.empty_carry(batch_size=2)
        inputs = {
            "token_ids": ops.convert_to_tensor(
                np.random.randint(0, 50, (2, 8)).astype(np.int32)
            ),
        }

        # First step
        carry_1, outputs_1 = layer(carry, inputs, training=False)

        # Second step with updated carry
        carry_2, outputs_2 = layer(carry_1, inputs, training=False)

        # Outputs should differ because carry state changed
        assert not np.allclose(
            ops.convert_to_numpy(outputs_1["logits"]),
            ops.convert_to_numpy(outputs_2["logits"]),
            atol=1e-5
        )

    def test_carry_detachment(self, layer_config):
        """Test that carry states are detached from computation graph."""
        import tensorflow as tf

        layer = HierarchicalReasoningCore(**layer_config)

        carry = layer.empty_carry(batch_size=2)
        inputs = {
            "token_ids": ops.convert_to_tensor(
                np.random.randint(0, 50, (2, 8)).astype(np.int32)
            ),
        }

        with tf.GradientTape() as tape:
            new_carry, outputs = layer(carry, inputs, training=True)
            # Try to compute gradient through carry — should be None
            loss = ops.mean(new_carry["z_h"])

        grad = tape.gradient(loss, layer.trainable_variables)
        # All gradients should be None since carry is stop_gradient'd
        assert all(g is None for g in grad)
