"""Inference algorithm tests for `dl_techniques.models.language.ptrm`.

Tests cover PTRMInference stochastic rollouts, noise injection, Q-head selection,
and pass@k metrics.
"""

import pytest
import numpy as np
from typing import Dict, Any

import keras
import tensorflow as tf

from dl_techniques.models.language.ptrm import PTRM, PTRMInference, run_ptrm_inference


class TestPTRMInferenceBasic:
    """Test basic PTRMInference functionality."""

    @pytest.fixture(scope="class")
    def tiny_model(self) -> PTRM:
        """Create a tiny PTRM model for testing."""
        model = PTRM(
            vocab_size=100,
            hidden_size=32,
            num_heads=4,
            expansion=2.0,
            seq_len=16,
            puzzle_emb_len=4,
            h_layers=1,
            l_layers=1,
            halt_max_steps=4,
            halt_exploration_prob=0.1,
            no_act_continue=True,
        )
        # Build the model
        batch = {"inputs": keras.ops.zeros((2, 16), dtype='int32')}
        carry = model.initial_carry(batch)
        _ = model(carry, batch, training=False)
        return model

    @pytest.fixture(scope="class")
    def sample_batch(self) -> Dict[str, keras.KerasTensor]:
        """Create a sample input batch."""
        return {
            "inputs": keras.ops.convert_to_tensor(
                np.random.randint(0, 100, size=(4, 16)),
                dtype='int32'
            ),
        }

    def test_inference_initialization(self, tiny_model):
        """Test PTRMInference initialization."""
        inference = PTRMInference(tiny_model, noise_scale=0.2)
        assert inference.model is tiny_model
        assert inference.noise_scale == 0.2
        assert inference.use_q_continue is True

    def test_inference_initialization_custom_params(self, tiny_model):
        """Test PTRMInference with custom parameters."""
        inference = PTRMInference(tiny_model, noise_scale=0.5, use_q_continue=False)
        assert inference.noise_scale == 0.5
        assert inference.use_q_continue is False

    def test_inference_invalid_model_raises(self):
        """Test that invalid model raises ValueError."""
        class FakeModel:
            pass
        
        with pytest.raises(ValueError, match="must have `initial_carry` method"):
            PTRMInference(FakeModel())

    def test_inference_call_shapes(self, tiny_model, sample_batch):
        """Test inference call returns correct shapes."""
        inference = PTRMInference(tiny_model, noise_scale=0.2)
        
        best_logits, best_q, all_q = inference(
            sample_batch,
            num_rollouts=5,
            supervision_steps=4,
            training=False,
        )
        
        # Check shapes
        batch_size = sample_batch["inputs"].shape[0]
        vocab_size = tiny_model.vocab_size
        seq_len = tiny_model.seq_len
        
        assert best_logits.shape == (batch_size, seq_len, vocab_size)
        assert best_q.shape == (batch_size,)
        assert all_q.shape == (5, batch_size)  # K=5, B=4

    def test_inference_deterministic_with_zero_noise(self, tiny_model, sample_batch):
        """Test that zero noise produces deterministic results across rollouts."""
        inference = PTRMInference(tiny_model, noise_scale=0.0)
        
        best_logits, best_q, all_q = inference(
            sample_batch,
            num_rollouts=3,
            supervision_steps=2,
            training=False,
        )
        
        # With zero noise, all rollouts should be identical
        # So all_q should have identical values across K dimension
        all_q_np = keras.ops.convert_to_numpy(all_q)
        for b in range(all_q_np.shape[1]):
            assert np.allclose(all_q_np[:, b], all_q_np[0, b], atol=1e-6)

    def test_inference_stochastic_with_noise(self, tiny_model, sample_batch):
        """Test that non-zero noise produces different rollouts."""
        inference = PTRMInference(tiny_model, noise_scale=1.0)
        
        best_logits, best_q, all_q = inference(
            sample_batch,
            num_rollouts=10,
            supervision_steps=3,
            training=False,
        )
        
        # With high noise, rollouts should differ
        all_q_np = keras.ops.convert_to_numpy(all_q)
        # At least some batch items should have variation across rollouts
        has_variation = False
        for b in range(all_q_np.shape[1]):
            if np.max(all_q_np[:, b]) - np.min(all_q_np[:, b]) > 1e-3:
                has_variation = True
                break
        assert has_variation, "Expected variation in Q values with noise injection"

    def test_inference_selects_best_q(self, tiny_model, sample_batch):
        """Test that inference selects rollout with highest Q value."""
        inference = PTRMInference(tiny_model, noise_scale=0.2, use_q_continue=True)
        
        best_logits, best_q, all_q = inference(
            sample_batch,
            num_rollouts=5,
            supervision_steps=2,
            training=False,
        )
        
        # For each batch item, best_q should equal max over rollouts
        all_q_np = keras.ops.convert_to_numpy(all_q)
        best_q_np = keras.ops.convert_to_numpy(best_q)
        
        expected_best = np.max(all_q_np, axis=0)
        np.testing.assert_allclose(best_q_np, expected_best, atol=1e-5)

    def test_inference_use_q_halt(self, tiny_model, sample_batch):
        """Test inference with use_q_continue=False (uses q_halt)."""
        inference = PTRMInference(tiny_model, noise_scale=0.2, use_q_continue=False)
        
        best_logits, best_q, all_q = inference(
            sample_batch,
            num_rollouts=5,
            supervision_steps=2,
            training=False,
        )
        
        # Should still return correct shapes
        assert best_logits.shape == (4, 16, 100)
        assert best_q.shape == (4,)
        assert all_q.shape == (5, 4)


class TestPTRMInferenceNoiseInjection:
    """Test noise injection mechanism."""

    def test_noise_injection_changes_latent_states(self, tiny_model, sample_batch):
        """Test that _inject_noise actually modifies z_H and z_L."""
        inference = PTRMInference(tiny_model, noise_scale=1.0)
        
        carry = tiny_model.initial_carry(sample_batch)
        z_H_before = carry["inner_carry"]["z_H"]
        z_L_before = carry["inner_carry"]["z_L"]
        
        noisy_carry = inference._inject_noise(carry)
        z_H_after = noisy_carry["inner_carry"]["z_H"]
        z_L_after = noisy_carry["inner_carry"]["z_L"]
        
        # Noise should change the states
        diff_H = keras.ops.convert_to_numpy(z_H_after - z_H_before)
        diff_L = keras.ops.convert_to_numpy(z_L_after - z_L_before)
        
        assert np.std(diff_H) > 0.1, "z_H should change with noise"
        assert np.std(diff_L) > 0.1, "z_L should change with noise"
        
        # Mean should be close to zero (zero-mean noise)
        assert abs(np.mean(diff_H)) < 0.1, "Noise should be zero-mean"
        assert abs(np.mean(diff_L)) < 0.1, "Noise should be zero-mean"

    def test_noise_scale_parameter(self, tiny_model, sample_batch):
        """Test that noise_scale parameter controls noise magnitude."""
        carry = tiny_model.initial_carry(sample_batch)
        
        # Small noise
        inference_small = PTRMInference(tiny_model, noise_scale=0.01)
        carry_small = inference_small._inject_noise(carry)
        diff_small = keras.ops.convert_to_numpy(
            carry_small["inner_carry"]["z_H"] - carry["inner_carry"]["z_H"]
        )
        
        # Large noise
        inference_large = PTRMInference(tiny_model, noise_scale=2.0)
        carry_large = inference_large._inject_noise(carry)
        diff_large = keras.ops.convert_to_numpy(
            carry_large["inner_carry"]["z_H"] - carry["inner_carry"]["z_H"]
        )
        
        # Large noise should have larger variance
        assert np.std(diff_large) > 10 * np.std(diff_small)

    def test_noise_preserves_carry_structure(self, tiny_model, sample_batch):
        """Test that noise injection preserves carry structure."""
        inference = PTRMInference(tiny_model, noise_scale=0.5)
        carry = tiny_model.initial_carry(sample_batch)
        
        noisy_carry = inference._inject_noise(carry)
        
        # All keys should be present
        assert "inner_carry" in noisy_carry
        assert "steps" in noisy_carry
        assert "halted" in noisy_carry
        assert "current_data" in noisy_carry
        
        # inner_carry should have z_H and z_L
        assert "z_H" in noisy_carry["inner_carry"]
        assert "z_L" in noisy_carry["inner_carry"]
        
        # Non-noisy parts should be unchanged
        assert noisy_carry["steps"] is carry["steps"]
        assert noisy_carry["halted"] is carry["halted"]
        assert noisy_carry["current_data"] is carry["current_data"]


class TestPTRMInferenceConvenienceFunction:
    """Test run_ptrm_inference convenience function."""

    def test_run_ptrm_inference(self, tiny_model, sample_batch):
        """Test convenience function returns correct values."""
        best_logits, best_q = run_ptrm_inference(
            tiny_model,
            sample_batch,
            num_rollouts=3,
            supervision_steps=2,
            noise_scale=0.2,
            use_q_continue=True,
            training=False,
        )
        
        assert best_logits.shape == (4, 16, 100)
        assert best_q.shape == (4,)

    def test_run_ptrm_inference_equivalent_to_class(self, tiny_model, sample_batch):
        """Test convenience function matches class-based API."""
        # Class-based
        inference = PTRMInference(tiny_model, noise_scale=0.2, use_q_continue=True)
        best_logits1, best_q1, _ = inference(sample_batch, num_rollouts=3, supervision_steps=2)
        
        # Function-based
        best_logits2, best_q2 = run_ptrm_inference(
            tiny_model, sample_batch, num_rollouts=3, supervision_steps=2,
            noise_scale=0.2, use_q_continue=True
        )
        
        # Results should be identical (same random seed not set, but structure same)
        assert best_logits1.shape == best_logits2.shape
        assert best_q1.shape == best_q2.shape


class TestPTRMInferencePassAtK:
    """Test pass@k metric computation."""

    def test_compute_pass_at_k(self, tiny_model, sample_batch):
        """Test pass@k computation returns expected metrics."""
        inference = PTRMInference(tiny_model, noise_scale=0.2)
        
        # Create labels matching the inputs for deterministic test
        labels = sample_batch["inputs"]
        
        metrics = inference.compute_pass_at_k(
            sample_batch,
            labels,
            num_rollouts=5,
            supervision_steps=2,
        )
        
        assert "pass_at_1" in metrics
        assert "pass_at_k" in metrics
        assert 0.0 <= metrics["pass_at_1"] <= 1.0
        assert 0.0 <= metrics["pass_at_k"] <= 1.0
        assert metrics["pass_at_k"] >= metrics["pass_at_1"]

    def test_pass_at_k_perfect_match(self, tiny_model, tiny_config):
        """Test pass@k when labels match a rollout perfectly."""
        # Create a batch where we know the answer - use model's seq_len
        seq_len = tiny_config['seq_len']
        batch = {"inputs": keras.ops.zeros((2, seq_len), dtype='int32')}
        labels = keras.ops.zeros((2, seq_len), dtype='int32')
        
        inference = PTRMInference(tiny_model, noise_scale=0.0)  # Deterministic
        
        # With zero noise, rollout 0 is deterministic
        metrics = inference.compute_pass_at_k(
            batch, labels, num_rollouts=3, supervision_steps=2
        )
        
        # pass_at_1 and pass_at_k should be same (all rollouts identical)
        np.testing.assert_allclose(metrics["pass_at_1"], metrics["pass_at_k"], atol=1e-6)


class TestPTRMInferenceMultipleBatchSizes:
    """Test inference works with various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
    def test_various_batch_sizes(self, tiny_model, batch_size):
        """Test inference with different batch sizes."""
        inference = PTRMInference(tiny_model, noise_scale=0.2)
        batch = {"inputs": keras.ops.zeros((batch_size, 16), dtype='int32')}
        
        best_logits, best_q, all_q = inference(
            batch, num_rollouts=3, supervision_steps=2
        )
        
        assert best_logits.shape == (batch_size, 16, 100)
        assert best_q.shape == (batch_size,)
        assert all_q.shape == (3, batch_size)


class TestPTRMInferenceEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_rollout(self, tiny_model, sample_batch):
        """Test inference with K=1 (degenerates to deterministic)."""
        inference = PTRMInference(tiny_model, noise_scale=0.2)
        
        best_logits, best_q, all_q = inference(
            sample_batch, num_rollouts=1, supervision_steps=2
        )
        
        assert best_logits.shape == (4, 16, 100)
        assert best_q.shape == (4,)
        assert all_q.shape == (1, 4)

    def test_single_supervision_step(self, tiny_model, sample_batch):
        """Test inference with D=1."""
        inference = PTRMInference(tiny_model, noise_scale=0.2)
        
        best_logits, best_q, all_q = inference(
            sample_batch, num_rollouts=3, supervision_steps=1
        )
        
        assert best_logits.shape == (4, 16, 100)
        assert best_q.shape == (4,)
        assert all_q.shape == (3, 4)

    def test_training_mode_flag(self, tiny_model, sample_batch):
        """Test that training flag is passed to model."""
        inference = PTRMInference(tiny_model, noise_scale=0.2)
        
        # Should not raise
        best_logits, best_q, _ = inference(
            sample_batch, num_rollouts=2, supervision_steps=2, training=True
        )
        best_logits, best_q, _ = inference(
            sample_batch, num_rollouts=2, supervision_steps=2, training=False
        )


class TestPTRMInferenceDeterministicBaseline:
    """Test that PTRM with zero noise matches deterministic TRM."""

    def test_zero_noise_matches_deterministic(self, tiny_model, sample_batch):
        """PTRM with σ=0 should match standard TRM inference."""
        # Run deterministic TRM
        carry = tiny_model.initial_carry(sample_batch)
        for _ in range(3):
            carry, outputs = tiny_model(carry, sample_batch, training=False)
        det_logits = outputs["logits"]
        det_q = outputs["q_continue_logits"]
        
        # Run PTRM with zero noise, K=1
        inference = PTRMInference(tiny_model, noise_scale=0.0)
        best_logits, best_q, _ = inference(
            sample_batch, num_rollouts=1, supervision_steps=3
        )
        
        # Should match (within numerical precision)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(best_logits),
            keras.ops.convert_to_numpy(det_logits),
            atol=1e-5, rtol=1e-5,
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(best_q),
            keras.ops.convert_to_numpy(det_q),
            atol=1e-5, rtol=1e-5,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])