"""Inference algorithm tests for `dl_techniques.models.language.ptrm`.

Tests cover PTRMInference stochastic rollouts, noise injection, Q-head selection,
and pass@k metrics.
"""

import itertools

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
        # The paper selects by the halting logit q_halt (D-018).
        assert inference.use_q_continue is False

    def test_inference_initialization_custom_params(self, tiny_model):
        """Test PTRMInference with custom parameters."""
        inference = PTRMInference(tiny_model, noise_scale=0.5, use_q_continue=True)
        assert inference.noise_scale == 0.5
        assert inference.use_q_continue is True

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
        
        # With high noise, the rollouts of EVERY batch item should differ. An
        # item the model halts on its learned signal used to be reset to the init
        # states on the next step, which threw its noise away and gave it the
        # same Q value in every rollout; with all four items halting (about one
        # model init in sixteen) "at least one item varies" failed too.
        spread = np.ptp(keras.ops.convert_to_numpy(all_q), axis=0)
        assert np.all(spread > 1e-3), (
            f"Expected variation in Q values with noise injection for every "
            f"item, got per-item spread {spread}"
        )

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

    def test_one_injection_moves_z_L_and_leaves_z_H_bit_identical(
        self, tiny_model, sample_batch
    ):
        """The paper perturbs the reasoning latent ``z`` only (D-018).

        Sghaier et al. 2026, arXiv:2605.19943, Figure 4 left, line 6:
        ``z_{t-1} += eps``. ``z`` is ``z_L`` in TRM's code and ``z_H`` is the
        answer state ``y``. This test used to assert that ``z_H`` moves too;
        the expectation changed with the semantics.
        """
        inference = PTRMInference(tiny_model, noise_scale=1.0)

        # A non-zero state, so "unchanged" is not "still zero".
        carry = tiny_model.initial_carry(sample_batch)
        rng = np.random.default_rng(0)
        carry["inner_carry"] = {
            name: keras.ops.convert_to_tensor(
                rng.normal(size=tuple(state.shape)).astype("float32")
            )
            for name, state in carry["inner_carry"].items()
        }
        z_H_before = keras.ops.convert_to_numpy(carry["inner_carry"]["z_H"])
        z_L_before = keras.ops.convert_to_numpy(carry["inner_carry"]["z_L"])

        noisy_carry = inference._inject_noise(carry)
        z_H_after = keras.ops.convert_to_numpy(noisy_carry["inner_carry"]["z_H"])
        z_L_after = keras.ops.convert_to_numpy(noisy_carry["inner_carry"]["z_L"])

        assert np.array_equal(z_H_after, z_H_before), "z_H must not be perturbed"
        diff_L = z_L_after - z_L_before
        assert np.std(diff_L) > 0.1, "z_L should change with noise"
        # Mean should be close to zero (zero-mean noise)
        assert abs(np.mean(diff_L)) < 0.1, "Noise should be zero-mean"

    def test_noise_scale_parameter(self, tiny_model, sample_batch):
        """Test that noise_scale parameter controls noise magnitude."""
        carry = tiny_model.initial_carry(sample_batch)
        
        # Small noise (measured on z_L, the latent the noise goes into)
        inference_small = PTRMInference(tiny_model, noise_scale=0.01)
        carry_small = inference_small._inject_noise(carry)
        diff_small = keras.ops.convert_to_numpy(
            carry_small["inner_carry"]["z_L"] - carry["inner_carry"]["z_L"]
        )
        
        # Large noise
        inference_large = PTRMInference(tiny_model, noise_scale=2.0)
        carry_large = inference_large._inject_noise(carry)
        diff_large = keras.ops.convert_to_numpy(
            carry_large["inner_carry"]["z_L"] - carry["inner_carry"]["z_L"]
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


class TestEveryRolloutStepKeepsItsNoise:
    """A rollout runs all D steps from the init states, as Algorithm 1 says.

    Both arms are deterministic in the defect: they do not depend on which
    items a randomly initialised Q head happens to halt.
    """

    @staticmethod
    def _spread(model, batch, supervision_steps):
        _, _, all_q = PTRMInference(model, noise_scale=1.0)(
            batch, num_rollouts=4, supervision_steps=supervision_steps
        )
        return np.ptp(keras.ops.convert_to_numpy(all_q), axis=0)

    def test_the_first_steps_noise_reaches_the_model(self, tiny_model, sample_batch):
        """The first step's noise used to land on the zero carry and be reset away."""
        spread = self._spread(tiny_model, sample_batch, supervision_steps=1)
        assert np.all(spread > 1e-3), spread

    def test_an_item_that_halts_on_the_learned_signal_keeps_its_noise(
        self, tiny_config, sample_batch
    ):
        """Every item halts after one step here, and every item must still vary."""
        model = PTRM(**tiny_config)
        carry = model.initial_carry(sample_batch)
        model(carry, sample_batch, training=False)
        model.inner.q_head.bias.assign(np.array([10.0, 0.0], dtype="float32"))
        carry, _ = model(model.initial_carry(sample_batch), sample_batch, training=False)
        assert bool(np.all(keras.ops.convert_to_numpy(carry["halted"]))), (
            "anti-vacuity: the q_halt bias did not make every item halt"
        )

        spread = self._spread(model, sample_batch, supervision_steps=3)
        assert np.all(spread > 1e-3), spread


class TestTheDefaultSelectionIsByQHalt:
    """Rollouts are selected by the halting logit ``q_halt`` unless asked (D-018).

    The paper selects ``k* = argmax_k q_hat_k`` with ``q_hat = f_Q(y_D)``
    (Sghaier et al. 2026, arXiv:2605.19943, Figure 4 left, lines 10 and 12),
    and ``q_hat`` is the halting logit TRM trains against answer correctness
    (Sec. 2). That is ``q_halt`` here. With ``no_act_continue=True`` (the PTRM
    default) ``q_continue`` is never trained, so a ``q_continue`` default read
    an untrained logit. The crafted Q head makes ``q_continue == -q_halt``, so
    the two selections disagree on every item.
    """

    K = 4
    D = 2

    @pytest.fixture(scope="class")
    def disagreeing_model(self, tiny_config):
        assert tiny_config["no_act_continue"] is True
        model = PTRM(**tiny_config)
        batch = {"inputs": keras.ops.zeros((2, tiny_config["seq_len"]), dtype="int32")}
        model(model.initial_carry(batch), batch, training=False)
        kernel = keras.ops.convert_to_numpy(model.inner.q_head.kernel)
        kernel[:, 1] = -kernel[:, 0]
        model.inner.q_head.kernel.assign(kernel)
        model.inner.q_head.bias.assign(np.zeros(2, dtype="float32"))
        return model

    def _last_steps(self, recorder):
        last = [recorder.step_outputs[(k + 1) * self.D - 1] for k in range(self.K)]
        return {
            key: np.stack([keras.ops.convert_to_numpy(o[key]) for o in last])
            for key in ("logits", "q_halt_logits", "q_continue_logits")
        }

    @pytest.mark.parametrize(
        "kwargs, selected_by",
        [({}, "q_halt_logits"), ({"use_q_continue": True}, "q_continue_logits")],
        ids=["default", "opt_in_q_continue"],
    )
    def test_the_class_selects_by_the_documented_logit(
        self, disagreeing_model, sample_batch, kwargs, selected_by
    ):
        recorder = _RecordingModel(disagreeing_model)
        best_logits, best_q, all_q = PTRMInference(
            recorder, noise_scale=1.0, **kwargs
        )(sample_batch, num_rollouts=self.K, supervision_steps=self.D)
        steps = self._last_steps(recorder)

        by_halt = np.argmax(steps["q_halt_logits"], axis=0)
        by_continue = np.argmax(steps["q_continue_logits"], axis=0)
        assert np.all(by_halt != by_continue), (
            "anti-vacuity: the crafted Q head must make the two selections "
            f"disagree on every item ({by_halt} vs {by_continue})"
        )

        chosen = np.argmax(steps[selected_by], axis=0)
        items = np.arange(chosen.shape[0])
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(all_q), steps[selected_by])
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(best_q), steps[selected_by][chosen, items]
        )
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(best_logits), steps["logits"][chosen, items]
        )

    def test_the_convenience_function_defaults_to_q_halt(
        self, disagreeing_model, sample_batch
    ):
        recorder = _RecordingModel(disagreeing_model)
        _, best_q = run_ptrm_inference(
            recorder, sample_batch, num_rollouts=self.K, supervision_steps=self.D,
            noise_scale=1.0,
        )
        steps = self._last_steps(recorder)
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(best_q), np.max(steps["q_halt_logits"], axis=0)
        )


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


class _RecordingModel:
    """Delegates to a PTRM and records the outputs of every step call."""

    def __init__(self, model):
        self._model = model
        self.step_outputs = []

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, carry, batch, training=False):
        carry, outputs = self._model(carry, batch, training=training)
        self.step_outputs.append(outputs)
        return carry, outputs


class TestPassAtKReportsBestQFromItsOwnRollouts:
    """``compute_pass_at_k`` scores the K rollouts it ran, and runs them once.

    It used to run a second, discarded K-rollout inference and never return the
    best-Q@k metric its comments promised.
    """

    K = 4
    D = 2
    # Eight items, so it is very unlikely that rollout 0 is the best-Q rollout
    # for every one of them (which would make pass@1 equal best-Q@k).
    INPUTS = np.random.default_rng(0).integers(0, 100, size=(8, 16)).astype("int32")

    @pytest.fixture
    def pinned_noise(self, monkeypatch):
        """Replace the noise draw with a seeded NumPy one.

        ``keras.utils.set_random_seed`` does not reproduce a later
        ``keras.random`` draw on this backend, so two runs only see the same
        rollouts if the draw itself is pinned.
        """

        def install(seed):
            rng = np.random.default_rng(seed)

            def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
                draw = rng.normal(mean, stddev, size=tuple(int(d) for d in shape))
                dtype = keras.backend.standardize_dtype(dtype or "float32")
                return keras.ops.convert_to_tensor(draw.astype(dtype))

            monkeypatch.setattr(keras.random, "normal", normal)

        return install

    def _run(self, model, batch, labels, use_q_continue):
        recorder = _RecordingModel(model)
        inference = PTRMInference(
            recorder, noise_scale=1.0, use_q_continue=use_q_continue
        )
        metrics = inference.compute_pass_at_k(
            batch, labels, num_rollouts=self.K, supervision_steps=self.D
        )
        return metrics, recorder

    def _rollouts(self, recorder, use_q_continue):
        """Each rollout's last step: predictions (K, B, S), the selection Q
        (K, B), and the other Q head (K, B)."""
        keys = ["q_halt_logits", "q_continue_logits"]
        if use_q_continue:
            keys.reverse()
        last = [recorder.step_outputs[(k + 1) * self.D - 1] for k in range(self.K)]
        predictions = np.stack(
            [np.argmax(keras.ops.convert_to_numpy(o["logits"]), axis=-1) for o in last]
        )
        selected, other = (
            np.stack([keras.ops.convert_to_numpy(o[key]) for o in last])
            for key in keys
        )
        return predictions, selected, other

    @staticmethod
    def _labels(predictions, selected, other):
        """Labels under which pass@1, pass@k and best-Q@k all differ.

        Each item's label is one rollout's prediction, or ``-1``, which no
        rollout predicts. A choice adds to five totals: rollout 0 right, any
        right, best-Q right, and two wrong selections right (the lowest Q, and
        the highest Q of the other head). The first combination wins whose
        first three totals all differ and whose best-Q total differs from both
        wrong selections, so selecting by either of them fails too. ``None`` if
        the rollouts allow none.
        """
        picks = [
            np.argmax(selected, axis=0),
            np.argmin(selected, axis=0),
            np.argmax(other, axis=0),
        ]
        options = []
        for item in range(predictions.shape[1]):
            by_effect = {(0,) * 5: np.full(predictions.shape[2], -1)}
            for k in range(predictions.shape[0]):
                right = np.all(predictions[:, item] == predictions[k, item], axis=-1)
                effect = (int(right[0]), 1) + tuple(
                    int(right[pick[item]]) for pick in picks
                )
                by_effect.setdefault(effect, predictions[k, item])
            options.append(list(by_effect.items()))
        for choice in itertools.product(*options):
            first, any_, top, lowest, other_top = np.sum(
                [effect for effect, _ in choice], axis=0
            )
            if len({first, any_, top}) == 3 and top not in (lowest, other_top):
                return np.stack([label for _, label in choice]).astype("int32")
        return None

    @pytest.mark.parametrize("use_q_continue", [True, False])
    def test_best_q_at_k_is_the_q_selected_rollouts_exact_match(
        self, tiny_model, pinned_noise, use_q_continue
    ):
        batch = {"inputs": keras.ops.convert_to_tensor(self.INPUTS, dtype="int32")}
        pinned_noise(7)
        _, first = self._run(tiny_model, batch, self.INPUTS, use_q_continue)
        first_predictions = self._rollouts(first, use_q_continue)[0]
        labels = self._labels(*self._rollouts(first, use_q_continue))
        assert labels is not None, (
            "anti-vacuity: no labels tell best-Q@k apart from pass@1, pass@k "
            "and the two wrong selections"
        )

        pinned_noise(7)
        metrics, recorder = self._run(tiny_model, batch, labels, use_q_continue)
        predictions, q_values, _ = self._rollouts(recorder, use_q_continue)
        assert np.array_equal(predictions, first_predictions), (
            "anti-vacuity: the pinned noise did not reproduce the rollouts"
        )

        exact = np.all(predictions == labels[None], axis=-1)  # (K, B)
        best = np.argmax(q_values, axis=0)
        expected = {
            "pass_at_1": float(np.mean(exact[0])),
            "pass_at_k": float(np.mean(np.any(exact, axis=0))),
            "best_q_at_k": float(np.mean(exact[best, np.arange(exact.shape[1])])),
        }
        assert metrics == expected

    def test_one_pass_runs_k_rollouts_of_d_steps(self, tiny_model, sample_batch):
        _, recorder = self._run(
            tiny_model, sample_batch, sample_batch["inputs"], use_q_continue=True
        )
        assert len(recorder.step_outputs) == self.K * self.D


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
        """PTRM with σ=0 should match D deep recursions from the init states.

        The reference drives the inner module directly, so it carries none of
        the outer ACT loop's halting: Algorithm 1 runs every rollout for all D
        steps. Driving ``tiny_model(carry, ...)`` instead would restart any item
        that halts on its learned signal, which is the behaviour PTRM must not
        have.
        """
        inner = tiny_model.inner
        batch_size = sample_batch["inputs"].shape[0]
        z = {
            "z_H": keras.ops.repeat(inner.H_init, batch_size, axis=0),
            "z_L": keras.ops.repeat(inner.L_init, batch_size, axis=0),
        }
        for _ in range(3):
            # The default selection reads q_halt (D-018), so compare against it.
            z, det_logits, (det_q, _) = inner(z, sample_batch, training=False)
        
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