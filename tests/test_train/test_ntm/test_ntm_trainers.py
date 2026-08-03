"""Tests for the NTM training entry points under ``src/train/ntm/``.

This module covers the trainers' pure helpers at smoke scale. It never calls
``model.fit`` and never builds a real NTM: every model is a stub whose
predictions are a deterministic function of its input, so the metrics
``evaluate_model`` reports can be hand-computed and asserted exactly rather
than merely checked for "did not raise".

The first group pins the defect this module was created for:
``evaluate_model`` indexed a flattened ``(seq_len, output_dim)`` prediction
with a per-TIMESTEP ``(seq_len,)`` mask, which is an ``IndexError`` for every
``output_dim > 1`` — that is, for every real copy-task run.

The later groups pin four further correctness fixes in the same trainers:
the ``dynamic_ngram`` final-timestep mask, the undiluted ``per_step_accuracy``
/ ``bit_error_rate`` in ``evaluate_copy_task``, the ``--patience`` flag
actually reaching ``create_callbacks``, and ``num_eval_samples`` actually
sizing the evaluation batch it is logged as sizing.
"""

import sys
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from train.ntm import train_multitask
from train.ntm.config import CopyTaskConfig
from train.ntm.data_generators import CopyTaskGenerator, TaskData
from train.ntm.metrics import evaluate_copy_task
from train.ntm.train_multitask import MultitaskNTMConfig, UnifiedTaskGenerator
from train.ntm.train_ntm import evaluate_model


# ---------------------------------------------------------------------
# Fixtures and stubs
# ---------------------------------------------------------------------


class CopyEchoStub:
    """A stand-in for a trained copy-task model.

    Reconstructs the copy-task target directly from the input encoding, so its
    predictions stay aligned with whatever row permutation ``evaluate_model``
    draws. Optional deterministic corruption makes the expected bit accuracy
    computable by hand.

    :param seq_len: Length of the copied sequence (not the total timeline).
    :param vec_size: Width of each copied vector, i.e. the output dimension.
    :param delay: Blank timesteps between the delimiter and the output phase.
    :param corrupt_bits: Number of leading output-phase bits to invert in
        every row. ``0`` yields a perfect model. Corruption is applied
        uniformly to all rows because ``evaluate_model`` draws a random row
        permutation, so no per-row expectation would be stable.
    """

    def __init__(
            self,
            seq_len: int,
            vec_size: int,
            delay: int,
            corrupt_bits: int = 0,
    ) -> None:
        self.seq_len = seq_len
        self.vec_size = vec_size
        self.delay = delay
        self.corrupt_bits = corrupt_bits

    @property
    def output_start(self) -> int:
        """First timestep of the output phase."""
        return self.seq_len + self.delay + 2

    def predict(self, inputs: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Emit the copied sequence in the output phase.

        :param inputs: Copy-task inputs, shape ``(batch, total_steps, features)``.
        :param verbose: Accepted for Keras API parity; ignored.
        :return: Probabilities of shape ``(batch, total_steps, vec_size)``.
        """
        batch, total_steps, _ = inputs.shape
        preds = np.zeros((batch, total_steps, self.vec_size), dtype=np.float32)

        sequence = inputs[:, 1:1 + self.seq_len, :self.vec_size]
        start = self.output_start
        preds[:, start:start + self.seq_len, :] = sequence

        if self.corrupt_bits:
            window = preds[:, start:start + self.seq_len, :].reshape(batch, -1)
            window[:, :self.corrupt_bits] = 1.0 - window[:, :self.corrupt_bits]
            preds[:, start:start + self.seq_len, :] = window.reshape(
                batch, self.seq_len, self.vec_size
            )

        return preds


def make_copy_data(
        num_samples: int = 4,
        seq_len: int = 4,
        vec_size: int = 4,
) -> Tuple[TaskData, CopyTaskConfig]:
    """Build a tiny real copy-task batch.

    With the defaults, the timeline is ``2*4 + delay(1) + 2 = 11`` steps and the
    output dimension is ``4`` — the exact 44-vs-11 shape that made the original
    ``IndexError`` fire.

    :param num_samples: Number of sequences.
    :param seq_len: Length of the copied sequence.
    :param vec_size: Width of each copied vector.
    :return: The generated ``TaskData`` and the config used to make it.
    """
    config = CopyTaskConfig(
        sequence_length=seq_len,
        vector_size=vec_size,
        num_samples=num_samples,
        random_seed=0,
    )
    return CopyTaskGenerator(config).generate(), config


# ---------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------


class TestEvaluateModel:
    """Tests for ``train.ntm.train_ntm.evaluate_model``."""

    def test_multi_dimensional_output_does_not_raise(self):
        """A per-timestep mask must index a per-element prediction correctly.

        Regression guard: with ``output_dim=4`` and 11 timesteps the flattened
        prediction has 44 elements while the mask has 11, which used to raise
        ``IndexError: boolean index did not match indexed array``.
        """
        data, config = make_copy_data()
        assert data.targets.shape[1:] == (11, 4)
        assert data.masks.shape[1:] == (11,)

        stub = CopyEchoStub(config.sequence_length, config.vector_size,
                            config.delay_length)
        results = evaluate_model(stub, data, num_eval=len(data.inputs))

        assert np.isfinite(results["bit_accuracy"])

    def test_perfect_model_scores_one(self):
        """An exact copy scores 1.0 on both metrics."""
        data, config = make_copy_data()
        stub = CopyEchoStub(config.sequence_length, config.vector_size,
                            config.delay_length)

        results = evaluate_model(stub, data, num_eval=len(data.inputs))

        assert results["bit_accuracy"] == pytest.approx(1.0)
        assert results["sequence_accuracy"] == pytest.approx(1.0)
        assert results["num_evaluated"] == len(data.inputs)

    def test_known_corruption_matches_hand_computed_values(self):
        """Two inverted bits per row give 14/16 bit accuracy and 0 sequence accuracy.

        The output phase carries ``seq_len * vec_size = 4 * 4 = 16`` supervised
        elements per row. Inverting the first two makes every row wrong as a
        whole (sequence accuracy 0.0) while 14 of 16 bits still match.
        """
        data, config = make_copy_data()
        stub = CopyEchoStub(config.sequence_length, config.vector_size,
                            config.delay_length, corrupt_bits=2)

        results = evaluate_model(stub, data, num_eval=len(data.inputs))

        assert results["bit_accuracy"] == pytest.approx(14.0 / 16.0)
        assert results["sequence_accuracy"] == pytest.approx(0.0)

    def test_metrics_cover_output_phase_only(self):
        """Errors outside the masked output phase are not counted.

        The stub emits zeros everywhere but the output phase; the targets are
        zero there too, so a metric that ignored the mask would still read
        1.0. Corrupting the whole output phase separates the two: masked-only
        reduction gives 0.0, an unmasked one would give ``(11-4)*4/44``.
        """
        data, config = make_copy_data()
        stub = CopyEchoStub(config.sequence_length, config.vector_size,
                            config.delay_length, corrupt_bits=16)

        results = evaluate_model(stub, data, num_eval=len(data.inputs))

        assert results["bit_accuracy"] == pytest.approx(0.0)

    def test_fully_masked_row_is_skipped(self):
        """A row with an all-zero mask leaves the denominator, it is not counted.

        One of three rows is fully masked out, so only two rows may reach the
        metrics. Counting the skipped row would both inflate the denominator to
        3 and poison the bit accuracy with ``nan``, since its selected-element
        array is empty.
        """
        data, config = make_copy_data(num_samples=3)
        masks = data.masks.copy()
        masks[0] = 0.0
        masked = TaskData(inputs=data.inputs, targets=data.targets,
                          masks=masks, metadata=data.metadata)

        stub = CopyEchoStub(config.sequence_length, config.vector_size,
                            config.delay_length)

        results = evaluate_model(stub, masked, num_eval=len(masked.inputs))

        assert results["num_evaluated"] == 2
        assert results["sequence_accuracy"] == pytest.approx(1.0)
        assert results["bit_accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------
# C1 — dynamic_ngram supervision mask
# ---------------------------------------------------------------------


class TestDynamicNGramMask:
    """The ``dynamic_ngram`` mask must cover exactly the DEFINED targets."""

    def test_mask_zeros_coincide_with_undefined_targets(self):
        """Warm-up context and the final timestep are both unsupervised.

        ``DynamicNGramGenerator.generate`` fills ``targets[i, t, :]`` only for
        ``t < sequence_length - 1``, so the final target row is all zero — there
        is no "next token" after the last one. The trainer's mask therefore has
        to drop that position as well as the two warm-up positions; otherwise
        the loss trains the model to emit "no token" there and the accuracy
        metric scores it.

        The assertion that makes this non-vacuous is the *coincidence* one: the
        all-zero (undefined) target region is exactly the final timestep, and
        the mask is 0 across the whole of it.
        """
        config = MultitaskNTMConfig(batch_size=3)
        generator = UnifiedTaskGenerator(config, mode='val')

        data = generator._generate_raw_data("dynamic_ngram")
        mask = data.masks

        assert np.all(mask[:, 0:2] == 0.0)
        assert np.all(mask[:, -1] == 0.0)

        # A target row is "undefined" iff the generator never set a one-hot in it.
        undefined = ~data.targets.any(axis=-1)
        assert np.all(undefined[:, -1]), "the final target row should be all zero"
        assert not undefined[:, :-1].any(), (
            "only the final timestep may have an undefined target"
        )
        assert np.all(mask[undefined] == 0.0), (
            "the mask must be 0 everywhere the target is undefined"
        )

    def test_interior_positions_stay_supervised(self):
        """The fix must not blank the positions the task actually learns."""
        config = MultitaskNTMConfig(batch_size=3)
        generator = UnifiedTaskGenerator(config, mode='val')

        mask = generator._generate_raw_data("dynamic_ngram").masks

        assert np.all(mask[:, 2:-1] == 1.0)


# ---------------------------------------------------------------------
# C2 — evaluate_copy_task metric dilution
# ---------------------------------------------------------------------


class FixedPredictionStub:
    """A model whose ``predict`` returns a preset array.

    :param predictions: The array to return, shape ``(batch, steps, output_dim)``.
    """

    def __init__(self, predictions: np.ndarray) -> None:
        self._predictions = predictions

    def predict(self, inputs: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Return the preset predictions, ignoring the inputs.

        :param inputs: Ignored; accepted for Keras API parity.
        :param verbose: Ignored; accepted for Keras API parity.
        :return: The preset prediction array.
        """
        return self._predictions


def _diluted_case() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the hand-computed 2-row case used by the C2 guards.

    Timeline is 6 steps with a 2-step supervised tail (``mask =
    [0,0,0,0,1,1]``) and ``output_dim = 2``, so 8 of the 24 elements are real
    and 16 are padding — the same roughly-half split a real ``CopyTaskConfig``
    produces.

    Row 0 gets 2 of its 4 supervised bits wrong; row 1 is exact in the
    supervised region but deliberately emits high-confidence garbage in the
    padded region.

    Hand-computed:

    * masked-only (correct): 6 of 8 supervised elements agree →
      ``per_step_accuracy = 0.75``, ``bit_error_rate = 0.25``.
    * full-tensor (diluted): the 16 padded elements agree trivially (0 == 0),
      so 22 of 24 agree → ``per_step_accuracy = 0.9166667``,
      ``bit_error_rate = 0.0833333``.
    * ``sequence_accuracy = 0.5`` either way — row 1 matches exactly once the
      padding is zeroed on both sides, row 0 does not.

    :return: ``(predictions, targets, masks)``.
    """
    masks = np.array([[0., 0., 0., 0., 1., 1.]] * 2, dtype=np.float32)

    targets = np.zeros((2, 6, 2), dtype=np.float32)
    targets[:, 0:4, :] = 1.0        # padded region: nonzero, must not be scored
    targets[0, 4] = [1.0, 1.0]
    targets[0, 5] = [1.0, 0.0]
    targets[1, 4] = [1.0, 1.0]
    targets[1, 5] = [1.0, 0.0]

    predictions = np.zeros((2, 6, 2), dtype=np.float32)
    predictions[1, 0:4, :] = 0.99   # padded garbage, must not be scored
    predictions[0, 4] = [0.9, 0.1]  # -> [1, 0] vs [1, 1]: 1 wrong
    predictions[0, 5] = [0.2, 0.2]  # -> [0, 0] vs [1, 0]: 1 wrong
    predictions[1, 4] = [0.9, 0.8]  # -> [1, 1] vs [1, 1]: exact
    predictions[1, 5] = [0.7, 0.1]  # -> [1, 0] vs [1, 0]: exact

    return predictions, targets, masks


class TestEvaluateCopyTaskMasking:
    """``evaluate_copy_task`` must score only the supervised elements."""

    def test_per_step_and_ber_are_not_diluted_by_padding(self):
        """6 of 8 supervised elements agree: 0.75, not the diluted 0.9166667.

        See ``_diluted_case`` for the full hand computation. The diluted answer
        (22/24 = 0.9166667, BER 0.0833333) is what a full-tensor reduction
        returns, because the 16 padded elements are zeroed on both sides and
        therefore always agree.
        """
        predictions, targets, masks = _diluted_case()
        model = FixedPredictionStub(predictions)

        results = evaluate_copy_task(model, np.zeros((2, 6, 3)), targets, masks)

        assert results.metrics["per_step_accuracy"].value == pytest.approx(0.75)
        assert results.metrics["bit_error_rate"].value == pytest.approx(0.25)
        # The diluted values must NOT be what we get.
        assert results.metrics["per_step_accuracy"].value != pytest.approx(22.0 / 24.0)

    def test_sequence_accuracy_is_unchanged(self):
        """``sequence_accuracy`` keeps its existing zero-both-sides semantics.

        It is correct *because* of the masking-by-zeroing: row 1's padded
        garbage is zeroed away, so the row counts as an exact match on the
        supervised region alone. One of two rows matches → 0.5.
        """
        predictions, targets, masks = _diluted_case()
        model = FixedPredictionStub(predictions)

        results = evaluate_copy_task(model, np.zeros((2, 6, 3)), targets, masks)

        assert results.metrics["sequence_accuracy"].value == pytest.approx(0.5)
        assert results.error_rate == pytest.approx(0.5)

    def test_unmasked_call_reduces_over_everything(self):
        """With ``masks=None`` every element is supervised.

        Guards against a fix that silently drops elements when no mask is
        given. Nothing is zeroed here, so the padded region is scored on its
        raw values: row 0 predicts 0 against a padded target of 1 in all 8 of
        its padded elements, row 1's 0.99 binarizes to 1 and agrees in all 8 of
        its own. Adding row 0's two supervised errors gives 10 disagreements,
        so 14 of 24 agree.
        """
        predictions, targets, _ = _diluted_case()
        model = FixedPredictionStub(predictions)

        results = evaluate_copy_task(model, np.zeros((2, 6, 3)), targets, masks=None)

        assert results.metrics["per_step_accuracy"].value == pytest.approx(14.0 / 24.0)
        assert results.metrics["bit_error_rate"].value == pytest.approx(10.0 / 24.0)

    def test_fully_masked_input_yields_nan_without_warning(self):
        """An all-zero mask selects nothing: nan, not a RuntimeWarning."""
        predictions, targets, masks = _diluted_case()
        model = FixedPredictionStub(predictions)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            results = evaluate_copy_task(
                model, np.zeros((2, 6, 3)), targets, np.zeros_like(masks)
            )

        assert np.isnan(results.metrics["per_step_accuracy"].value)
        assert np.isnan(results.metrics["bit_error_rate"].value)


# ---------------------------------------------------------------------
# C3 — --patience must reach create_callbacks
# ---------------------------------------------------------------------


class _StubKerasModel:
    """Minimal stand-in for the compiled multi-task NTM."""

    def fit(self, *args: Any, **kwargs: Any) -> None:
        """Pretend to train; return no history.

        :return: ``None`` — the trainer only passes it through.
        """
        return None

    def save(self, path: str) -> None:
        """Pretend to save.

        :param path: Ignored.
        """
        return None


class TestPatienceWiring:
    """``--patience`` must change the value ``create_callbacks`` receives."""

    def test_patience_flag_reaches_create_callbacks(self, monkeypatch, tmp_path):
        """A non-default ``--patience 7`` must arrive at ``create_callbacks``.

        This drives the WHOLE chain — ``sys.argv`` → argparse →
        ``MultitaskNTMConfig`` → ``train_multitask_ntm`` → ``create_callbacks``
        — because asserting that the flag merely parses would still pass while
        the call site hardcodes a literal.
        """
        recorded: List[Dict[str, Any]] = []

        def spy_create_callbacks(**kwargs: Any):
            recorded.append(kwargs)
            return [], str(tmp_path)

        monkeypatch.setattr(train_multitask, "setup_gpu", lambda gpu: None)
        monkeypatch.setattr(train_multitask, "create_generators",
                            lambda config: (None, None))
        monkeypatch.setattr(train_multitask, "create_multitask_ntm_model",
                            lambda config: _StubKerasModel())
        monkeypatch.setattr(train_multitask, "compile_model",
                            lambda model, config: None)
        monkeypatch.setattr(train_multitask, "create_callbacks", spy_create_callbacks)
        monkeypatch.setattr(train_multitask, "evaluate_tasks",
                            lambda model, config: None)
        monkeypatch.setattr(sys, "argv", ["train_multitask", "--patience", "7"])

        train_multitask.main()

        assert len(recorded) == 1, "create_callbacks was never reached"
        assert recorded[0]["patience"] == 7

    def test_default_patience_comes_from_the_base_parser(self):
        """The config default must match the base parser's ``--patience`` default."""
        assert MultitaskNTMConfig().patience == 50


# ---------------------------------------------------------------------
# C4 — num_eval_samples must size the evaluation batch
# ---------------------------------------------------------------------


class _ShapeRecordingModel:
    """Records the batch dimension of every ``predict`` call.

    :param output_shape: Trailing ``(steps, features)`` shape to emit.
    """

    def __init__(self, output_shape: Tuple[int, int]) -> None:
        self.output_shape = output_shape
        self.seen_batches: List[int] = []

    def predict(self, inputs: Dict[str, np.ndarray], verbose: int = 0) -> np.ndarray:
        """Return zeros shaped like the model's output.

        :param inputs: Dict with ``sequence_in`` and ``task_id_in``.
        :param verbose: Ignored; accepted for Keras API parity.
        :return: Zeros of shape ``(batch,) + output_shape``.
        """
        batch = inputs["sequence_in"].shape[0]
        self.seen_batches.append(batch)
        assert inputs["task_id_in"].shape[0] == batch
        return np.zeros((batch,) + self.output_shape, dtype=np.float32)


class _LogRecorder:
    """Collects the messages the trainer logs."""

    def __init__(self) -> None:
        self.messages: List[str] = []

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Record an info message."""
        self.messages.append(message)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Record a warning message."""
        self.messages.append(message)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Record an error message."""
        self.messages.append(message)


class TestEvaluationSampleCount:
    """``evaluate_tasks`` must evaluate the number of samples it claims to."""

    def test_num_eval_samples_sizes_the_eval_batch(self, monkeypatch):
        """The logged count and the evaluated count must be the same number.

        ``num_eval_samples`` is deliberately set to a value that differs from
        ``batch_size``, so a run that still sized the evaluation from
        ``batch_size`` would show 2 where the log claims 5.
        """
        recorder = _LogRecorder()
        monkeypatch.setattr(train_multitask, "logger", recorder)

        config = MultitaskNTMConfig(batch_size=2, num_eval_samples=5)
        model = _ShapeRecordingModel(
            (config.max_seq_length, config.max_vector_size)
        )

        train_multitask.evaluate_tasks(model, config)

        assert model.seen_batches == [5] * config.num_tasks
        assert any("Evaluating on 5 samples" in message
                   for message in recorder.messages)
