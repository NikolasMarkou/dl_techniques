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

The final three groups are structural rather than defect-specific: the argparse
contract all three entry points must honour, the padding/truncation contract of
``_pad_and_normalize``, and the phase layout of ``CopyTaskGenerator`` — the one
generator the audit certified as correct, pinned so it cannot drift silently.
"""

import sys
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from train.ntm import run_benchmark_suite, train_multitask, train_ntm
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

    def test_default_patience_agrees_between_parser_and_config(self, monkeypatch):
        """The ``--patience`` default is 50 in BOTH the parser and the config.

        Renamed from ``test_default_patience_comes_from_the_base_parser``: the
        base parser is gone (``train_multitask`` is now Pattern 2, a local
        ``argparse``), so that name asserted a source of truth that no longer
        exists. The fact worth pinning survived the move — the two defaults must
        agree, or a bare ``python -m train.ntm.train_multitask`` and a direct
        ``MultitaskNTMConfig()`` would train with different patience.

        Asserting only ``MultitaskNTMConfig().patience == 50`` would be
        half-blind: it would still pass if the local parser were written with the
        shared parser's old default of 15.
        """
        monkeypatch.setattr(sys, "argv", ["train_multitask"])
        args = train_multitask.parse_arguments()

        assert args.patience == 50
        assert MultitaskNTMConfig().patience == 50


# ---------------------------------------------------------------------
# Pattern 2 — every flag must reach MultitaskNTMConfig
# ---------------------------------------------------------------------


def _config_from_argv(monkeypatch, tmp_path, argv: List[str]) -> MultitaskNTMConfig:
    """Drive ``main()`` end-to-end and return the config it actually built.

    The capture point is ``create_multitask_ntm_model``, the first consumer of
    the config inside ``train_multitask_ntm``. Capturing at the parser instead
    would prove only that argparse works, which is never the defect: the
    documented bug class in this repo is a flag that parses fine and is then
    never forwarded into the config.

    :param monkeypatch: pytest monkeypatch fixture.
    :param tmp_path: pytest tmp_path fixture, used as the fake results dir.
    :param argv: Command-line arguments AFTER the program name.
    :return: The ``MultitaskNTMConfig`` the trainer constructed.
    """
    captured: List[MultitaskNTMConfig] = []

    def spy_create_model(config: MultitaskNTMConfig):
        captured.append(config)
        return _StubKerasModel()

    monkeypatch.setattr(train_multitask, "setup_gpu", lambda gpu: None)
    monkeypatch.setattr(train_multitask, "create_generators",
                        lambda config: (None, None))
    monkeypatch.setattr(train_multitask, "create_multitask_ntm_model",
                        spy_create_model)
    monkeypatch.setattr(train_multitask, "compile_model",
                        lambda model, config: None)
    monkeypatch.setattr(train_multitask, "create_callbacks",
                        lambda **kwargs: ([], str(tmp_path)))
    monkeypatch.setattr(train_multitask, "evaluate_tasks",
                        lambda model, config: None)
    monkeypatch.setattr(sys, "argv", ["train_multitask"] + argv)

    train_multitask.main()

    assert len(captured) == 1, "create_multitask_ntm_model was never reached"
    return captured[0]


class TestPatternTwoArgumentSurface:
    """The local parser must expose exactly the trainer's real config space.

    Six ``MultitaskNTMConfig`` fields previously had no flag at all, and five
    flags inherited from ``create_base_argument_parser`` were parsed and
    discarded. Both halves are pinned here — the additions one assertion at a
    time (a single mutation of the shared ``MultitaskNTMConfig(...)`` call would
    otherwise "prove" all six at once), and the removals as a set.
    """

    # --- the six added flags, one assertion + one mutation each -------------

    def test_controller_type_flag_reaches_the_config(self, monkeypatch, tmp_path):
        """``--controller-type gru`` must arrive as ``config.controller_type``."""
        config = _config_from_argv(
            monkeypatch, tmp_path, ["--controller-type", "gru"])
        assert config.controller_type == "gru"

    def test_num_read_heads_flag_reaches_the_config(self, monkeypatch, tmp_path):
        """``--num-read-heads 3`` must arrive as ``config.num_read_heads``."""
        config = _config_from_argv(
            monkeypatch, tmp_path, ["--num-read-heads", "3"])
        assert config.num_read_heads == 3

    def test_num_write_heads_flag_reaches_the_config(self, monkeypatch, tmp_path):
        """``--num-write-heads 2`` must arrive as ``config.num_write_heads``."""
        config = _config_from_argv(
            monkeypatch, tmp_path, ["--num-write-heads", "2"])
        assert config.num_write_heads == 2

    def test_shift_range_flag_reaches_the_config(self, monkeypatch, tmp_path):
        """``--shift-range 5`` must arrive as ``config.shift_range``.

        5 is used rather than 4 because ``NTMConfig`` rejects even shift ranges,
        so an even probe value would be untrainable and could not be smoke-run.
        """
        config = _config_from_argv(monkeypatch, tmp_path, ["--shift-range", "5"])
        assert config.shift_range == 5

    def test_max_seq_length_flag_reaches_the_config(self, monkeypatch, tmp_path):
        """``--max-seq-length 40`` must arrive as ``config.max_seq_length``."""
        config = _config_from_argv(
            monkeypatch, tmp_path, ["--max-seq-length", "40"])
        assert config.max_seq_length == 40

    def test_max_vector_size_flag_reaches_the_config(self, monkeypatch, tmp_path):
        """``--max-vector-size 12`` must arrive as ``config.max_vector_size``."""
        config = _config_from_argv(
            monkeypatch, tmp_path, ["--max-vector-size", "12"])
        assert config.max_vector_size == 12

    # --- controls: every probe value differs from the dataclass default ------

    def test_probe_values_differ_from_the_dataclass_defaults(self):
        """Non-vacuity control for the six assertions above.

        Each test passes a value the dataclass would NOT produce on its own, so
        dropping that field from the ``MultitaskNTMConfig(...)`` call makes the
        matching assertion fail. If a probe ever equalled the default, its guard
        would pass whether or not the flag was wired — this control is what
        stops that from happening silently.
        """
        default = MultitaskNTMConfig()
        assert default.controller_type != "gru"
        assert default.num_read_heads != 3
        assert default.num_write_heads != 2
        assert default.shift_range != 5
        assert default.max_seq_length != 40
        assert default.max_vector_size != 12

    # --- the five dropped flags ---------------------------------------------

    @pytest.mark.parametrize(
        "flag",
        ["--dataset", "--image-size", "--weight-decay", "--lr-schedule",
         "--show-plots"],
    )
    def test_inapplicable_base_parser_flags_are_gone(self, flag, monkeypatch):
        """A flag this trainer never reads must be REFUSED, not silently ignored.

        These five came from ``create_base_argument_parser``, which has no
        opt-out. They parsed cleanly and were then discarded — a user setting
        ``--weight-decay 0.05`` got no weight decay and no warning. Exiting
        non-zero on an unknown flag is the only outcome that tells them.
        """
        monkeypatch.setattr(sys, "argv", ["train_multitask", flag, "1"])

        with pytest.raises(SystemExit) as excinfo:
            train_multitask.parse_arguments()

        assert excinfo.value.code != 0

    def test_the_kept_flags_are_all_still_parseable(self, monkeypatch):
        """The twelve retained flags must survive the Pattern 2 conversion.

        Companion to the test above: a parser that refused everything would pass
        the dropped-flag parametrisation for the wrong reason.
        """
        monkeypatch.setattr(sys, "argv", [
            "train_multitask",
            "--epochs", "1", "--batch-size", "4", "--learning-rate", "1e-3",
            "--patience", "2", "--memory-size", "8", "--memory-dim", "4",
            "--controller-dim", "16", "--steps-per-epoch", "2",
            "--validation-steps", "1", "--clip-norm", "0.5",
            "--num-eval-samples", "8", "--gpu", "1",
        ])

        args = train_multitask.parse_arguments()

        assert (args.epochs, args.batch_size, args.patience) == (1, 4, 2)
        assert args.learning_rate == pytest.approx(1e-3)
        assert (args.memory_size, args.memory_dim, args.controller_dim) == (8, 4, 16)
        assert (args.steps_per_epoch, args.validation_steps) == (2, 1)
        assert args.clip_norm == pytest.approx(0.5)
        assert (args.num_eval_samples, args.gpu) == (8, 1)


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


# ---------------------------------------------------------------------
# E1 — the argparse contract of both entry points
# ---------------------------------------------------------------------


class TestCommandLineContract:
    """Every runnable script here must parse arguments before doing anything."""

    @pytest.mark.parametrize(
        "module",
        [train_ntm, train_multitask, run_benchmark_suite],
        ids=["train_ntm", "train_multitask", "run_benchmark_suite"],
    )
    def test_help_exits_zero_without_starting_training(
            self, module, monkeypatch, capsys):
        """``--help`` must print usage and exit 0, having started nothing.

        This is the regression guard for the defect that motivated the
        multi-task consolidation: the deleted trainer had NO argparse at all,
        so ``python -m train.ntm.train_multitask --help`` silently launched a
        100-epoch run instead of printing help.

        Why this guard is not vacuous: ``setup_gpu`` is the first side effect
        after ``parse_args()`` in every ``main()`` body here, and it is replaced
        here by a tripwire that raises. A module that lost its argparse would
        never raise ``SystemExit`` — it would either fall through to the
        tripwire (``AssertionError``) or die on the now-undefined ``args``
        (``NameError``). Both make ``pytest.raises(SystemExit)`` fail, and
        neither is reachable while the parser exits during ``parse_args``.
        The tripwire is therefore also the proof that no training started.
        """

        def tripwire(_gpu):
            raise AssertionError(
                f"{module.__name__}.main() reached setup_gpu on --help: "
                "argument parsing did not short-circuit"
            )

        monkeypatch.setattr(module, "setup_gpu", tripwire)
        monkeypatch.setattr(sys, "argv", [module.__name__, "--help"])

        with pytest.raises(SystemExit) as excinfo:
            module.main()

        assert excinfo.value.code == 0
        assert "usage:" in capsys.readouterr().out


# ---------------------------------------------------------------------
# E1 — _pad_and_normalize shape/mask invariants
# ---------------------------------------------------------------------


def _padding_generator(max_seq: int = 8, max_vec: int = 6) -> UnifiedTaskGenerator:
    """Build a generator whose padding frame is small enough to reason about.

    :param max_seq: Timeline width every task is normalized to.
    :param max_vec: Feature width every task is normalized to.
    :return: A validation-mode generator with that frame.
    """
    config = MultitaskNTMConfig(
        batch_size=2, max_seq_length=max_seq, max_vector_size=max_vec
    )
    return UnifiedTaskGenerator(config, mode='val')


def _ramp(batch: int, steps: int, dim: int, offset: float = 0.0) -> np.ndarray:
    """Build a distinct, non-binary value at every position.

    Non-binary values matter: they would expose a normalization or rescaling
    step, which a 0/1 payload could hide.

    :param batch: Number of rows.
    :param steps: Number of timesteps.
    :param dim: Feature width.
    :param offset: Added to every element, to separate inputs from targets.
    :return: Array of shape ``(batch, steps, dim)``.
    """
    size = batch * steps * dim
    return (np.arange(size, dtype=np.float32).reshape(batch, steps, dim)
            * 0.125 + 1.0 + offset)


class TestPadAndNormalize:
    """``_pad_and_normalize`` must pad with zeros and preserve real content."""

    @pytest.mark.parametrize("curr_seq", [5, 8], ids=["shorter", "exactly_max"])
    def test_padding_preserves_content_and_zeroes_the_rest(self, curr_seq):
        """Real content survives byte-exact; every padded cell is zero.

        Runs at a sequence shorter than the frame and at exactly the frame
        width, where no padding is needed at all — an off-by-one that dropped
        the final timestep would pass the ``shorter`` case and fail this one.

        The values are non-binary and all distinct, so a rescale or a
        transpose could not reproduce them by accident.
        """
        generator = _padding_generator()
        max_seq = generator.config.max_seq_length
        max_vec = generator.config.max_vector_size
        batch, in_dim = 2, 4

        inputs = _ramp(batch, curr_seq, in_dim)
        targets = _ramp(batch, curr_seq, in_dim, offset=100.0)
        masks = np.zeros((batch, curr_seq), dtype=np.float32)
        masks[:, -2:] = 1.0

        data = TaskData(inputs=inputs, targets=targets, masks=masks)
        out_in, out_tg, out_mask = generator._pad_and_normalize(data, "copy")

        assert out_in.shape == (batch, max_seq, max_vec)
        assert out_tg.shape == (batch, max_seq, max_vec)
        assert out_mask.shape == (batch, max_seq)

        # Content preserved, unscaled and untruncated.
        np.testing.assert_array_equal(out_in[:, :curr_seq, :in_dim], inputs)
        np.testing.assert_array_equal(out_tg[:, :curr_seq, :in_dim], targets)
        np.testing.assert_array_equal(out_mask[:, :curr_seq], masks)

        # Padding is zero in inputs, targets AND mask.
        assert np.all(out_in[:, curr_seq:, :] == 0.0)
        assert np.all(out_in[:, :, in_dim:] == 0.0)
        assert np.all(out_tg[:, curr_seq:, :] == 0.0)
        assert np.all(out_tg[:, :, in_dim:] == 0.0)
        assert np.all(out_mask[:, curr_seq:] == 0.0)

    def test_oversized_input_is_truncated_not_wrapped(self):
        """Content past the frame is dropped, never folded back into it.

        The function clamps with ``min(curr_seq, max_seq)`` on both axes, so
        an over-long sequence loses its tail. The guard that matters is the
        negative one: the out-of-frame payload carries a sentinel value that
        must appear nowhere in the output, which a wrap-around or a reshape
        would smuggle back in. The mask is truncated the same way, so no
        dropped position stays supervised.
        """
        generator = _padding_generator()
        max_seq = generator.config.max_seq_length
        max_vec = generator.config.max_vector_size
        batch, curr_seq, in_dim = 2, 12, 9
        sentinel = -7.5

        inputs = _ramp(batch, curr_seq, in_dim)
        inputs[:, max_seq:, :] = sentinel
        inputs[:, :, max_vec:] = sentinel
        targets = inputs.copy()
        masks = np.ones((batch, curr_seq), dtype=np.float32)

        data = TaskData(inputs=inputs, targets=targets, masks=masks)
        out_in, out_tg, out_mask = generator._pad_and_normalize(data, "copy")

        assert out_in.shape == (batch, max_seq, max_vec)
        np.testing.assert_array_equal(out_in, inputs[:, :max_seq, :max_vec])
        np.testing.assert_array_equal(out_tg, targets[:, :max_seq, :max_vec])
        assert not np.any(out_in == sentinel), "truncated content re-entered the frame"
        assert not np.any(out_tg == sentinel), "truncated content re-entered the frame"
        assert np.all(out_mask == 1.0)


# ---------------------------------------------------------------------
# E1 — CopyTaskGenerator phase layout
# ---------------------------------------------------------------------


class TestCopyTaskGeneratorLayout:
    """``CopyTaskGenerator``'s mask and targets must match its stated layout."""

    def test_mask_marks_exactly_the_output_phase(self):
        """Mask, targets and the documented ``output_start`` all agree.

        The generator's own comment fixes the layout as ``[start marker,
        sequence, delimiter, delay, output]`` with the output phase beginning
        at ``seq_len + delay + 2``. This pins all three facts that layout
        implies, against a batch large enough that a stray off-by-one cannot
        hide:

        1. the mask is 1 on exactly ``[output_start, output_start + seq_len)``
           and 0 everywhere else;
        2. the targets in that window are the sequence the inputs presented in
           ``[1, 1 + seq_len)``, so mask and supervision coincide;
        3. the targets are all-zero outside it, which is what makes
           ``evaluate_copy_task``'s zero-both-sides ``sequence_accuracy``
           sound.
        """
        seq_len, vec_size, num_samples = 6, 5, 8
        config = CopyTaskConfig(
            sequence_length=seq_len,
            vector_size=vec_size,
            num_samples=num_samples,
            random_seed=1,
        )
        data = CopyTaskGenerator(config).generate()

        delay = config.delay_length
        output_start = seq_len + delay + 2
        total_steps = seq_len * 2 + delay + 2
        assert data.inputs.shape == (num_samples, total_steps, vec_size + 2)
        assert data.targets.shape == (num_samples, total_steps, vec_size)

        expected_mask = np.zeros((num_samples, total_steps), dtype=np.float32)
        expected_mask[:, output_start:output_start + seq_len] = 1.0
        np.testing.assert_array_equal(data.masks, expected_mask)

        presented = data.inputs[:, 1:1 + seq_len, :vec_size]
        supervised = data.targets[:, output_start:output_start + seq_len, :]
        np.testing.assert_array_equal(supervised, presented)

        outside = data.targets.copy()
        outside[:, output_start:output_start + seq_len, :] = 0.0
        assert np.all(outside == 0.0), "targets exist outside the masked window"

        # The sequence is not degenerate, so the equality above is informative.
        assert presented.min() == 0.0 and presented.max() == 1.0
