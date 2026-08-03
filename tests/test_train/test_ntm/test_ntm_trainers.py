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
"""

from typing import Tuple

import numpy as np
import pytest

from train.ntm.config import CopyTaskConfig
from train.ntm.data_generators import CopyTaskGenerator, TaskData
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
