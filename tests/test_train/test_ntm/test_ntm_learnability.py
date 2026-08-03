"""Opt-in convergence test: does the NTM actually LEARN the copy task?

Every other test in ``tests/test_train/test_ntm/`` and
``tests/test_layers/test_memory/`` checks that the NTM *runs* -- shapes, masks,
serialization, gradient flow. None of them would fail if the addressing
mechanism were wired backwards but still produced finite output of the right
shape. This module is the one that would: it trains a tiny NTM end to end and
asserts it reaches >90% validation bit accuracy on the copy task.

Why it is gated twice
---------------------
It costs ~50 s on a GPU, so it must not run in the default suite. The repo's
``pytest.mark.slow`` marker alone does NOT achieve that: there is no
``addopts = -m "not slow"`` in ``pyproject.toml`` (measured), so a marked test
is still collected and executed by a bare ``pytest``. The marker is therefore
paired with an explicit opt-in ``skipif`` on ``NTM_RUN_LEARNABILITY``, which is
the mechanism that actually keeps it out. Do not delete the ``skipif`` on the
grounds that the marker "already handles it" -- it does not, until an
``addopts`` line exists.

Run it with::

    NTM_RUN_LEARNABILITY=1 CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg \\
        python -m pytest tests/test_train/test_ntm/test_ntm_learnability.py -q

This test writes nothing to disk: it uses no ``create_callbacks`` and no
``ModelCheckpoint``, so it creates no ``results/`` artefacts to clean up.
"""

import os
import time
from typing import Dict, List, Optional

import keras
import numpy as np
import pytest

from dl_techniques.layers.memory import create_ntm
from dl_techniques.utils.logger import logger
from train.ntm.config import CopyTaskConfig
from train.ntm.data_generators import CopyTaskGenerator

# ---------------------------------------------------------------------
# Measured constants -- see the module docstring and D-006.
# ---------------------------------------------------------------------

#: Opt-in switch. Set to "1" to run this module.
OPT_IN_ENV_VAR = "NTM_RUN_LEARNABILITY"

#: Bit-accuracy threshold on the held-out split.
TARGET_BIT_ACCURACY = 0.90

# DECISION plan-2026-08-03T161943-02be1d7e/D-013
# Do NOT remove `keras.utils.set_random_seed(RANDOM_SEED)` "because a real test
# should not depend on a seed". Measured, not assumed: seed 7 never reached the
# threshold in 250 epochs (302 s) while five other seeds crossed in 20-76. An
# unseeded run is therefore a coin flip, and a guard that fires on luck teaches
# the next author to delete it. See decisions.md D-013.
#: Pinned seed. The run is seed-sensitive and this is not cosmetic: at
#: ``sequence_length=8`` seeds 42/99/0/1234 crossed the threshold at epochs
#: 20/22/22/26 and seed 2024 at 76, but seed 7 was still at 0.882 after 250
#: epochs (measured, GPU 1, 302 s). An unseeded version of this test would be
#: flaky in the "never converges" direction, so the seed is part of the
#: assertion, not decoration.
#: Seed 1234 reproduced epoch-for-epoch across two consecutive runs
#: (val_acc 0.8862 @25, 0.9378 @26 both times).
RANDOM_SEED = 1234

# DECISION plan-2026-08-03T161943-02be1d7e/D-016
# Do NOT lower MAX_EPOCHS back towards the pinned seed's 26 epochs to make the
# test snappier, and do NOT compensate for a slow run by lowering
# TARGET_BIT_ACCURACY. The threshold is the claim; the cap is only patience. The
# previous cap of 120 was derived from ONE seed's 26 epochs and looked generous
# at 4.6x -- but a plain 3-seed sweep found seed 2024 needing 76, i.e. 63% of
# that cap, outside this plan's own 60%-headroom rule. Any change here needs a
# NEW multi-seed measurement, not a single observation. See decisions.md D-016.
#: Epoch cap. Derived from the SPREAD, not from the pinned seed: a six-seed
#: sweep at this exact configuration crossed the threshold at epochs
#: **20 (seed 42), 22 (seed 99), 22 (seed 0), 26 (seed 1234), 76 (seed 2024)**,
#: with seed 7 not crossing within 250. The cap is therefore sized against the
#: worst observed CONVERGING run (76), not the pinned one (26): at 200 the
#: pinned seed consumes 13% and the slowest observed seed 38%, both comfortably
#: inside the "first run must not exceed 60% of the cap" rule this plan set for
#: convergence guards. The previous cap of 120 satisfied that rule only for the
#: pinned seed -- seed 2024 would have consumed 63% of it. At ~1.15 s per epoch
#: the worst case (never converging) costs ~230 s, which is the price of the
#: cap being patience rather than the claim; TARGET_BIT_ACCURACY is the claim.
MAX_EPOCHS = 200

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2


class _StopWhenLearned(keras.callbacks.Callback):
    """Halt training the first epoch validation bit accuracy clears a threshold.

    Keeps a converging run cheap (~50 s) while still allowing the full epoch
    budget for a slow one, and records *which* epoch crossed so the test can
    report the convergence curve rather than a bare pass.

    :param threshold: Validation bit accuracy that counts as learned.
    """

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold
        self.epoch_reached: Optional[int] = None
        self.history: List[float] = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        value = (logs or {}).get("val_acc")
        if value is None:
            return
        self.history.append(float(value))
        if value > self.threshold and self.epoch_reached is None:
            self.epoch_reached = epoch + 1
            self.model.stop_training = True


def build_copy_task_ntm(input_dim: int, output_dim: int, seq_len: int) -> keras.Model:
    """Build the tiny NTM used by this test.

    :param input_dim: Width of one input timestep.
    :param output_dim: Width of one target timestep.
    :param seq_len: Number of timesteps per sample.
    :return: A compiled-ready (not yet compiled) Keras model.
    """
    inputs = keras.Input(shape=(seq_len, input_dim), name="input_sequence")
    ntm = create_ntm(
        memory_size=16,
        memory_dim=8,
        output_dim=output_dim,
        controller_dim=32,
        return_sequences=True,
    )
    outputs = keras.layers.Activation("sigmoid", name="binary_output")(ntm(inputs))
    return keras.Model(inputs, outputs, name="tiny_copy_task_ntm")


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get(OPT_IN_ENV_VAR) != "1",
        reason=(
            f"Opt-in only: set {OPT_IN_ENV_VAR}=1 to run the ~50 s GPU "
            "convergence test. The `slow` marker alone does not deselect it -- "
            "this repo has no `addopts = -m \"not slow\"`."
        ),
    ),
]


def test_tiny_ntm_learns_the_copy_task():
    """A 8k-parameter NTM reaches >90% validation bit accuracy on copy.

    The assertion is on the *held-out* split, over the supervised (masked)
    positions only, so a model that echoes the input phase or emits a constant
    cannot pass: the copy task's output phase is all-zero in the input, and the
    per-bit target distribution is balanced, so both degenerate strategies sit
    at ~50%.
    """
    keras.utils.set_random_seed(RANDOM_SEED)

    config = CopyTaskConfig(
        sequence_length=8, vector_size=6, num_samples=4000, delay_length=1
    )
    data = CopyTaskGenerator(config).generate()

    model = build_copy_task_ntm(
        input_dim=data.inputs.shape[2],
        output_dim=data.targets.shape[2],
        seq_len=data.inputs.shape[1],
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="binary_crossentropy",
        # weighted, not plain: `masks` must gate the metric as well as the loss,
        # otherwise the unsupervised input-phase timesteps (target 0, trivially
        # predicted) inflate the accuracy this test asserts on.
        weighted_metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )

    stopper = _StopWhenLearned(TARGET_BIT_ACCURACY)
    start = time.time()
    model.fit(
        data.inputs,
        data.targets,
        sample_weight=data.masks,
        validation_split=VALIDATION_SPLIT,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        verbose=0,
        callbacks=[stopper],
    )
    elapsed = time.time() - start

    best = max(stopper.history) if stopper.history else float("nan")
    # Logged so a passing run still reports its convergence curve: the epoch cap
    # is the tolerance under test, and a pass that silently drifted from epoch 26
    # to epoch 110 is the signal that the cap needs re-deriving, not a green tick.
    logger.info(
        f"NTM copy-task learnability: best val bit accuracy {best:.4f} after "
        f"{len(stopper.history)} epoch(s) in {elapsed:.1f}s; crossed "
        f"{TARGET_BIT_ACCURACY:.0%} at epoch {stopper.epoch_reached} "
        f"of a {MAX_EPOCHS}-epoch cap"
    )
    assert stopper.epoch_reached is not None, (
        f"NTM did not reach {TARGET_BIT_ACCURACY:.0%} validation bit accuracy "
        f"within {MAX_EPOCHS} epochs ({elapsed:.0f}s). Best seen: {best:.4f} "
        f"after {len(stopper.history)} epochs. Measured reference at seed "
        f"{RANDOM_SEED}: crossed at epoch 26."
    )
    assert stopper.epoch_reached <= MAX_EPOCHS
    assert np.isfinite(best)
