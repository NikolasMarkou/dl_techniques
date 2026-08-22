import os
import sys
from pathlib import Path

import pytest

# Add src to Python path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )


# --- Golden-value reference device -----------------------------------------
#
# The single source of truth for the device a stored-reference ("golden value")
# probe must be built and run on. Centralized here because the *policy* and its
# justification must not be restated per module: there are already two golden
# modules (``test_models/test_scunet/test_golden_values.py`` and
# ``test_models/test_swin_transformer/test_golden_values.py``) with two pinned
# sites each, and a per-file copy would be four places to drift.
#
# WHY A PIN IS REQUIRED, measured (not assumed) on GPU 1 (RTX 4070) at the Swin
# golden config, GPU output vs the CPU reference:
#
#   | regime                    | max|diff| vs CPU |
#   |---------------------------|------------------|
#   | float32, TF32 ON (default)| 2.254173e-05     |
#   | float32, TF32 OFF         | 1.490116e-08     |
#   | float64                   | 2.081668e-17     |
#
# So the deviation is ordinary reduced-precision matmul reassociation -- TF32
# accounts for ~99.93% of it -- and NOT a device-dependent difference in the
# Swin code path. It is still fatal to these guards: the signal they exist to
# catch is 1.04e-07 (the D-006 single-window-rule neuter), so their tolerance is
# 1e-8, and even the TF32-OFF residual exceeds it. Widening the tolerance to
# accommodate a GPU would make the guards blind to exactly what they watch.
#
# WHAT NOT TO DO: do not "fix" a golden-value failure on GPU by relaxing atol,
# and do not skip these modules when a GPU is visible -- a skipped guard on the
# machine the models are developed on is not a guard. Pin the probe instead.
# The cost is recorded honestly: these guards pin CPU numerics only.
GOLDEN_REFERENCE_DEVICE = "cpu"


@pytest.fixture(scope="session")
def golden_reference_device() -> str:
    """Device string that golden-value probes must build and run inside.

    Use as ``with keras.device(golden_reference_device): ...`` at BOTH the
    reference-producing site and any site whose output is compared against it --
    pinning only one of the two makes the comparison cross-device, which is the
    failure this fixture exists to prevent.

    :return: a ``keras.device``-compatible device string (never ``None``).
    """
    return GOLDEN_REFERENCE_DEVICE


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# --- D-1: no test may write into the repo-root results/ directory ----------
#
# The defect CLASS, not one instance. Trainer entry points resolve a relative
# `output_dir` (typically the literal default `"results"`) against the REPO
# ROOT, so any test that calls a real trainer -- or a real `ModelCheckpoint`,
# or a real `model.save()` through one -- deposits a run directory into the
# user's `results/` tree. That tree is gitignored AND untracked, so anything
# that lands there is indistinguishable from a real experiment, and anything
# deleted from there is UNRECOVERABLE (62 run directories, including a
# published paper's subject checkpoint, were destroyed exactly once this way).
#
# `tests/test_train/test_sam3/test_train_sam3.py` carried this snapshot inline
# in a single test body. It is promoted here so the next test anywhere in the
# suite that reaches a trainer entry point is watched by default rather than by
# remembering.
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"


def _results_entries() -> frozenset:
    """Names directly under the repo-root ``results/``; empty if absent.

    One ``os.scandir`` on ONE directory, no recursion and no ``stat``, because
    this runs twice per test for the whole suite.

    :return: entry names, or an empty frozenset when ``results/`` does not
        exist (a fresh clone) or is unreadable.
    """
    try:
        with os.scandir(RESULTS_ROOT) as it:
            return frozenset(entry.name for entry in it)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return frozenset()


@pytest.fixture(autouse=True)
def no_repo_root_results_writes():
    """Fail any test that adds an entry to the repo-root ``results/``.

    ASSERT-ONLY, AND THIS IS NOT NEGOTIABLE. This fixture MUST NEVER delete,
    move, truncate, rename or otherwise "clean up" anything under ``results/``
    -- not even an entry it can prove the test under it just created, and not
    in a ``finally``. ``results/`` is gitignored and untracked: there is no
    history and no backup, so a deletion here is permanent, and a cleanup step
    that names that directory is precisely how 62 run directories (including a
    published paper's subject checkpoint) were destroyed on 2026-08-12. When
    this assertion fires, the repair is to route the test's config through
    ``tmp_path``; the reported directory is then removed BY A HUMAN, if at all.

    Failing rather than skipping is deliberate: a run directory in ``results/``
    is not a cosmetic mess, it is an artefact a later human reads as an
    experiment.

    :raises AssertionError: if new entries appear directly under ``results/``
        while the test runs.
    """
    before = _results_entries()
    yield
    new = sorted(_results_entries() - before)
    assert not new, (
        f"this test wrote into the repo-root results/ directory: {new}. "
        "A trainer's relative output_dir resolves against the repo root; route "
        "the config through tmp_path (e.g. dataclasses.replace(config, "
        "output_dir=str(tmp_path))). results/ is gitignored and untracked -- do "
        "NOT delete what this assertion reports, and do NOT make this fixture "
        "clean up after itself.")


# --- R-165: process-global RNG state is captured and restored per test ------
#
# These imports deliberately sit at the BOTTOM of this module, after the
# `TF_CPP_MIN_LOG_LEVEL` assignment above: TensorFlow reads that variable when
# it is imported, so importing it any earlier in this file would silently undo
# the log-level setting for the whole session.
import random

import numpy as np
from keras.src.backend.common import global_state as _keras_global_state
from tensorflow.python.eager import context as _tf_eager_context
from tensorflow.python.ops import stateful_random_ops as _tf_stateful_random


def _capture_global_rng_state() -> dict:
    """Snapshot every process-global RNG stream a test can mutate.

    The five streams, and why each is here:

    1. ``random`` -- Python's global Mersenne Twister. Written by
       ``keras.utils.set_random_seed``.
    2. ``numpy.random`` -- the LEGACY global generator behind ``np.random.*``.
       Written by ``keras.utils.set_random_seed``. A ``np.random.Generator``
       instance a test creates for itself is not global and needs no capture.
    3. The TensorFlow eager GLOBAL seed (``context.global_seed()``) plus the
       ``Random`` instance TF derives per-op seeds from
       (``context.context()._rng``, absent until a seed is first set). Written
       by ``keras.utils.set_random_seed`` via ``tf.random.set_seed``.
    4. The Keras 3 GLOBAL ``SeedGenerator`` backing unseeded ``keras.random.*``
       calls. NOT written by ``set_random_seed`` -- it advances on every
       unseeded draw instead, which is the same cross-test coupling by a
       different door. It is created lazily, so ``None`` here means "no test has
       drawn from it yet" and must be restored as absence, not as a value.
    5. ``tf.random``'s global ``Generator`` (``tf.random.get_global_generator``).
       Read WITHOUT the public getter, because that getter CREATES the generator
       as a side effect -- calling it here would manufacture the very global
       state this function exists to observe.

    :return: an opaque snapshot for :func:`_restore_global_rng_state`.
    :rtype: dict
    """
    ctx = _tf_eager_context.context()
    tf_op_rng = getattr(ctx, "_rng", None)
    keras_seed_generator = _keras_global_state.get_global_attribute(
        "global_seed_generator")
    tf_global_generator = _tf_stateful_random.global_generator

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "tf_global_seed": _tf_eager_context.global_seed(),
        "tf_op_rng_present": tf_op_rng is not None,
        "tf_op_rng": tf_op_rng.getstate() if tf_op_rng is not None else None,
        "keras_seed_generator": keras_seed_generator,
        "keras_seed_generator_state": (
            None if keras_seed_generator is None
            else np.array(keras_seed_generator.state)),
        "tf_global_generator": tf_global_generator,
        "tf_global_generator_state": (
            None if tf_global_generator is None
            else np.array(tf_global_generator.state)),
    }


def _restore_global_rng_state(snapshot: dict) -> None:
    """Write a :func:`_capture_global_rng_state` snapshot back, exactly.

    The TF eager seed is restored through ``context.set_global_seed`` rather
    than by assigning ``ctx._seed``, because that call also clears the kernel
    cache -- ops that already captured a derived seed must be invalidated, which
    a bare attribute write does not do. Its side effect of building a FRESH
    per-op ``Random`` is then undone by writing the captured ``_rng`` state back
    on top, so a test that ran 30 seeded ops does not hand the next test a
    rewound op-seed stream.

    WHAT IS NOT FULLY RESTORABLE, stated rather than pretended:

    * If a test is the first to touch ``tf.random``'s global ``Generator``, the
      generator object it created is left in place; only the module global is
      reset to ``None`` when it was absent before, so the next consumer builds a
      fresh one. That generator is built ``from_non_deterministic_state()``, so
      neither leaving it nor dropping it makes any later assertion order-
      dependent -- there was no determinism to preserve either way.
    * Anything a test seeded on a *device* (e.g. cuDNN kernel state) or inside a
      dataset iterator it left alive is outside this fixture's reach.

    :param snapshot: the dict returned by :func:`_capture_global_rng_state`.
    """
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])

    ctx = _tf_eager_context.context()
    if _tf_eager_context.global_seed() != snapshot["tf_global_seed"]:
        _tf_eager_context.set_global_seed(snapshot["tf_global_seed"])
    if snapshot["tf_op_rng_present"]:
        op_rng = getattr(ctx, "_rng", None)
        if op_rng is None:
            op_rng = random.Random()
            ctx._rng = op_rng
        op_rng.setstate(snapshot["tf_op_rng"])

    previous_keras_generator = snapshot["keras_seed_generator"]
    if previous_keras_generator is None:
        # Created during the test: drop the reference so the next unseeded
        # `keras.random.*` call rebuilds it from the (now restored) Python RNG.
        if _keras_global_state.get_global_attribute(
                "global_seed_generator") is not None:
            _keras_global_state.set_global_attribute(
                "global_seed_generator", None)
    else:
        previous_keras_generator.state.assign(
            snapshot["keras_seed_generator_state"])
        _keras_global_state.set_global_attribute(
            "global_seed_generator", previous_keras_generator)

    previous_tf_generator = snapshot["tf_global_generator"]
    if previous_tf_generator is None:
        _tf_stateful_random.global_generator = None
    else:
        previous_tf_generator.state.assign(
            snapshot["tf_global_generator_state"])
        _tf_stateful_random.global_generator = previous_tf_generator


# DECISION plan-2026-08-22T035419-a11304c8/D-006
# `keras.utils.set_random_seed(N)` is called BARE -- no fixture, no `finally` --
# inside test bodies in at least nine directories (`test_sam2` plus the eight
# R-165 packages `test_bert`, `test_distilbert`, `test_fnet`, `test_gemma`,
# `test_gpt2`, `test_modern_bert`, `test_qwen`, `test_tree_transformer`; 3/6/11/
# 3/1/5/2/1 call sites respectively). It rewrites Python's `random`, NumPy's
# global RNG and TF's global seed for the rest of the PROCESS, so every later
# test's randomly-initialized weights depend on collection order. Proven by
# intervention: deselecting the single seeding test at
# `test_models/test_sam2/test_training_model.py:303` flips
# `TestIoUSupervision::test_the_iou_loss_equals_a_hand_computed_GATED_value`
# from RED to GREEN in the same file, same process.
# WHAT NOT TO DO: do not "fix" this by wrapping each of those ~32 call sites in
# its own `try/finally`. The leak is cross-FILE and cross-DIRECTORY -- a
# per-site guard cannot protect a test in another package, it is 32 copies of
# which the 33rd is the one that forgets, and it does nothing about the NEXT
# bare seed call somebody writes. Centralizing it here is the same ruling, for
# the same reason, that `tests/test_layers/conftest.py` already records for the
# dtype policy (D-031) and TF32.
# Do not narrow this to `random` + `numpy` either: the sam2 failure is driven by
# Keras weight initializers, which draw from the TF/Keras seed streams, not from
# Python's.
@pytest.fixture(autouse=True)
def _restore_process_global_rng_state():
    """Restore every process-global RNG stream after each test (R-165).

    Capture / restore-in-``finally`` / assert-the-restoration, the same harness
    `tests/test_layers/conftest.py`'s `dtype_policy` and `tf32_disabled`
    fixtures use for the other two process-globals this suite mutates.

    The restore runs even when the test body raises -- a test that fails AFTER
    calling `set_random_seed` is exactly the case that would otherwise poison
    the rest of the session, and it is the case a `try`-less teardown misses.

    :yield: nothing; this fixture is state management only.
    """
    snapshot = _capture_global_rng_state()
    try:
        yield
    finally:
        _restore_global_rng_state(snapshot)
        # Assert the restoration, per the `tf32_disabled` precedent: a restore
        # that silently no-ops is indistinguishable from no fixture at all.
        assert random.getstate() == snapshot["python"], (
            "the Python global RNG state was not restored after this test")
        assert _tf_eager_context.global_seed() == snapshot["tf_global_seed"], (
            "the TensorFlow global seed was not restored after this test")
