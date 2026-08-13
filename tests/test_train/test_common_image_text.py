"""Guards for ``train.common.image_text``'s preprocessing constants.

Two things are pinned here, both regressions this module has actually shipped:

1. **Importing ``train.common`` must not allocate a GPU device.** The package
   ``__init__`` re-exports ``IMAGE_MEAN``/``IMAGE_STD``, so *any*
   ``import train.common.<anything>`` executes this module. While those
   constants were built with ``tf.constant(...)`` at module scope, that import
   initialized TF's eager context and created a GPU device — making ``--help``
   allocate a GPU on 125 entry points, forcing four trainers to carry local
   deferred-import workarounds, and once producing a false 12-error test
   "regression" that was really ``cudaSetDevice()`` self-contention.

2. **The constants' numerics are unchanged by being plain Python lists.**
   ``augment_and_normalize`` runs *traced*, inside ``tf.data.Dataset.map``, so
   the check below runs it there rather than eagerly.

The oracle in check 2 transcribes the constants as **literals** and rebuilds the
pre-change ``tf.constant`` form. It deliberately does not import them from the
module under test: an oracle that reads its expected value out of the
implementation agrees by construction and proves nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import tensorflow as tf

import train.common
from train.common.image_text import (
    IMAGE_MEAN,
    IMAGE_STD,
    augment_and_normalize,
)


# ---------------------------------------------------------------------------
# Pre-change constants, TRANSCRIBED AS LITERALS. Never import these from the
# module under test — see the module docstring.
# ---------------------------------------------------------------------------

ORACLE_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
ORACLE_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

DEVICE_LINE = "Created device"


def _subprocess_env() -> dict:
    """Env for the import probes: a visible GPU and TF's INFO logs enabled."""
    env = dict(os.environ)
    # Pin a device so the *absence* assertion is not satisfied vacuously by a
    # box with nothing to allocate; the liveness arm below proves it isn't.
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    # TF_CPP_MIN_LOG_LEVEL >= 1 suppresses the very line we count.
    env["TF_CPP_MIN_LOG_LEVEL"] = "0"
    env["MPLBACKEND"] = "Agg"
    return env


def _run_snippet(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
        timeout=600,
    )


def _count_device_lines(stderr: str) -> int:
    return sum(1 for line in stderr.splitlines() if DEVICE_LINE in line)


# ---------------------------------------------------------------------------
# (a) Import-allocation guard
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def liveness_probe() -> subprocess.CompletedProcess:
    """POSITIVE arm: a subprocess that deliberately DOES allocate a device.

    Without this, "zero ``Created device`` lines" is unfalsifiable — a CPU-only
    box, a wrong ``CUDA_VISIBLE_DEVICES``, or a silenced logger all produce a
    green absence check while the detector is blind.
    """
    return _run_snippet(
        """
        import tensorflow as tf
        # Force real device placement, not just an import.
        with tf.device("/GPU:0"):
            x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
            _ = tf.reduce_sum(x * 2.0)
        print("ALLOCATED", len(tf.config.list_physical_devices("GPU")))
        """
    )


def test_liveness_the_detector_can_see_a_real_device_allocation(liveness_probe):
    """ASSERT-LIVENESS-DEVICE-LINE-OBSERVED.

    Proves the stderr-scan in the absence test below can actually observe the
    thing it claims is missing. If no GPU is reachable here, SKIP — a pass
    would be vacuous.
    """
    if liveness_probe.returncode != 0 or "ALLOCATED" not in liveness_probe.stdout:
        pytest.skip(
            "liveness arm could not run a GPU op (rc="
            f"{liveness_probe.returncode}); the absence assertion cannot be "
            "shown non-vacuous on this machine. stderr tail:\n"
            + liveness_probe.stderr[-2000:]
        )
    if liveness_probe.stdout.strip().endswith(" 0"):
        pytest.skip(
            "no physical GPU visible to the subprocess; the absence assertion "
            "cannot be shown non-vacuous on this machine."
        )
    observed = _count_device_lines(liveness_probe.stderr)
    assert observed >= 1, (
        "ASSERT-LIVENESS-DEVICE-LINE-OBSERVED: a subprocess that placed an op "
        f"on /GPU:0 emitted {observed} {DEVICE_LINE!r} lines on stderr. The "
        "detector used by "
        "test_importing_train_common_allocates_no_gpu_device is therefore "
        "blind, and that test's zero proves nothing.\n"
        f"stderr:\n{liveness_probe.stderr[-4000:]}"
    )


@pytest.mark.parametrize(
    "module", ["train.common", "train.common.args", "train.common.image_text"]
)
def test_importing_train_common_allocates_no_gpu_device(module, liveness_probe):
    """ASSERT-NO-DEVICE-ON-IMPORT.

    Depends on the liveness arm above: this is an absence assertion and is only
    meaningful once the detector is shown to be able to see a presence.
    """
    if liveness_probe.returncode != 0 or "ALLOCATED" not in liveness_probe.stdout:
        pytest.skip("liveness arm unavailable — absence check would be vacuous")
    if _count_device_lines(liveness_probe.stderr) < 1:
        pytest.skip("detector blind (no device line in liveness arm)")

    proc = _run_snippet(f"import {module}")

    # A crashed subprocess emits no device line either, and would sail through
    # a naive absence check.
    assert proc.returncode == 0, (
        f"ASSERT-IMPORT-SUBPROCESS-EXITED-ZERO: `import {module}` exited "
        f"{proc.returncode}. An absence check over a crashed process is "
        f"vacuous.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-4000:]}"
    )

    observed = _count_device_lines(proc.stderr)
    assert observed == 0, (
        f"ASSERT-NO-DEVICE-ON-IMPORT: `import {module}` emitted {observed} "
        f"{DEVICE_LINE!r} line(s) on stderr; expected 0. Something at MODULE "
        "scope in train/common/ is running an eager TF op at import time — see "
        "the D-003 anchor in src/train/common/image_text.py.\n"
        f"stderr:\n{proc.stderr[-4000:]}"
    )


def test_public_constant_names_are_still_importable_and_exported():
    """ASSERT-PUBLIC-NAMES-PRESERVED."""
    assert "IMAGE_MEAN" in train.common.__all__, (
        "ASSERT-PUBLIC-NAMES-PRESERVED: IMAGE_MEAN dropped from "
        "train.common.__all__"
    )
    assert "IMAGE_STD" in train.common.__all__, (
        "ASSERT-PUBLIC-NAMES-PRESERVED: IMAGE_STD dropped from "
        "train.common.__all__"
    )
    assert len(train.common.IMAGE_MEAN) == 3
    assert len(train.common.IMAGE_STD) == 3


# ---------------------------------------------------------------------------
# (b) .map()-path numerics
# ---------------------------------------------------------------------------


def _fixed_uint8_image(image_size: int = 8) -> np.ndarray:
    """A deterministic, non-degenerate uint8 image covering the 0..255 range."""
    n = image_size * image_size * 3
    return (np.arange(n, dtype=np.int64) * 7 % 256).astype(np.uint8).reshape(
        image_size, image_size, 3
    )


def _oracle_normalize(img_uint8: np.ndarray) -> np.ndarray:
    """Pre-change semantics, rebuilt from LITERAL constants.

    Reproduces the exact op sequence ``augment_and_normalize`` used before the
    constants became plain lists: cast -> /255.0 -> subtract a float32
    ``tf.constant`` -> divide by a float32 ``tf.constant``.
    """
    mean = tf.constant(ORACLE_IMAGE_MEAN, dtype=tf.float32)
    std = tf.constant(ORACLE_IMAGE_STD, dtype=tf.float32)
    img = tf.cast(tf.convert_to_tensor(img_uint8), tf.float32) / 255.0
    return ((img - mean) / std).numpy()


def test_augment_and_normalize_in_a_real_map_pipeline_matches_the_oracle():
    """ASSERT-MAP-PATH-BIT-IDENTICAL.

    Runs the REAL pipeline shape (``tf.data.Dataset.map``, i.e. traced through
    AutoGraph) rather than an eager subtraction — a captured Python list and a
    captured ``tf.constant`` are not obviously the same thing under tracing,
    and eager agreement does not license a graph-mode claim.
    """
    image_size = 8
    img = _fixed_uint8_image(image_size)

    ds = tf.data.Dataset.from_tensors(tf.constant(img, dtype=tf.uint8))
    ds = ds.map(
        lambda x: augment_and_normalize(x, image_size, False),
        num_parallel_calls=1,
    )
    actual = next(iter(ds)).numpy()

    expected = _oracle_normalize(img)

    assert actual.shape == expected.shape, (
        f"ASSERT-MAP-PATH-SHAPE: {actual.shape} != {expected.shape}"
    )
    assert actual.dtype == np.float32, (
        f"ASSERT-MAP-PATH-DTYPE: {actual.dtype} != float32"
    )
    max_abs_delta = float(np.max(np.abs(actual - expected)))
    assert max_abs_delta == 0.0, (
        "ASSERT-MAP-PATH-BIT-IDENTICAL: augment_and_normalize's output inside "
        "tf.data.Dataset.map differs from the literal-constant oracle by "
        f"max|delta| = {max_abs_delta!r}; expected exactly 0.0. The "
        "normalization constants or their promotion at the subtraction site "
        "changed."
    )


def test_module_constants_equal_the_transcribed_literals():
    """ASSERT-CONSTANT-VALUES-UNCHANGED.

    Catches a value edit that the .map() check would also catch, but names it
    directly so the failure says *which* number moved.
    """
    assert list(IMAGE_MEAN) == ORACLE_IMAGE_MEAN, (
        f"ASSERT-CONSTANT-VALUES-UNCHANGED: IMAGE_MEAN is {list(IMAGE_MEAN)!r}, "
        f"expected {ORACLE_IMAGE_MEAN!r}"
    )
    assert list(IMAGE_STD) == ORACLE_IMAGE_STD, (
        f"ASSERT-CONSTANT-VALUES-UNCHANGED: IMAGE_STD is {list(IMAGE_STD)!r}, "
        f"expected {ORACLE_IMAGE_STD!r}"
    )
