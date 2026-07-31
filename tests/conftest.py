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
