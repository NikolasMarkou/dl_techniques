"""Shared fixtures for `tests/test_models/test_dino/`.

**Why this file exists at all** (MEASURED, plan-2026-08-01T105809-dc0c402e step 7):
`tests/test_layers/conftest.py` already defines a restore-safe parametrized
`dtype_policy` fixture, and the obvious move is to reuse it. It is NOT reachable
from here. pytest resolves `conftest.py` by DIRECTORY ancestry, and
`tests/test_models/` is a sibling of `tests/test_layers/`, not a descendant.
Measured by requesting the fixture from a throwaway test in this directory::

    E       fixture 'dtype_policy' not found
    >       available fixtures: anyio_backend, ..., golden_reference_device, ...

`tests/conftest.py` (the common ancestor) defines only `golden_reference_device`.
So the choice was between adding `dtype_policy` to `tests/conftest.py` — which
would put a process-global mutation fixture in scope for the WHOLE suite,
including trees that have never been audited for policy sensitivity — and
copying the restore-safe pattern locally. This file is the local copy. Do NOT
"deduplicate" it by importing from `tests/test_layers/conftest.py`: a
cross-directory conftest import is not a supported pytest arrangement and would
couple two independent test trees' collection order.

The pattern itself is copied deliberately and in full, including the
`finally:` teardown AND the assertion that the restoration actually happened —
`keras.mixed_precision.set_global_policy` is PROCESS-GLOBAL, so a policy leaked
by one test silently re-types every model built after it in the session. The
signature of that failure is a rising count of failures in files you never
touched.
"""

import keras
import pytest

# float32 is the no-regression baseline. mixed_float16 is the one that MEASURED
# a real defect here: `DINOHead.call`'s L2 normalization reduced `sum(x**2)`
# in fp16 and overflowed 65504, returning a head output of EXACTLY zero (see
# D-020 and `test_dino_v1.py::TestDINOHeadMixedPrecision`). float64 proves the
# fix does not pin the computation to fp32 behind the caller's back.
DTYPE_POLICIES = ("float32", "mixed_float16", "float64")


@pytest.fixture(params=DTYPE_POLICIES)
def dtype_policy(request):
    """Set the Keras GLOBAL dtype policy for one test, then ALWAYS restore it.

    A test that merely REQUESTS this fixture proves nothing — if the policy
    silently failed to apply, every assertion in the body would still run under
    float32 and pass. Every test using this fixture therefore asserts the ACTIVE
    policy inside its own body via `keras.mixed_precision.dtype_policy().name`.

    :param request: pytest request carrying the parametrized policy name.
    :yield: the policy name in force for this test (e.g. ``'mixed_float16'``).
    :rtype: str
    """
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy(request.param)
    try:
        yield request.param
    finally:
        # Runs even if the test body raises. A leaked policy poisons the session.
        keras.mixed_precision.set_global_policy(previous)
        assert keras.mixed_precision.global_policy().name == previous, (
            "the global dtype policy leaked out of this test"
        )
