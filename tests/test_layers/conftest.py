"""
Shared pytest fixtures for `tests/test_layers/`.

Currently hosts exactly one thing: the global mixed-precision policy fixture used by the
Energy Transformer dtype tests (plan plan_2026-07-13_57c9833e, success criterion S13).

**Why it lives HERE and not in each test module.** `keras.mixed_precision.set_global_policy`
is PROCESS-GLOBAL. A test that sets it and fails to reset it corrupts every subsequent test
in the session (the signature is a rising failure count in test files you never touched).
The reset therefore lives in exactly ONE place, in a fixture teardown that runs even when
the test body raises — rather than being copy-pasted into three test modules, where the
third copy is the one that forgets the `finally`.
"""

import keras
import pytest

# ---------------------------------------------------------------------

# The dtypes every Energy Transformer layer must survive. `mixed_float16` is the one that
# shipped 512/512 NaN in iteration 1 (`_MASK_BIAS_VALUE = -1e9` overflows to `-inf` in
# fp16); `float32` is the no-regression baseline; `float64` proves the fix does not pin the
# computation to fp32 behind the caller's back.
DTYPE_POLICIES = ("float32", "mixed_float16", "float64")


@pytest.fixture(params=DTYPE_POLICIES)
def dtype_policy(request):
    """Set the Keras GLOBAL dtype policy for one test, then ALWAYS restore it.

    :param request: pytest request carrying the parametrized policy name.

    :yield: The policy name currently in force (e.g. ``'mixed_float16'``).
    :rtype: str
    """
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy(request.param)
    try:
        yield request.param
    finally:
        # Runs even if the test body raises. A leaked policy poisons the whole session.
        keras.mixed_precision.set_global_policy(previous)
