"""Directory-local fixtures for `tests/test_layers/test_activations/`.

Hosts the ONE fixture this directory needs and its parent `tests/test_layers/conftest.py`
deliberately does not provide: a **genuine** float64 arm.

The parent conftest's `dtype_policy` / `mixed_float16_policy` fixtures set
`keras.mixed_precision.set_global_policy(...)` and nothing else. That is correct for the
mixed-precision policies, but it is NOT sufficient for float64: with the policy alone,
`keras.backend.floatx()` stays `'float32'`, so `keras.ops.convert_to_tensor` and
`keras.Input` keep producing float32 and the "float64" arm silently agrees with float32 to
eight digits. A precision guard written on top of that cannot fail. Guide rule L-38.

`float64_policy` therefore sets `floatx` IN ADDITION to the policy, and restores BOTH in a
`finally`. It lives here rather than being copy-pasted into the two modules that need it
(`test_the_dtype_floor_never_narrows.py`,
`test_the_gelu_constant_follows_the_input_dtype.py`), and it is NOT pushed up into
`tests/test_layers/conftest.py`, because `set_floatx` is process-global and several hundred
default-policy parametrizations in sibling directories have no reason to be exposed to an
extra teardown they never ask for.

**Every test using this fixture must additionally assert the REALISED input dtype.** The
fixture makes float64 reachable; only the assert proves it arrived. A related measured trap:
`set_floatx` does not re-point an already-materialised policy, so a layer built before the
fixture runs stays float32 -- build inside the test, never at module import.
"""

import keras
import pytest

# ---------------------------------------------------------------------


@pytest.fixture
def float64_policy():
    """Force a genuine float64 policy for one test, then ALWAYS restore both globals.

    Sets ``keras.backend.set_floatx('float64')`` as well as the global mixed-precision
    policy; see this module's docstring for why the policy alone is not enough.

    :yield: the literal string ``'float64'``.
    :rtype: str
    """
    previous_policy = keras.mixed_precision.global_policy().name
    previous_floatx = keras.backend.floatx()
    keras.backend.set_floatx("float64")
    keras.mixed_precision.set_global_policy("float64")
    try:
        yield "float64"
    finally:
        # Runs even if the test body raises. A leaked policy or a leaked `floatx`
        # poisons every subsequent test in the session.
        keras.mixed_precision.set_global_policy(previous_policy)
        keras.backend.set_floatx(previous_floatx)
