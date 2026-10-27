"""
Shared pytest fixtures for `tests/test_layers/`.

Hosts the PROCESS-GLOBAL settings this tree mutates: the Keras mixed-precision
policy (`dtype_policy`, `mixed_float16_policy`, used by the Energy Transformer dtype
tests, plan plan_2026-07-13_57c9833e success criterion S13), a genuine float64 arm
(`float64_policy`: the policy AND `floatx`) and TensorFloat-32 matmul
(`tf32_disabled` + `_tf32_leak_canary`).

**Why they live HERE and not in each test module.** `keras.mixed_precision.set_global_policy`
and `tf.config.experimental.enable_tensor_float_32_execution` are both PROCESS-GLOBAL. A
test that sets one and fails to reset it corrupts every subsequent test in the session (the
signature is a rising failure count in test files you never touched). The reset therefore
lives in exactly ONE place, in a fixture teardown that runs even when the test body raises —
rather than being copy-pasted into three test modules, where the third copy is the one that
forgets the `finally`.
"""

import keras
import pytest
import tensorflow as tf

from tests.numerics import true_float32_matmul

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


@pytest.fixture
def mixed_float16_policy():
    """Force ``mixed_float16`` for one test, then ALWAYS restore the previous policy.

    The unparametrized sibling of :func:`dtype_policy`, for guards that are only
    meaningful under half precision (an fp16 overflow cannot be observed at
    float32, so parametrizing over all three policies would yield two tests that
    cannot fail). The restore lives here, with :func:`dtype_policy`'s, for the
    reason given in this module's docstring.

    :yield: the literal string ``'mixed_float16'``.
    :rtype: str
    """
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield "mixed_float16"
    finally:
        keras.mixed_precision.set_global_policy(previous)


@pytest.fixture
def float64_policy():
    """Force a genuine float64 policy for one test, then ALWAYS restore both globals.

    The ONE owner of the float64 arm for every directory under ``tests/test_layers/``.
    ``dtype_policy`` and ``mixed_float16_policy`` above set the global policy and
    nothing else. That is correct for mixed precision, but it is NOT sufficient for
    float64: with the policy alone, ``keras.backend.floatx()`` stays ``'float32'``, so
    ``keras.ops.convert_to_tensor`` and ``keras.Input`` keep producing float32 and the
    "float64" arm silently agrees with float32 to eight digits. A precision guard
    written on top of that cannot fail (guide rule L-38). This fixture therefore sets
    ``floatx`` IN ADDITION to the policy and restores BOTH in a ``finally``.

    **Every test using it must additionally assert the REALISED input dtype.** The
    fixture makes float64 reachable; only the assert proves it arrived. A related
    measured trap: ``set_floatx`` does not re-point an already-materialised policy, so a
    layer built before the fixture runs stays float32 -- build inside the test, never at
    module import.

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


# ---------------------------------------------------------------------
# TensorFloat-32
# ---------------------------------------------------------------------

# The ambient TF32 setting at conftest import, i.e. BEFORE any test module under
# `tests/test_layers/` has been imported. `True` on a stock process (measured:
# `tf.config.experimental.tensor_float_32_execution_enabled()` is True even with no
# CUDA device, because it is a config flag whose NUMERIC effect — not its value — is
# device-dependent). Captured rather than assumed, because a pytest invocation wider
# than this tree may legitimately have set it first.
_TF32_SESSION_BASELINE = tf.config.experimental.tensor_float_32_execution_enabled()

# Set by `tf32_disabled` for the duration of a module that opted in, so the canary
# below can tell "a module deliberately scoped TF32 off" from "a module leaked".
_TF32_SCOPED_OFF = False


# DECISION plan-2026-07-30T140922-8af1028f/D-031
# Four test modules used to call `enable_tensor_float_32_execution(False)` as a
# TOP-LEVEL statement with no restore, so whichever of them pytest collected first
# decided TF32 for every test that ran after it in the process. Do NOT reintroduce
# that: an import-time global with no teardown is why an unrelated measurement in
# `test_capsule_routing_attention.py` swings ~1500x between "this file alone" and
# "the whole directory", and why a guard in `test_clifford_block_ds_v2.py` was
# passing on TF32 rounding noise until an unrelated module removed it.
# Do NOT replace this with a per-module `try/finally` either — that is the copy that
# gets forgotten (this file's own docstring records why the dtype policy is
# centralised for exactly the same reason).
@pytest.fixture(scope="module")
def tf32_disabled():
    """Disable TF32 tensor-core matmul for ONE module, then ALWAYS restore it.

    Opt in per module with `pytestmark = pytest.mark.usefixtures("tf32_disabled")`.

    The capture / restore-in-`finally` / ASSERT-the-restoration harness is
    `tests/numerics.py::true_float32_matmul`, the suite's one TF32 writer (D-164).
    The prior value is CAPTURED there, never assumed to be `True`, because another
    module may already have disabled it.

    :yield: `False`, the TF32 state in force for the module's tests.
    :rtype: bool
    """
    global _TF32_SCOPED_OFF
    with true_float32_matmul():
        _TF32_SCOPED_OFF = True
        try:
            yield False
        finally:
            # Runs even if a test body raises; the helper then restores and asserts.
            _TF32_SCOPED_OFF = False


@pytest.fixture(autouse=True)
def _tf32_leak_canary():
    """Fail the FIRST test that runs after any TF32 leak in this tree.

    Checked at setup, so a module that mutates TF32 without restoring it is caught
    by the next test to run — which is precisely the cross-file, collection-order
    coupling the four import-time disables used to create.

    This canary is device-independent and therefore fully verifiable on CPU: it
    asserts on the process-global FLAG (the coupling mechanism), not on any number
    the flag influences (the device-dependent consequence).

    It is safe as an autouse over the whole tree because the only TF32 writer under
    `tests/` is `tests/numerics.py::tf32_execution`, which restores and asserts
    (enforced by `tests/test_the_tf32_flag_is_set_only_in_numerics.py`, D-164). A
    new unrestored mutation anywhere under `tests/test_layers/` is exactly what
    this is meant to turn RED.

    # DECISION plan-2026-07-30T140922-8af1028f/D-038
    The canary asserts against the EXPECTED value for the current scope; it does
    NOT skip while a module has TF32 scoped off. Do NOT "simplify" this back to
    `if not _TF32_SCOPED_OFF: assert ...` -- that shape disabled the canary for
    the entire duration of every opted-in module, i.e. it was inert in exactly
    the four files that manipulate TF32, which are the only files where a leak
    can originate. A test inside `test_energy_transformer.py` that enabled TF32
    without restoring would then run the remaining ~160 tests of that module in
    the wrong regime, green, and `tf32_disabled`'s own teardown would repair the
    damage silently (it writes `previous` back unconditionally, so its
    `== previous` assertion cannot see it either).
    """
    expected = False if _TF32_SCOPED_OFF else _TF32_SESSION_BASELINE
    actual = tf.config.experimental.tensor_float_32_execution_enabled()
    scope = (
        "this module has TF32 scoped OFF via the `tf32_disabled` fixture"
        if _TF32_SCOPED_OFF
        else f"TF32 was {_TF32_SESSION_BASELINE} at session start"
    )
    assert actual == expected, (
        f"TF32 leaked: the process-global tensor-float-32 setting is {actual} "
        f"at the start of this test, but {scope}, so it should be {expected}. "
        "Some test or module mutated it without restoring; every float32 "
        "tolerance that runs after it now depends on execution order."
    )
    yield
