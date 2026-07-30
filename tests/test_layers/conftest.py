"""
Shared pytest fixtures for `tests/test_layers/`.

Hosts the two PROCESS-GLOBAL settings this tree mutates: the Keras mixed-precision
policy (`dtype_policy`, used by the Energy Transformer dtype tests, plan
plan_2026-07-13_57c9833e success criterion S13) and TensorFloat-32 matmul
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

    Same capture / restore-in-`finally` / ASSERT-the-restoration harness as
    `test_transformers/test_gated_linear_attention_block.py`'s
    `test_chunked_matches_sequential_float32_without_tf32` (~`:1705-1733`) — one
    convention, not two. The prior value is CAPTURED, never assumed to be `True`,
    because another module may already have disabled it.

    :yield: `False`, the TF32 state in force for the module's tests.
    :rtype: bool
    """
    global _TF32_SCOPED_OFF
    previous = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    _TF32_SCOPED_OFF = True
    try:
        yield False
    finally:
        # Runs even if a test body raises. A leaked toggle silently changes every
        # later float32 precision assertion in the session.
        tf.config.experimental.enable_tensor_float_32_execution(previous)
        _TF32_SCOPED_OFF = False
        assert (
            tf.config.experimental.tensor_float_32_execution_enabled() == previous
        ), "TF32 setting leaked out of this module"


@pytest.fixture(autouse=True)
def _tf32_leak_canary():
    """Fail the FIRST test that runs after any TF32 leak in this tree.

    Checked at setup, so a module that mutates TF32 without restoring it is caught
    by the next test to run — which is precisely the cross-file, collection-order
    coupling the four import-time disables used to create.

    This canary is device-independent and therefore fully verifiable on CPU: it
    asserts on the process-global FLAG (the coupling mechanism), not on any number
    the flag influences (the device-dependent consequence).

    It is safe as an autouse over the whole tree because
    `grep -rn tensor_float_32 tests/` shows exactly two mutation sites left, both
    restore-safe: `tf32_disabled` above and the self-contained toggle in
    `test_gated_linear_attention_block.py`. A new unrestored mutation anywhere
    under `tests/test_layers/` is exactly what this is meant to turn RED.

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
