"""
R-142: the five directories that NAME ``float64`` in a test and never ran one.

Batch 1 charged exactly five packages -- ``SAM``, ``bias_free_denoisers``,
``ideogram4``, ``sd3_mmdit``, ``time_series`` -- with the same shape:
``set_floatx`` count **0** across the directory, while ``float64`` is named in
14 test files for SAM and in one or two for each of the others. The charge is
"the arm is absent", not "float64 is broken".

The mechanism is not what the rule says it is
---------------------------------------------
R-142 is written as "needs ``keras.backend.set_floatx('float64')``, not just the
mixed-precision policy". On Keras 3.8 that is backwards. Both switches work
**on their own** if they are the first thing in the process, and
``set_floatx`` alone is a **no-op** for every layer dtype once anything has read
``keras.mixed_precision.global_policy()`` -- which lazily builds
``DTypePolicy(floatx())`` and CACHES it. Measured, argued and pinned at
``precision_arm_oracle.float64_scope`` (D-074), which sets both and restores
both. That cache is why the audit found these arms "silently running at
float32", and it is why an arm written the way the rule prescribes would still
be one.

What this family asserts, and why the third part is the only interesting one
---------------------------------------------------------------------------
1. The scope took (``floatx``, the global policy, and the MODEL's policy).
2. The float INPUT leaves are float64 -- R-142's own ``inputs[0].dtype``
   requirement. Integer leaves are exempt, the same D-058 rule the precision
   arm uses.
3. Every float OUTPUT leaf is float64. Parts 1 and 2 can only fail on the
   harness; part 3 is the one that can fail on a model, and it is what catches
   a hard-coded ``dtype="float32"`` constant or a cast island that silently
   downgrades the caller's requested precision.

``SAM`` is represented by ``SAMTrainingModel``, for the same reason the R-143
family does it: ``SAM.call``'s ``ops.image.resize`` is a graph-mode limitation
with nothing to do with precision, and the wrapper is the supervised path.
"""

import keras
import pytest

from .precision_arm_oracle import assert_float64_arm, float64_scope
from .precision_arm_subjects import FLOAT64_CHARGED, SUBJECTS, float64_subject


def test_every_charged_package_has_a_float64_subject():
    missing = sorted(set(FLOAT64_CHARGED) - set(SUBJECTS))
    assert not missing, f"charged packages with no subject: {missing}"
    assert len(FLOAT64_CHARGED) == 5, (
        "batch 1 charged exactly five R-142 rows; changing this number needs "
        "the audit row that justifies it")


@pytest.mark.parametrize("name", sorted(FLOAT64_CHARGED))
def test_the_package_really_runs_at_float64(name):
    """All three parts of the arm, for one charged package."""
    build, make_inputs, kwargs = float64_subject(name)
    report = assert_float64_arm(build=build, make_inputs=make_inputs, **kwargs)
    assert report["floatx_seen"] == "float64"
    assert report["policy_seen"] == "float64"


def test_set_floatx_alone_is_a_no_op_once_the_policy_has_been_read():
    """The RED half of D-074, and the reason ``float64_scope`` sets BOTH.

    This is the arm the audit says these five directories were missing --
    written the way R-142 prescribes it. It measures float32, in this process,
    right now. If Keras ever makes ``set_floatx`` invalidate the cached global
    policy, this test fails and ``float64_scope`` becomes simplifiable; until
    then it is the proof that the one-call version cannot be trusted.
    """
    prev_floatx = keras.backend.floatx()
    prev_policy = keras.mixed_precision.global_policy()
    # The cache is populated by this read -- and by every test that ran before.
    assert prev_policy.name in ("float32", "mixed_float16", "float64")
    try:
        keras.backend.set_floatx("float64")
        dense = keras.layers.Dense(3)
        out = dense(keras.ops.zeros((1, 4)))
        assert keras.backend.floatx() == "float64"
        assert "float32" in str(out.dtype), (
            "set_floatx alone now DOES move the layer dtype; the D-074 finding "
            "no longer holds and float64_scope can be simplified")
        assert "float32" in str(dense.kernel.dtype)
    finally:
        keras.mixed_precision.set_global_policy(prev_policy)
        keras.backend.set_floatx(prev_floatx)


def test_the_scope_sets_both_switches_and_restores_both():
    """Anti-leak. ``floatx`` and the policy are process-global; a float64 arm
    that does not restore them silently re-types every test after it."""
    before = (keras.backend.floatx(),
              keras.mixed_precision.global_policy().name)
    with float64_scope():
        assert keras.backend.floatx() == "float64"
        assert keras.mixed_precision.global_policy().name == "float64"
    after = (keras.backend.floatx(),
             keras.mixed_precision.global_policy().name)
    assert before == after, f"the scope leaked: {before} -> {after}"


def test_the_output_dtype_part_can_convict():
    """Liveness for part 3, on a model that pins float32 the way a real one does.

    Without this, ``assert_float64_arm`` could be asserting nothing but its own
    scope: a model whose head casts to float32 must FAIL, and the failure must
    name the output dtype rather than crash somewhere else.
    """
    class _Pinned(keras.Model):
        def __init__(self):
            super().__init__()
            self.d = keras.layers.Dense(3)

        def call(self, inputs, training=False):
            return keras.ops.cast(self.d(inputs), "float32")

    import numpy as np
    with pytest.raises(AssertionError, match="float outputs are"):
        assert_float64_arm(build=_Pinned,
                           make_inputs=lambda: np.zeros((1, 4), "float32"))
