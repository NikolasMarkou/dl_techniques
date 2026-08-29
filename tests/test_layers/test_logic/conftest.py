"""Directory-wide instrumentation for the ``layers/logic`` suite.

Two fixtures live here because §13.1 rule 4 requires process-global
state to be owned by exactly one fixture that restores it in
``finally`` and asserts the restoration:

* ``finite_forward_observer`` -- autouse. Installs the §16.3 finiteness
  check on every forward pass of the four classes, for every test in
  this directory, and removes it afterwards.
* ``dtype_policy`` -- opt-in and parametrized. Owns the Keras global
  dtype policy and ``floatx`` for the precision arms.
"""

import keras
import pytest

from .logic_subject_oracle import FiniteForwardObserver

#: The three arms of §13.2.6. ``float32`` is the control, first.
DTYPE_POLICIES = ("float32", "mixed_float16", "float64")


# DECISION plan-2026-08-29T112804-aff039c4/D-005 -- the §16.3
# finiteness rule is an autouse OBSERVER, not a line in each test.
# Do NOT "simplify" it by adding isfinite to the 80 forward tests
# instead: the 81st would silently omit it. See decisions.md D-005.
@pytest.fixture(autouse=True)
def finite_forward_observer():
    """Assert every concrete forward output of the four classes is
    finite, for the duration of one test.

    Autouse so the rule holds for tests written before this instrument
    existed and for tests written after it. Yields the observer so a
    meta-test can read its counters.
    """
    observer = FiniteForwardObserver()
    observer.install()
    try:
        yield observer
    finally:
        observer.uninstall()


@pytest.fixture(params=DTYPE_POLICIES)
def dtype_policy(request):
    """Set the Keras global dtype policy for one test, then restore it.

    ``floatx`` is set alongside the policy for the ``float64`` arm:
    ``keras.Input`` reads ``backend.floatx()``, not the policy, so
    without this the graph rounds to float32 at the boundary and the
    arm becomes a fake reading that agrees with float32 to eight
    digits (§13.2.6).
    """
    previous_policy = keras.mixed_precision.global_policy().name
    previous_floatx = keras.backend.floatx()
    keras.mixed_precision.set_global_policy(request.param)
    if request.param == "float64":
        keras.backend.set_floatx("float64")
    try:
        yield request.param
    finally:
        keras.mixed_precision.set_global_policy(previous_policy)
        keras.backend.set_floatx(previous_floatx)
        assert keras.mixed_precision.global_policy().name == (
            previous_policy
        ), "the dtype policy leaked out of the fixture"
        assert keras.backend.floatx() == previous_floatx, (
            "floatx leaked out of the fixture"
        )
