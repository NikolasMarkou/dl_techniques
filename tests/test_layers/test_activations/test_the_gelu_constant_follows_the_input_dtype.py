"""Single-claim guard: `GELU` and `xGELU` build their `sqrt(2)` at the INPUT's dtype.

The claim under test is the `# DECISION plan-2026-08-23T203721-009b7ccf/D-017` anchor that
appears twice in `src/dl_techniques/layers/activations/expanded_activations.py`::

    root_two = keras.ops.cast(keras.ops.sqrt(2.0), inputs.dtype)   # shipped
    root_two = keras.ops.sqrt(2.0)                                 # the pre-D-017 form

`keras.ops.sqrt(2.0)` returns a **float32 tensor under every dtype policy**. Dividing a
float16 (or float64) tensor by it is a hard TypeError in the TensorFlow backend, so the bare
form does not merely lose precision -- it makes these two layers **unable to run at all**
outside float32. That is the failure that once blocked BERT's whole mixed-precision path.

**Why this file exists: the anchor was measured UNPINNED.** Reverting both sites to the bare
form and running the full directory gives `848 passed, 4 skipped, 4 xfailed` -- a clean
sweep, with GELU and xGELU dead on any mixed-precision or float64 forward pass. Not one of
the 67 tests in `test_expanded_activations.py` leaves the default float32 policy, so the
anchored repair could be "tidied away" by a future reader with no test saying otherwise.
(Evidence: `plans/plan-2026-08-27T103353-60745fe0/evidence/iter2-step-10-unpinned-proof.txt`.)

Measured behaviour of the two forms, one process per policy
(`evidence/iter2-step-10-d017-probe-*.txt`):

=============  ===============================  ==============================
policy         `keras.ops.sqrt(2.0)` (bare)     `ops.cast(..., inputs.dtype)`
=============  ===============================  ==============================
float32        works (constant is float32)      works, identical bits
mixed_float16  **TypeError: float16 != float32** works, output float16
float64        **TypeError: float64 != float32** works, output float64
=============  ===============================  ==============================

**One thing this file deliberately does NOT pin: the constant's own precision.** The shipped
form evaluates `sqrt(2.0)` in float32 and then widens, so under a float64 policy `root_two`
is `1.4142135381698608` rather than `1.4142135623730951`, and the layer's output sits
`5.023e-09` from a true-`sqrt(2)` reference. That is a real (small) residual narrowing of the
same L-30 family, it is REPORTED in this plan's findings, and it is out of this step's scope
to fix. The float64 arm below is therefore written against **either** admissible constant, so
it stays GREEN today and stays GREEN after that residual is repaired, while still going RED
on the failure it is for -- a compute path that ran in float32, which misses BOTH references
by `2.44e-07`, six orders above the bound.
"""

import keras
import numpy as np
import pytest
from scipy.special import erf as _scipy_erf

from dl_techniques.layers.activations.expanded_activations import GELU, xGELU

# --------------------------------------------------------------------------- #
# Fixture geometry.
#
# N = 512 along the feature axis, per guide rule L-37: a dtype arm measured at a
# toy width can miss a reduction hazard that only appears at realistic size.
# --------------------------------------------------------------------------- #
BATCH = 64
WIDTH = 512

#: Two admissible values for the constant, both float64. The layer's own
#: `sqrt(2.0)` is evaluated in float32 and then widened, so under a float64
#: policy it delivers `_ROOT_TWO_F32_ROUNDED`; a version that evaluated the
#: square root at the input dtype would deliver `_ROOT_TWO_EXACT`. Neither is
#: the failure this module is about, so the float64 arm accepts either.
_ROOT_TWO_EXACT = np.sqrt(2.0)
_ROOT_TWO_F32_ROUNDED = np.float64(np.float32(np.sqrt(2.0)))

#: Bounds, each set from a MEASURED number with orders of headroom rather than
#: tuned to just-pass. Measured `max|layer - float64 reference|` on this fixture:
#: float32 2.444e-07, mixed_float16 1.922e-03, float64 4.441e-16 (against the
#: f32-rounded constant). The number that matters for each is the GAP to the
#: state it must reject -- for the float64 arm, a float32 compute path reads
#: 2.44e-07, six orders above its bound.
_ATOL_FLOAT32 = 1e-06
_ATOL_FLOAT16 = 1e-02
_ATOL_FLOAT64 = 1e-13


def _reference_gelu(x64: np.ndarray, root_two: float) -> np.ndarray:
    """Exact-GELU reference in float64 NumPy, with an explicit `sqrt(2)`.

    Taken from the published definition ``0.5 * x * (1 + erf(x / sqrt(2)))``
    (Hendrycks & Gimpel, 2016) with `scipy`'s `erf`, not from the layer.

    With the default ``alpha_initializer='zeros'``, `xGELU`'s widened form
    ``x * (gate * (1 + 2a) - a)`` collapses to ``x * gate`` exactly, so the same
    reference serves both classes. That collapse is asserted, not assumed.

    :param x64: input values, float64.
    :type x64: numpy.ndarray
    :param root_two: the value of ``sqrt(2)`` to divide by.
    :type root_two: float
    :return: the reference output, float64, same shape as ``x64``.
    :rtype: numpy.ndarray
    """
    return 0.5 * x64 * (1.0 + _scipy_erf(x64 / root_two))


def _min_deviation_over_admissible_constants(y64: np.ndarray, x64: np.ndarray) -> float:
    """Smallest `max|y - reference|` over the two admissible `sqrt(2)` values.

    See the module docstring: which of the two the layer uses is deliberately
    not pinned here; that the arithmetic ran at the input's precision is.

    :param y64: the layer's output, promoted to float64.
    :type y64: numpy.ndarray
    :param x64: the inputs the layer actually received, promoted to float64.
    :type x64: numpy.ndarray
    :return: the smaller of the two maximum absolute deviations.
    :rtype: float
    """
    return min(
        float(np.max(np.abs(y64 - _reference_gelu(x64, c))))
        for c in (_ROOT_TWO_EXACT, _ROOT_TWO_F32_ROUNDED)
    )


def _run(layer_cls, x_np: np.ndarray, compute_dtype: str):
    """Build the layer INSIDE the active policy and push one batch through it.

    Building inside the test is load-bearing, not stylistic: a layer materialised
    before the policy fixture ran keeps the old policy, and the arm then silently
    measures float32 (a measured false negative recorded in this plan's findings).

    :param layer_cls: `GELU` or `xGELU`.
    :param x_np: the input batch as float64 NumPy.
    :type x_np: numpy.ndarray
    :param compute_dtype: the dtype the input must be cast to and arrive as.
    :type compute_dtype: str
    :return: ``(output as float64 numpy, realised input as float64 numpy)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    x = keras.ops.cast(keras.ops.convert_to_tensor(x_np), compute_dtype)
    assert keras.backend.standardize_dtype(x.dtype) == compute_dtype, (
        f"premise violated: the REALISED input dtype is "
        f"{keras.backend.standardize_dtype(x.dtype)!r}, not {compute_dtype!r}. "
        f"Setting the policy without `keras.backend.set_floatx` leaves tensors at "
        f"float32 and this arm cannot fail (guide rule L-38)."
    )

    layer = layer_cls()
    y = layer(x)

    if layer_cls is xGELU:
        alpha = np.asarray(keras.ops.convert_to_numpy(layer.alpha), dtype=np.float64)
        assert np.all(alpha == 0.0), (
            "premise violated: xGELU's default alpha is no longer all-zeros, so "
            "`x * (gate*(1+2a) - a)` no longer collapses to the plain GELU "
            "reference this module compares against"
        )

    assert keras.backend.standardize_dtype(y.dtype) == compute_dtype, (
        f"{layer_cls.__name__} returned {keras.backend.standardize_dtype(y.dtype)!r} "
        f"under a {compute_dtype!r} compute dtype"
    )
    return (
        np.asarray(keras.ops.convert_to_numpy(y), dtype=np.float64),
        np.asarray(keras.ops.convert_to_numpy(x), dtype=np.float64),
    )


@pytest.mark.parametrize("layer_cls", [GELU, xGELU], ids=["GELU", "xGELU"])
class TestTheGeluConstantFollowsTheInputDtype:
    """One arm per policy. The two mixed/wide arms are the ones that can go RED."""

    def test_float32_control(self, layer_cls) -> None:
        """CONTROL. Must stay GREEN in BOTH directions -- the bare form works here.

        Its job is to show that the two arms below are detecting a *dtype* defect
        and not a broken reference: the same input, the same reference and the
        same code path, at the one policy where `keras.ops.sqrt(2.0)` happens to
        agree with the input's dtype by luck.
        """
        x_np = np.random.default_rng(0).standard_normal((BATCH, WIDTH))
        y64, x64 = _run(layer_cls, x_np, "float32")

        dev = _min_deviation_over_admissible_constants(y64, x64)
        assert dev < _ATOL_FLOAT32, (
            f"float32 control deviates from the published GELU reference by "
            f"{dev:.6e}, above {_ATOL_FLOAT32:.1e}. This is the arm that is "
            f"supposed to be unaffected by D-017; a failure here means the "
            f"reference or the fixture is wrong, not the dtype handling."
        )

    def test_mixed_float16_arm(self, layer_cls, mixed_float16_policy) -> None:
        """RED when either D-017 site reverts to the bare `keras.ops.sqrt(2.0)`.

        The bare form does not degrade under half precision, it RAISES:
        ``TypeError: `x` and `y` must have the same dtype, got tf.float16 !=
        tf.float32``. This arm therefore fails at the layer call itself, before
        any tolerance is consulted -- which is the strongest shape a guard can
        have.
        """
        # Scale 3 and an explicit clamp: fp16's ceiling is 65504, and an input
        # that has already overflowed it is itself non-finite, which produces a
        # false all-NaN reading that looks exactly like a layer defect. The
        # finiteness assert is on the INPUT for that reason (a near-miss recorded
        # in this plan's findings).
        x_np = np.clip(
            np.random.default_rng(0).standard_normal((BATCH, WIDTH)) * 3.0,
            -6.0e4,
            6.0e4,
        )
        assert np.all(np.isfinite(np.asarray(x_np, dtype=np.float16))), (
            "premise violated: the fp16 INPUT is already non-finite, so a "
            "non-finite output would be garbage-in-garbage-out, not a defect"
        )

        y64, x64 = _run(layer_cls, x_np, "float16")

        assert np.all(np.isfinite(y64)), (
            f"{layer_cls.__name__} produced non-finite output under "
            f"mixed_float16 on an input asserted finite before the call"
        )
        dev = _min_deviation_over_admissible_constants(y64, x64)
        assert dev < _ATOL_FLOAT16, (
            f"max|layer - float64 reference| is {dev:.6e} under mixed_float16, "
            f"above {_ATOL_FLOAT16:.1e} (measured 1.922e-03). The half-precision "
            f"path is no longer computing GELU."
        )

    def test_float64_arm(self, layer_cls, float64_policy) -> None:
        """RED when a D-017 site reverts, AND when the whole path is narrowed to float32.

        Two distinct failures land here. The bare `keras.ops.sqrt(2.0)` raises
        ``TypeError: ... got tf.float64 != tf.float32`` at the call. And an
        absolute `ops.cast(inputs, "float32")` anywhere in this `call()` -- the
        L-30 defect fixed elsewhere in this package -- would leave the deviation
        at float32's `2.44e-07`, six orders above the bound below.

        The bound accepts either admissible `sqrt(2)`; see the module docstring
        for why the constant's own `5.023e-09` residual is reported rather than
        pinned.
        """
        x_np = np.random.default_rng(0).standard_normal((BATCH, WIDTH))
        y64, x64 = _run(layer_cls, x_np, "float64")

        dev = _min_deviation_over_admissible_constants(y64, x64)
        assert dev < _ATOL_FLOAT64, (
            f"max|layer - float64 reference| is {dev:.6e} under a float64 policy "
            f"(with `set_floatx('float64')` and the realised input dtype "
            f"asserted), above {_ATOL_FLOAT64:.1e}. A reading near 2.44e-07 means "
            f"the arithmetic ran in float32 and the float64 policy bought "
            f"nothing -- guide rule L-30."
        )
