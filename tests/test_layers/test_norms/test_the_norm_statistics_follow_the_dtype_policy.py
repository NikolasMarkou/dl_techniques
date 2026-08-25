"""
Guard for B6: the RMS/band norm family must compute its internal statistics at
no LESS precision than the active dtype policy.

Six layers hardcode ``ops.cast(inputs, "float32")`` in ``call()``. Under a
float64 policy the layer's declared dtype and its returned dtype are both
float64, yet the arithmetic silently runs in float32 - a truncation that no
dtype assertion can see (``test_bias_free_batch_norm.py:254-259`` is exactly
such a dtype-only assertion, and it would NOT catch this).

The probe input is ``[[1e8, 1e8+1, 1e8+2, 1e8+3]]``. The float32 spacing at 1e8
is 8.0, so all four values collapse to exactly 1e8 in float32 while remaining
distinct in float64. The float32 and float64 answers therefore differ by orders
of magnitude more than the tolerance, and every reference value used below is
exactly representable in float64 (sum = 4e8+6, mean = 1e8+1.5), so the
comparison is deterministic rather than reduction-order dependent.

Each case asserts BOTH directions:
  * the layer is within ``1e-12`` of a float64 NumPy reference, and
  * the layer is CLOSER to the float64 reference than to a float32-truncated one.
A one-sided "differs from float32" assertion would be satisfied by noise.
"""

import numpy as np
import pytest
import keras

from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.norms.zero_centered_rms_norm import ZeroCenteredRMSNorm
from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.layers.norms.zero_centered_band_rms_norm import (
    ZeroCenteredBandRMSNorm,
)
from dl_techniques.layers.norms.adaptive_band_rms import AdaptiveBandRMS
from dl_techniques.layers.norms.zero_centered_adaptive_band_rms_norm import (
    ZeroCenteredAdaptiveBandRMS,
)

# Distinguishable in float64, all four collapse onto 1e8 in float32.
PROBE = np.array([[1e8, 1e8 + 1.0, 1e8 + 2.0, 1e8 + 3.0]], dtype=np.float64)

# Absolute tolerance against the float64 reference. The pre-fix miss was
# 1.5e-8, so this bound carries ~4 orders of margin.
FLOAT64_ATOL = 1e-12


@pytest.fixture()
def float64_policy():
    """Activate a process-global float64 policy and restore it unconditionally.

    ``set_floatx`` and ``set_global_policy`` are PROCESS-GLOBAL. Leaking either
    of them would silently change the measurements of every test that runs
    afterwards in the same process, so both are captured before and restored in
    a ``finally``. ``set_global_policy("float64")`` alone is not sufficient for
    a functional graph - ``set_floatx`` is also required.
    """
    original_floatx = keras.backend.floatx()
    original_policy = keras.mixed_precision.global_policy().name
    try:
        keras.backend.set_floatx("float64")
        keras.mixed_precision.set_global_policy("float64")
        yield
    finally:
        keras.mixed_precision.set_global_policy(original_policy)
        keras.backend.set_floatx(original_floatx)


def _reference(
    x64: np.ndarray,
    *,
    dtype: str,
    epsilon: float,
    center: bool,
    clamp_rms: bool,
    band_width: float | None,
) -> np.ndarray:
    """Recompute a layer's forward pass entirely in ``dtype``.

    The reference mirrors ``call()`` of the family: optional mean-centering,
    ``rms = sqrt(mean(x**2) + epsilon)`` (clamped from below by ``epsilon`` in
    the band variants), divide, then - for the band variants - multiply by the
    band scale.

    The band scale is NOT disabled; it is folded in as its known constant. All
    four band layers initialize their band parameter (a scalar weight, or a
    zero-initialized ``Dense`` kernel plus a zero bias for the adaptive pair)
    to zeros, so ``sigmoid(5 * 0) == 0.5`` and the scale is exactly the band
    MIDPOINT ``(1 - w) + w * 0.5``. Folding the constant in keeps the
    comparison sensitive only to the dtype of the STATISTICS.

    The result is widened back to float64 so the two references are directly
    comparable; the arithmetic above is what carries the dtype under test.
    """
    d = np.dtype(dtype).type
    x = x64.astype(dtype)
    if center:
        x = x - np.mean(x, axis=-1, keepdims=True)
    mean_square = np.mean(np.square(x), axis=-1, keepdims=True)
    rms = np.sqrt(mean_square + d(epsilon))
    if clamp_rms:
        rms = np.maximum(rms, d(epsilon))
    out = x / rms
    if band_width is not None:
        w = d(band_width)
        out = out * ((d(1.0) - w) + w * d(0.5))
    return np.asarray(out, dtype=np.float64)


# (id, layer factory, epsilon, center, clamp_rms, band_width)
CASES = [
    (
        "RMSNorm",
        lambda: RMSNorm(use_scale=False),
        1e-6,
        False,
        False,
        None,
    ),
    (
        "ZeroCenteredRMSNorm",
        lambda: ZeroCenteredRMSNorm(use_scale=False),
        1e-6,
        True,
        False,
        None,
    ),
    (
        "BandRMS",
        lambda: BandRMS(max_band_width=0.1),
        1e-7,
        False,
        True,
        0.1,
    ),
    (
        "ZeroCenteredBandRMSNorm",
        lambda: ZeroCenteredBandRMSNorm(max_band_width=0.1),
        1e-7,
        True,
        True,
        0.1,
    ),
    (
        "AdaptiveBandRMS",
        lambda: AdaptiveBandRMS(max_band_width=0.1),
        1e-7,
        False,
        True,
        0.1,
    ),
    (
        "ZeroCenteredAdaptiveBandRMS",
        lambda: ZeroCenteredAdaptiveBandRMS(max_band_width=0.1),
        1e-7,
        True,
        True,
        0.1,
    ),
]


@pytest.mark.parametrize(
    "name,factory,epsilon,center,clamp_rms,band_width",
    CASES,
    ids=[case[0] for case in CASES],
)
def test_statistics_follow_the_dtype_policy(
    float64_policy,
    name: str,
    factory,
    epsilon: float,
    center: bool,
    clamp_rms: bool,
    band_width: float | None,
) -> None:
    """The layer must match a float64 reference, not a float32-truncated one."""
    ref64 = _reference(
        PROBE,
        dtype="float64",
        epsilon=epsilon,
        center=center,
        clamp_rms=clamp_rms,
        band_width=band_width,
    )
    ref32 = _reference(
        PROBE,
        dtype="float32",
        epsilon=epsilon,
        center=center,
        clamp_rms=clamp_rms,
        band_width=band_width,
    )

    # The probe must actually discriminate, or the assertions below are vacuous.
    # The separation is scale-dependent: the mean-centering variants collapse to
    # exactly zero in float32 and so separate by O(1), while the plain variants
    # separate by only ~1.5e-8. Both are far above FLOAT64_ATOL, which is the
    # property that matters, so the vacuity floor is expressed relative to it.
    separation = float(np.max(np.abs(ref64 - ref32)))
    assert separation > 100.0 * FLOAT64_ATOL, (
        f"{name}: the probe does not separate float32 from float64 "
        f"(max|ref64-ref32|={separation!r}); the guard would be vacuous"
    )

    layer = factory()
    y = layer(keras.ops.convert_to_tensor(PROBE))

    assert keras.backend.standardize_dtype(y.dtype) == "float64", (
        f"{name}: the layer did not return the policy dtype"
    )

    out = np.asarray(keras.ops.convert_to_numpy(y), dtype=np.float64)
    dist64 = float(np.max(np.abs(out - ref64)))
    dist32 = float(np.max(np.abs(out - ref32)))

    assert dist64 < dist32, (
        f"{name}: the output is closer to the FLOAT32 reference "
        f"(max|delta|={dist32!r}) than to the float64 one ({dist64!r}) - the "
        f"internal statistics are still being computed in float32"
    )
    assert dist64 <= FLOAT64_ATOL, (
        f"{name}: max|layer - float64_reference| = {dist64!r} exceeds "
        f"{FLOAT64_ATOL!r} (float64 eps is ~2.2e-16); the statistics are not "
        f"being computed at the policy's precision"
    )


def test_the_float64_policy_fixture_does_not_leak(float64_policy) -> None:
    """Inside the fixture the policy really is float64.

    Paired with the module-scope restoration check below, this pins that the
    fixture both activates and releases the process-global state.
    """
    assert keras.backend.floatx() == "float64"
    assert keras.mixed_precision.global_policy().name == "float64"


def test_the_policy_is_restored_after_the_float64_cases() -> None:
    """No float64 policy may survive into any later test in this process.

    This test takes the fixture deliberately NOT as an argument: it runs after
    the parametrized cases above and would fail if any of them leaked.
    """
    assert keras.backend.floatx() == "float32"
    assert keras.mixed_precision.global_policy().name == "float32"
