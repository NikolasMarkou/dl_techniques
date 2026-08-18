"""THERA's central claim: a HEAT-EQUATION decay envelope on the field.

Why this file exists
--------------------
`ThermalActivation` computes ``sin(w0*x + phase) * exp(-(w0*norm)^2 * k * t)``.
The second factor -- the whole point of "Thermal Implicit Representation" -- is
what makes arbitrary-scale super-resolution alias-free: at a larger output
scale (larger heat time `t`) the high-frequency components must be attenuated
MORE than the low-frequency ones.

Every existing test in `test_thera/` passes `t = ops.ones(...)`. With a single
`t`, replacing the envelope by the constant `1.0` leaves all 64 of them green:
nothing measures a response to `t`, and nothing relates the attenuation to a
component's frequency.

Two claims here, in increasing sharpness:

1. Total field energy strictly decreases with `t`.
2. The attenuation is ORDERED BY FREQUENCY: the ratio
   ``amplitude(t2) / amplitude(t1)`` is monotonically decreasing in
   ``||components[:, j]||``. A constant envelope makes every ratio equal to 1;
   any envelope that is not a function of the frequency magnitude breaks the
   ordering.

MEASURED 2026-08-18 (hidden_dim = 16, seed 0, k = 0.0596, t = 0.001 -> 0.01):
component norms 16.41 .. 49.93, per-unit ratios 0.86538 down to 0.26227 in
exact frequency order (a 0.303 spread); total energy 436.43 -> 141.45.

A CORRECTION to the brief that asked for this test: it proposed `t = 0.1` vs
`t = 10.0`. At `t = 10` the envelope UNDERFLOWS -- measured output amplitude is
identically 0.0 for all 16 units, and the "energy decreased" claim degenerates
into "everything is zero". The exponent is `-norm^2 * k * t`, which at
`norm = 30` and `k = 0.06` is `-540` by `t = 10`. The times used here are
chosen so the envelope is in its informative range.
"""

from unittest import mock

import keras
import numpy as np
import pytest

from dl_techniques.layers.thera_heat_field import HeatField


HIDDEN = 16
BATCH, PIXELS = 1, 64
T_EARLY, T_LATE = 0.001, 0.01


@pytest.fixture(scope="module")
def field_and_inputs():
    keras.utils.set_random_seed(0)
    field = HeatField(hidden_dim=HIDDEN, out_dim=HIDDEN)
    rel_coords = (
        np.random.default_rng(0)
        .uniform(-1.0, 1.0, size=(BATCH, PIXELS, 2))
        .astype("float32")
    )
    phase = np.zeros((BATCH, PIXELS, HIDDEN), dtype="float32")
    # An identity per-pixel kernel, so output channel j IS hidden unit j and
    # each unit's amplitude can be read separately.
    kernel = np.tile(np.eye(HIDDEN, dtype="float32"), (BATCH, PIXELS, 1, 1))

    def run(t: float) -> np.ndarray:
        times = np.full((BATCH, 1), t, dtype="float32")
        out = field(
            keras.ops.convert_to_tensor(rel_coords),
            keras.ops.convert_to_tensor(phase),
            keras.ops.convert_to_tensor(kernel),
            keras.ops.convert_to_tensor(times),
            training=False,
        )
        return np.asarray(keras.ops.convert_to_numpy(out))

    run(T_EARLY)  # build, so `components` exists
    return field, run


def _unit_amplitudes(values: np.ndarray) -> np.ndarray:
    return np.sqrt((values ** 2).mean(axis=(0, 1)))


class TestHeatEnvelopeDecays:
    def test_energy_strictly_decreases_with_heat_time(self, field_and_inputs):
        _, run = field_and_inputs
        early = float((run(T_EARLY) ** 2).sum())
        late = float((run(T_LATE) ** 2).sum())
        # Measured 436.43 -> 141.45.
        assert late < 0.9 * early, (
            f"field energy did not fall with heat time: {early:.4f} at "
            f"t={T_EARLY} vs {late:.4f} at t={T_LATE}. A constant envelope "
            f"gives exactly equal energies."
        )

    def test_attenuation_is_ordered_by_frequency(self, field_and_inputs):
        field, run = field_and_inputs
        components = np.asarray(keras.ops.convert_to_numpy(field.components))
        norms = np.linalg.norm(components, axis=-2)
        assert norms.shape == (HIDDEN,)
        assert norms.max() > 2 * norms.min(), (
            f"the frequency components are too similar to order "
            f"({norms.min():.2f}..{norms.max():.2f}); the claim below would be "
            "untestable on this initialization"
        )

        ratios = _unit_amplitudes(run(T_LATE)) / np.maximum(
            _unit_amplitudes(run(T_EARLY)), 1e-30
        )
        order = np.argsort(norms)
        ordered = ratios[order]
        # Measured: 0.86538, 0.82189, 0.77627, ..., 0.26227 -- strictly
        # decreasing, exactly `exp(-norm^2 * k * dt)` in shape.
        assert np.all(np.diff(ordered) < 0.0), (
            "attenuation is not monotone in frequency; ratios in increasing "
            f"frequency order were {np.round(ordered, 5)}"
        )
        # Measured spread 0.303; the bar is 0.5, and a constant envelope
        # scores exactly 1.000.
        assert ordered[-1] < 0.5 * ordered[0], (
            f"the highest-frequency unit was attenuated to {ordered[-1]:.5f} "
            f"vs the lowest's {ordered[0]:.5f}: a ratio of "
            f"{ordered[-1] / ordered[0]:.3f}, where a constant envelope gives "
            f"exactly 1.000"
        )

    def test_both_claims_fail_when_the_envelope_is_the_constant_one(
        self, field_and_inputs
    ):
        """RED proof: the exact substitution the brief names.

        `ThermalActivation.call` is replaced by its oscillation term alone --
        the heat envelope forced to 1.0 -- which is what a model with the
        decay deleted computes. The whole point is that this substitution is
        invisible to every existing test in the package.
        """
        field, run = field_and_inputs
        thermal = field.thermal

        def _no_envelope(x, t, norm, k, phase, training=None):
            return keras.ops.sin(thermal.w0 * x + phase)

        with mock.patch.object(thermal, "call", _no_envelope):
            early = run(T_EARLY)
            late = run(T_LATE)
            # Claim 1 dies: the energies are now bit-identical.
            assert float((early ** 2).sum()) == float((late ** 2).sum())
            # Claim 2 dies: every ratio is exactly 1.
            ratios = _unit_amplitudes(late) / _unit_amplitudes(early)
            np.testing.assert_allclose(ratios, np.ones(HIDDEN), rtol=0, atol=0)
