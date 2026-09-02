"""The four bridge base processes reproduce their closed forms exactly.

``sigma`` / ``phi`` / ``C`` are not decoration: ``C`` divides the analytic score
targets and scales both direction-specific loss weightings, and ``sigma`` is the
diffusion coefficient of every Euler-Maruyama step. A wrong constant here is a
silently mistrained model, not a crash.

**Where the expected numbers come from.** Every ``GOLD_*`` constant below was
produced by a standalone stdlib-``math`` transcription of the formula table in
``findings/source-model-semantics.md`` section 3, run BEFORE ``sde.py`` existed.
The derivation is restated in the comment above each block so the constant can
be re-checked by hand. Nothing in this file calls the implementation to learn
what to expect; a golden value read off the code under test agrees with any
implementation, including a wrong one.

The one deliberate exception is :func:`test_C_matches_a_numerical_quadrature_of_sigma_squared`,
which is a *consistency* arm rather than a golden arm: for the driftless
variants the general rule is ``C(start, t_a, t_b) = int_start^min(t_a,t_b)
sigma(s)^2 ds``, so integrating the (separately golden-pinned) ``sigma``
numerically must reproduce the (separately golden-pinned) closed-form ``C``.
It cross-checks the two against each other and would catch an antiderivative
that is wrong in a way both golden sets happened to share.

Success criterion: SC-2.
"""

import math

import keras
import numpy as np
import pytest
from scipy.integrate import quad

from dl_techniques.models.vision_language.bit_diffusion.sde import (
    BridgeSDE,
    CosineDecayingVolatilitySDE,
    FlowMatchingODE,
    PeriodicVolatilitySDE,
    UniformVolatilitySDE,
)

# Golden values are float64 exact-ish; the implementation runs at float32 unless
# the input is float64, so the comparison tolerance is per-arm and stated at the
# call site rather than globally relaxed.
F64_TOL = 1e-12
F32_TOL = 1e-6


def _t(values, dtype="float64"):
    """A float tensor of the requested dtype, so the never-narrow floor is visible."""
    return keras.ops.convert_to_tensor(np.asarray(values, dtype=dtype))


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x), dtype="float64")


# =====================================================================
# 1. UniformVolatilitySDE, A == 0 -- plain Brownian motion
# =====================================================================
# sigma(t) = K everywhere; phi(start, end) = 1; C = K^2 * (min(t_a,t_b) - start).
#
#   K = 1.3, start = 0.2, t_a = 0.7, t_b = 0.45
#   min(t_a, t_b) = 0.45
#   C = 1.69 * (0.45 - 0.2) = 1.69 * 0.25 = 0.4225
GOLD_UNIF_K = 1.3
GOLD_UNIF_A0_C = 0.42250000000000004
# What the SAME call returns if `minimum(t_a, t_b)` degenerates to `t_a`
# (RED-proof injection ii): 1.69 * (0.7 - 0.2) = 0.845.
GOLD_UNIF_A0_C_IF_TA_ONLY = 0.845


def test_uniform_A0_sigma_is_the_constant_K():
    sde = UniformVolatilitySDE(A=0.0, K=GOLD_UNIF_K)
    t = _t([0.0, 0.13, 0.5, 0.99, 1.0])
    got = _np(sde.sigma(t))
    assert got.shape == (5,), f"sigma must be shape-preserving, got {got.shape}"
    np.testing.assert_allclose(got, np.full(5, GOLD_UNIF_K), rtol=0, atol=F64_TOL)


def test_uniform_A0_phi_is_one():
    sde = UniformVolatilitySDE(A=0.0, K=GOLD_UNIF_K)
    start = _t([0.0, 0.2, 0.75])
    end = _t([1.0, 0.7, 0.75])
    np.testing.assert_allclose(
        _np(sde.phi(start, end)), np.ones(3), rtol=0, atol=F64_TOL
    )


def test_uniform_A0_C_uses_the_minimum_of_both_times():
    """`min(t_a, t_b)` is only visible when t_a != t_b -- this arm makes it visible."""
    sde = UniformVolatilitySDE(A=0.0, K=GOLD_UNIF_K)
    start, t_a, t_b = _t([0.2]), _t([0.7]), _t([0.45])
    got = _np(sde.C(start, t_a, t_b))[0]
    assert got == pytest.approx(GOLD_UNIF_A0_C, rel=0, abs=F64_TOL), (
        f"C(0.2, 0.7, 0.45) must be K^2*(min-start) = {GOLD_UNIF_A0_C}, got {got}"
    )
    # Anti-vacuity: the t_a-only value is a DIFFERENT number, so the arm above
    # genuinely discriminates between `min(t_a,t_b)` and `t_a`.
    assert not math.isclose(GOLD_UNIF_A0_C, GOLD_UNIF_A0_C_IF_TA_ONLY)
    assert got != pytest.approx(GOLD_UNIF_A0_C_IF_TA_ONLY, rel=0, abs=1e-3)


def test_uniform_A0_C_is_symmetric_in_its_two_times():
    sde = UniformVolatilitySDE(A=0.0, K=GOLD_UNIF_K)
    forward = _np(sde.C(_t([0.2]), _t([0.7]), _t([0.45])))
    swapped = _np(sde.C(_t([0.2]), _t([0.45]), _t([0.7])))
    np.testing.assert_allclose(forward, swapped, rtol=0, atol=F64_TOL)


# =====================================================================
# 2. UniformVolatilitySDE, A != 0 -- the Ornstein-Uhlenbeck branch
# =====================================================================
# phi(start, end) = exp(-A*(end - start))
# C = K^2 * exp(-A*(t_a+t_b)) * (exp(2A*min) - exp(2A*start)) / (2A)
#
#   A = 0.75, K = 1.3, start = 0.2, t_a = 0.7, t_b = 0.45, min = 0.45
#   phi(0.2, 0.7) = exp(-0.375)                       = 0.6872892787909722
#   phi(0.0, 1.0) = exp(-0.75)                        = 0.4723665527410147
#   C = 1.69 * exp(-0.8625) * (exp(0.675) - exp(0.3)) / 1.5
#                                                     = 0.29208415728641124
GOLD_OU_A = 0.75
GOLD_OU_PHI_02_07 = 0.6872892787909722
GOLD_OU_PHI_00_10 = 0.4723665527410147
GOLD_OU_C = 0.29208415728641124
# The value the A == 0 branch would give at the SAME arguments (0.4225) and the
# value a `t_a`-only min would give (0.7170640984841317). Both differ from
# GOLD_OU_C, which is what makes this arm non-vacuous for the OU branch.
GOLD_OU_C_IF_A0_BRANCH = 0.42250000000000004
GOLD_OU_C_IF_TA_ONLY = 0.7170640984841317


def test_uniform_OU_phi_decays_exponentially():
    sde = UniformVolatilitySDE(A=GOLD_OU_A, K=GOLD_UNIF_K)
    got = _np(sde.phi(_t([0.2, 0.0]), _t([0.7, 1.0])))
    np.testing.assert_allclose(
        got, [GOLD_OU_PHI_02_07, GOLD_OU_PHI_00_10], rtol=0, atol=F64_TOL
    )
    # Anti-vacuity for the A != 0 branch: phi is NOT the constant 1 here.
    assert abs(got[0] - 1.0) > 0.3


def test_uniform_OU_C_is_the_ornstein_uhlenbeck_covariance():
    sde = UniformVolatilitySDE(A=GOLD_OU_A, K=GOLD_UNIF_K)
    got = _np(sde.C(_t([0.2]), _t([0.7]), _t([0.45])))[0]
    assert got == pytest.approx(GOLD_OU_C, rel=0, abs=F64_TOL), (
        f"OU C must be {GOLD_OU_C}, got {got}"
    )


def test_the_OU_branch_is_genuinely_exercised_not_the_A0_one():
    """An A=0-only test is vacuous for the OU branch; this arm forces the split.

    If `UniformVolatilitySDE.C` ever collapsed to the A == 0 expression, this
    arm fails while every A == 0 arm above stays green.
    """
    ou = UniformVolatilitySDE(A=GOLD_OU_A, K=GOLD_UNIF_K)
    brownian = UniformVolatilitySDE(A=0.0, K=GOLD_UNIF_K)
    args = (_t([0.2]), _t([0.7]), _t([0.45]))
    ou_value = _np(ou.C(*args))[0]
    brownian_value = _np(brownian.C(*args))[0]
    assert brownian_value == pytest.approx(GOLD_OU_C_IF_A0_BRANCH, rel=0, abs=F64_TOL)
    assert abs(ou_value - brownian_value) > 0.1, (
        "the A != 0 branch returned the A == 0 value; the OU code path is dead"
    )
    assert ou_value != pytest.approx(GOLD_OU_C_IF_TA_ONLY, rel=0, abs=1e-3)
    # phi likewise must differ between the two branches.
    assert abs(_np(ou.phi(_t([0.2]), _t([0.7])))[0]
               - _np(brownian.phi(_t([0.2]), _t([0.7])))[0]) > 0.3


def test_uniform_OU_C_at_a_second_A_reproduces_its_own_golden():
    # A = -0.4 (mean-EXPANDING; the formula is agnostic to the sign of A and the
    # 1/(2A) denominator must not be assumed positive):
    #   C = 1.69 * exp(0.4*1.15) * (exp(-0.8*0.45) - exp(-0.8*0.2)) / (-0.8)
    a = -0.4
    expected = (
        1.69
        * math.exp(-a * (0.7 + 0.45))
        * (math.exp(2 * a * 0.45) - math.exp(2 * a * 0.2))
        / (2 * a)
    )
    sde = UniformVolatilitySDE(A=a, K=GOLD_UNIF_K)
    got = _np(sde.C(_t([0.2]), _t([0.7]), _t([0.45])))[0]
    assert got == pytest.approx(expected, rel=0, abs=F64_TOL)
    assert got > 0.0, "a covariance must be positive whatever the sign of A"


# =====================================================================
# 3. PeriodicVolatilitySDE
# =====================================================================
# sigma(t) = alpha/2 * (1 - cos(2*pi*k*t)) + eps
# phi = 1 (driftless)
# C(start,t_a,t_b) = F(min(t_a,t_b)) - F(start), with the three-term antiderivative
#   F(s) = (3a^2/8 + a*e + e^2) * s
#          - a(a + 2e)/(4*pi*k) * sin(2*pi*k*s)
#          + a^2/(32*pi*k)      * sin(4*pi*k*s)
# (derived by expanding sigma^2 with cos^2 = (1+cos(2x))/2 and integrating term by term)
#
# Shipped paper parameters: alpha = 0.95, k = 1.0, eps = 0.05
#   sigma(0.00) = 0.05                  sigma(0.25) = 0.5249999999999999
#   sigma(0.37) = 0.8501598753161271    sigma(0.50) = 1.0
#   sigma(1.00) = 0.05
#   C(0.2, 0.7, 0.45) = 0.137520038599771
#   C(0.0, 1.0, 1.0)  = 3*0.95^2/8 + 0.95*0.05 + 0.05^2 = 0.3884375  (sines vanish)
# Second parameter set: alpha = 0.60, k = 2.5, eps = 0.11
#   sigma(0.13) = 0.5461971499218641    sigma(0.62) = 0.6953169548885461
#   C(0.1, 0.83, 0.55) = 0.09904956122368305
PER_SHIP = dict(alpha=0.95, k=1.0, eps=0.05)
PER_ALT = dict(alpha=0.60, k=2.5, eps=0.11)

GOLD_PER_SHIP_SIGMA = {
    0.0: 0.05,
    0.25: 0.5249999999999999,
    0.37: 0.8501598753161271,
    0.5: 1.0,
    1.0: 0.05,
}
GOLD_PER_ALT_SIGMA = {0.13: 0.5461971499218641, 0.62: 0.6953169548885461}
GOLD_PER_SHIP_C = 0.137520038599771
GOLD_PER_SHIP_C_FULL = 0.3884375
GOLD_PER_ALT_C = 0.09904956122368305
# With the antiderivative's `second_term` sign flipped (RED-proof injection i)
# the same call returns 0.035591734434747835 -- a different number, so the arm
# below discriminates.
GOLD_PER_SHIP_C_IF_SECOND_TERM_FLIPPED = 0.035591734434747835


def test_periodic_sigma_matches_the_shipped_parameter_goldens():
    sde = PeriodicVolatilitySDE(**PER_SHIP)
    times = sorted(GOLD_PER_SHIP_SIGMA)
    got = _np(sde.sigma(_t(times)))
    np.testing.assert_allclose(
        got, [GOLD_PER_SHIP_SIGMA[t] for t in times], rtol=0, atol=F64_TOL
    )


def test_periodic_sigma_matches_a_second_parameter_set():
    sde = PeriodicVolatilitySDE(**PER_ALT)
    times = sorted(GOLD_PER_ALT_SIGMA)
    got = _np(sde.sigma(_t(times)))
    np.testing.assert_allclose(
        got, [GOLD_PER_ALT_SIGMA[t] for t in times], rtol=0, atol=F64_TOL
    )


def test_periodic_phi_is_identically_one():
    sde = PeriodicVolatilitySDE(**PER_SHIP)
    np.testing.assert_allclose(
        _np(sde.phi(_t([0.0, 0.2, 0.9]), _t([1.0, 0.7, 0.9]))),
        np.ones(3),
        rtol=0,
        atol=F64_TOL,
    )


def test_periodic_C_matches_the_three_term_antiderivative():
    sde = PeriodicVolatilitySDE(**PER_SHIP)
    got = _np(sde.C(_t([0.2]), _t([0.7]), _t([0.45])))[0]
    assert got == pytest.approx(GOLD_PER_SHIP_C, rel=0, abs=F64_TOL), (
        f"periodic C(0.2, 0.7, 0.45) must be {GOLD_PER_SHIP_C}, got {got}"
    )
    # The sign of the middle term is the whole content of this arm: with it
    # flipped the value is a materially different number.
    assert got != pytest.approx(
        GOLD_PER_SHIP_C_IF_SECOND_TERM_FLIPPED, rel=0, abs=1e-4
    )


def test_periodic_C_over_the_whole_interval_is_the_mean_square_of_sigma():
    """Over [0, 1] with k = 1 both sines vanish, so C collapses to 3a^2/8+ae+e^2."""
    sde = PeriodicVolatilitySDE(**PER_SHIP)
    got = _np(sde.C(_t([0.0]), _t([1.0]), _t([1.0])))[0]
    assert got == pytest.approx(GOLD_PER_SHIP_C_FULL, rel=0, abs=F64_TOL)


def test_periodic_C_matches_a_second_parameter_set():
    sde = PeriodicVolatilitySDE(**PER_ALT)
    got = _np(sde.C(_t([0.1]), _t([0.83]), _t([0.55])))[0]
    assert got == pytest.approx(GOLD_PER_ALT_C, rel=0, abs=F64_TOL)


# =====================================================================
# 4. CosineDecayingVolatilitySDE -- Periodic at k = 0.5 PLUS a t-1 shift
# =====================================================================
# sigma(t) = Periodic(k=0.5).sigma(t - 1) = alpha/2*(1 + cos(pi*t)) + eps
# C(start,t_a,t_b) = Periodic(k=0.5).C(start-1, t_a-1, t_b-1)
#
# alpha = 0.95, eps = 0.05:
#   sigma(0.00) = 1.0                  vs 0.05                 UNSHIFTED
#   sigma(0.25) = 0.8608757210636101   vs 0.1891242789363899   UNSHIFTED
#   sigma(0.50) = 0.5249999999999999   vs 0.5249999999999999   <-- BLIND to the shift
#   sigma(1.00) = 0.05                 vs 1.0                  UNSHIFTED
#   C(0.2, 0.7, 0.45) = 0.14906920604974394  vs 0.022094332293881757 UNSHIFTED
#
# t = 0.5 is a half-period of the k = 0.5 cosine, so a shift-by-1 is EXACTLY
# invisible there. A test that sampled only t = 0.5 would pass with the shift
# deleted; the arms below therefore pin t = 0 and t = 0.25 as well.
COS_PARAMS = dict(alpha=0.95, eps=0.05)
GOLD_COS_SIGMA = {0.0: 1.0, 0.25: 0.8608757210636101, 0.5: 0.5249999999999999, 1.0: 0.05}
GOLD_COS_SIGMA_UNSHIFTED = {
    0.0: 0.05,
    0.25: 0.1891242789363899,
    0.5: 0.5249999999999999,
    1.0: 1.0,
}
GOLD_COS_C = 0.14906920604974394
GOLD_COS_C_UNSHIFTED = 0.022094332293881757


def test_cosine_decaying_sigma_matches_its_goldens():
    sde = CosineDecayingVolatilitySDE(**COS_PARAMS)
    times = sorted(GOLD_COS_SIGMA)
    got = _np(sde.sigma(_t(times)))
    np.testing.assert_allclose(
        got, [GOLD_COS_SIGMA[t] for t in times], rtol=0, atol=F64_TOL
    )


def test_cosine_decaying_sigma_carries_the_minus_one_time_shift():
    """The shift is the whole difference from Periodic(k=0.5); pin it at t=0 and t=1."""
    sde = CosineDecayingVolatilitySDE(**COS_PARAMS)
    got = _np(sde.sigma(_t([0.0, 0.25, 1.0])))
    unshifted = [GOLD_COS_SIGMA_UNSHIFTED[t] for t in (0.0, 0.25, 1.0)]
    for value, wrong, t in zip(got, unshifted, (0.0, 0.25, 1.0)):
        assert abs(value - wrong) > 0.1, (
            f"sigma({t}) = {value} equals the UNSHIFTED Periodic(k=0.5) value "
            f"{wrong}; the `t - 1` shift is missing"
        )
    # Direction: volatility DECAYS from alpha+eps at t=0 to eps at t=1.
    assert got[0] == pytest.approx(COS_PARAMS["alpha"] + COS_PARAMS["eps"], abs=F64_TOL)
    assert got[2] == pytest.approx(COS_PARAMS["eps"], abs=F64_TOL)
    assert got[0] > got[1] > got[2]


def test_cosine_decaying_C_carries_the_shift_too():
    """A k=0.5-only port shifts sigma and forgets C; that leaves sigma green."""
    sde = CosineDecayingVolatilitySDE(**COS_PARAMS)
    got = _np(sde.C(_t([0.2]), _t([0.7]), _t([0.45])))[0]
    assert got == pytest.approx(GOLD_COS_C, rel=0, abs=F64_TOL), (
        f"cosine-decaying C(0.2, 0.7, 0.45) must be {GOLD_COS_C}, got {got}"
    )
    assert abs(got - GOLD_COS_C_UNSHIFTED) > 0.1, (
        "C returned the unshifted Periodic(k=0.5) value; the `-1` shift is "
        "missing from C even if sigma has it"
    )


def test_cosine_decaying_k_is_one_half():
    """The other half of the definition: it IS a Periodic instance at k = 0.5."""
    sde = CosineDecayingVolatilitySDE(**COS_PARAMS)
    assert isinstance(sde, PeriodicVolatilitySDE)
    assert sde.k == 0.5
    assert sde.A == 0.0


# =====================================================================
# 5. FlowMatchingODE -- the RAISING contract
# =====================================================================
# It is a deterministic rectified-flow transport, not a diffusion. Upstream
# raises on all three quantities on purpose; returning 0.0 (or anything) would
# make the bridge-process math silently produce a degenerate score.
def test_flow_matching_sigma_raises_not_implemented():
    sde = FlowMatchingODE()
    with pytest.raises(NotImplementedError) as excinfo:
        sde.sigma(_t([0.3]))
    assert "sigma" in str(excinfo.value), (
        f"the message must name the missing quantity, got {excinfo.value!r}"
    )


def test_flow_matching_phi_raises_not_implemented():
    sde = FlowMatchingODE()
    with pytest.raises(NotImplementedError) as excinfo:
        sde.phi(_t([0.0]), _t([1.0]))
    assert "phi" in str(excinfo.value)


def test_flow_matching_C_raises_not_implemented():
    sde = FlowMatchingODE()
    with pytest.raises(NotImplementedError) as excinfo:
        sde.C(_t([0.0]), _t([0.5]), _t([1.0]))
    assert "C" in str(excinfo.value)


def test_flow_matching_returns_nothing_at_all():
    """A `return 0.0` instead of a raise is the failure this arm exists for.

    `pytest.raises` alone can be satisfied by an implementation that raises for
    some inputs and returns for others, so probe a spread of shapes and dtypes.
    """
    sde = FlowMatchingODE()
    probes = [
        _t([0.0]),
        _t([0.5, 0.5], dtype="float32"),
        _t(np.linspace(0.0, 1.0, 7)),
    ]
    for probe in probes:
        for call in (
            lambda p: sde.sigma(p),
            lambda p: sde.phi(p, p),
            lambda p: sde.C(p, p, p),
        ):
            with pytest.raises(NotImplementedError):
                call(probe)


def test_the_base_class_raises_for_every_quantity():
    base = BridgeSDE(A=0.0)
    for call in (
        lambda: base.sigma(_t([0.5])),
        lambda: base.phi(_t([0.0]), _t([1.0])),
        lambda: base.C(_t([0.0]), _t([0.5]), _t([1.0])),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_dX_t_and_simulate_refuse_to_run_without_a_network():
    """The sampler landed at step 8; what stays pinned is D-009's consequence.

    These classes deliberately do NOT hold a ``score_network`` the way upstream
    does, so both entry points must fail loudly rather than quietly sampling the
    base process with a zero score -- which would be finite, plausible and
    completely untrained.

    The behaviour of the sampler itself is pinned by
    ``test_the_sampler_skips_the_first_ode_step.py``.
    """
    sde = UniformVolatilitySDE(A=0.0, K=1.0)
    with pytest.raises(ValueError, match="score_network"):
        sde.dX_t(x_t=None, t=None, x_cond=None, y=None, dt=0.1)
    with pytest.raises(ValueError, match="score_network"):
        sde.simulate(x_start=None, num_steps=2, y=0)


# =====================================================================
# 6. Numerical cross-check: C is the integral of sigma^2 (driftless variants)
# =====================================================================
QUADRATURE_CASES = [
    ("uniform_A0", UniformVolatilitySDE(A=0.0, K=1.3), 0.2, 0.7, 0.45),
    ("periodic_ship", PeriodicVolatilitySDE(**PER_SHIP), 0.2, 0.7, 0.45),
    ("periodic_ship_full", PeriodicVolatilitySDE(**PER_SHIP), 0.0, 1.0, 1.0),
    ("periodic_alt", PeriodicVolatilitySDE(**PER_ALT), 0.1, 0.83, 0.55),
    ("cosine_decay", CosineDecayingVolatilitySDE(**COS_PARAMS), 0.2, 0.7, 0.45),
    ("cosine_decay_full", CosineDecayingVolatilitySDE(**COS_PARAMS), 0.0, 1.0, 1.0),
]


@pytest.mark.parametrize(
    "name,sde,start,t_a,t_b", QUADRATURE_CASES, ids=[c[0] for c in QUADRATURE_CASES]
)
def test_C_matches_a_numerical_quadrature_of_sigma_squared(name, sde, start, t_a, t_b):
    """For a driftless process C(start,t_a,t_b) = int_start^min sigma(s)^2 ds.

    If this disagrees with the closed form, say so loudly -- do NOT loosen the
    tolerance. `quad` on a smooth bounded integrand over a unit interval is good
    to ~1e-12, so a disagreement above 1e-9 is a real antiderivative error.
    """
    assert sde.A == 0.0, f"{name} is not driftless; the identity does not apply"
    # phi == 1 is the other half of the driftless premise; assert it rather than
    # assuming it.
    np.testing.assert_allclose(
        _np(sde.phi(_t([start]), _t([t_a]))), [1.0], rtol=0, atol=F64_TOL
    )

    def integrand(s):
        return float(_np(sde.sigma(_t([s])))[0]) ** 2

    upper = min(t_a, t_b)
    numeric, abserr = quad(integrand, start, upper, limit=400, epsabs=1e-13)
    closed = _np(sde.C(_t([start]), _t([t_a]), _t([t_b])))[0]
    assert abserr < 1e-9, f"{name}: the quadrature itself did not converge ({abserr})"
    assert closed == pytest.approx(numeric, rel=0, abs=1e-9), (
        f"{name}: closed form {closed!r} disagrees with the quadrature "
        f"{numeric!r} (delta {closed - numeric!r}); the antiderivative is wrong"
    )


# =====================================================================
# 7. Never-narrow dtype floor
# =====================================================================
NEVER_NARROW_SDES = [
    ("uniform_A0", UniformVolatilitySDE(A=0.0, K=1.3)),
    ("uniform_OU", UniformVolatilitySDE(A=0.75, K=1.3)),
    ("periodic", PeriodicVolatilitySDE(**PER_SHIP)),
    ("cosine_decay", CosineDecayingVolatilitySDE(**COS_PARAMS)),
]


@pytest.mark.parametrize(
    "name,sde", NEVER_NARROW_SDES, ids=[c[0] for c in NEVER_NARROW_SDES]
)
@pytest.mark.parametrize("in_dtype,expected", [
    ("float16", "float32"),
    ("float32", "float32"),
    ("float64", "float64"),
])
def test_the_closed_forms_run_at_max_input_float32(name, sde, in_dtype, expected):
    """`C` is O(1e-4) near the endpoints; fp16 there is an accuracy hazard."""
    t = _t([0.2, 0.45, 0.7], dtype=in_dtype)
    for out in (sde.sigma(t), sde.phi(t, t), sde.C(t[:1], t[1:2], t[2:3])):
        got = keras.backend.standardize_dtype(out.dtype)
        assert got == expected, (
            f"{name}: a {in_dtype} input produced {got}; the floor is "
            f"max(input, float32) and it must never NARROW a float64 input"
        )


@pytest.mark.parametrize(
    "name,sde", NEVER_NARROW_SDES, ids=[c[0] for c in NEVER_NARROW_SDES]
)
def test_float32_inputs_still_reach_the_goldens(name, sde):
    """The float64 goldens must survive a float32 input at float32 tolerance."""
    args64 = (_t([0.2]), _t([0.7]), _t([0.45]))
    args32 = tuple(_t([float(_np(a)[0])], dtype="float32") for a in args64)
    np.testing.assert_allclose(
        _np(sde.C(*args32)), _np(sde.C(*args64)), rtol=0, atol=F32_TOL
    )


# =====================================================================
# 8. Serialization surface
# =====================================================================
ROUND_TRIP_SDES = [
    UniformVolatilitySDE(A=0.75, K=1.3),
    PeriodicVolatilitySDE(**PER_SHIP),
    CosineDecayingVolatilitySDE(**COS_PARAMS),
    FlowMatchingODE(force_unconditional=True),
]


@pytest.mark.parametrize(
    "sde", ROUND_TRIP_SDES, ids=[type(s).__name__ for s in ROUND_TRIP_SDES]
)
def test_config_round_trip_preserves_every_knob(sde):
    restored = type(sde).from_config(sde.get_config())
    assert restored.get_config() == sde.get_config()
    if not isinstance(restored, FlowMatchingODE):
        np.testing.assert_allclose(
            _np(restored.C(_t([0.2]), _t([0.7]), _t([0.45]))),
            _np(sde.C(_t([0.2]), _t([0.7]), _t([0.45]))),
            rtol=0,
            atol=F64_TOL,
        )
