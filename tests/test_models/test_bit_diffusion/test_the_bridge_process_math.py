"""The bridge posterior, both score targets, both weightings and the two time samplers.

This is the data-pipeline math. Nothing here is a shape check: every quantity
either divides by ``C`` (which is exactly zero at an endpoint) or multiplies the
loss, so a wrong sign, a swapped direction or a dropped ``phi`` factor is a
silently mistrained model with no crash and no shape symptom.

**Where the expected values come from.** Three independent oracles, deliberately
none of which is the implementation's own algebra:

1. *Closed-form arms* recompute the target expression in the test body from
   ``sde.phi`` / ``sde.C`` (pinned to hand-derived goldens by
   ``test_the_sde_closed_forms.py`` in step 3) rather than calling the function
   under test to learn what it should return.
2. *A central finite difference in float64* of the corresponding Gaussian
   log-density. Both score targets are by definition ``grad_{x_t} log p(...)``,
   so differentiating the density numerically checks the differentiation
   algebra without re-deriving it. The log-density is built from ``phi``/``C``
   only.
3. *Statistical arms* draw a large batch from :func:`sample_bridge_x_t` and
   compare the empirical mean and standard deviation against the analytic
   moments, which is the only thing that can see a sampler that computes the
   right variance and then ignores it.

The asymmetry between the two weightings is the single most confusable part of
this port, so it gets three arms: a value arm per weighting against its own
closed form, an arm asserting they differ at a deliberately chosen ``t``, and an
arm documenting the ``t`` where they genuinely *coincide* (so nobody picks that
one by accident and reports a vacuous pass).

Success criteria: the Problem Statement's ``t = 0`` / ``t = 1`` and
``C(...) = 0`` boundary cases (plan.md, "Boundary / edge cases").
"""

import numpy as np
import keras
import pytest

from dl_techniques.models.vision_language.bit_diffusion.config import TIME_EPS
from dl_techniques.models.vision_language.bit_diffusion.sde import (
    CosineDecayingVolatilitySDE,
    FlowMatchingODE,
    PeriodicVolatilitySDE,
    UniformVolatilitySDE,
)
from dl_techniques.models.vision_language.bit_diffusion.bridge_process import (
    LOGIT_NORMAL_P_MEAN,
    LOGIT_NORMAL_P_STD,
    bridge_posterior_moments,
    dsm_weight_forward,
    dsm_weight_reverse,
    flow_matching_interpolant,
    flow_matching_target,
    sample_bridge_x_t,
    sample_timesteps_logit_normal,
    sample_timesteps_uniform,
    score_target_forward,
    score_target_reverse,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x), dtype="float64")


def _t(values, dtype="float32"):
    return keras.ops.convert_to_tensor(np.asarray(values, dtype=dtype))


#: The three genuinely stochastic variants. ``FlowMatchingODE`` has no ``C`` at
#: all and is covered by its own raising arms.
STOCHASTIC_SDES = [
    ("uniform_A0", UniformVolatilitySDE(A=0.0, K=1.0)),
    ("uniform_OU", UniformVolatilitySDE(A=1.5, K=0.8)),
    ("periodic", PeriodicVolatilitySDE(alpha=0.95, k=1.0, eps=0.05)),
    ("cosine_decay", CosineDecayingVolatilitySDE(alpha=0.95, eps=0.05)),
]


def _closed_form_moments(sde, x_0, x_1, t):
    """Mean and UNCLAMPED variance, recomputed here from ``phi``/``C`` alone.

    This is the reference the implementation is checked against; it deliberately
    keeps the raw (possibly slightly negative) variance so the clamp arms can
    see what the implementation clamped away.
    """
    t = np.asarray(t, dtype="float64")
    zeros = np.zeros_like(t)
    ones = np.ones_like(t)
    phi_0t = _np(sde.phi(zeros, t))
    phi_01 = _np(sde.phi(zeros, ones))
    c_tt = _np(sde.C(zeros, t, t))
    c_t1 = _np(sde.C(zeros, t, ones))
    c_11 = _np(sde.C(zeros, ones, ones))
    gain = c_t1 / c_11
    exp = lambda s: s.reshape((-1,) + (1,) * (x_0.ndim - 1))  # noqa: E731
    mean = exp(phi_0t) * x_0 + exp(gain) * (x_1 - exp(phi_01) * x_0)
    var = c_tt - c_t1 ** 2 / c_11
    return mean, var


# =====================================================================
# 1. Endpoint pinning -- the bridge is anchored at x_0 and x_1
# =====================================================================


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_the_posterior_mean_pins_x_0_at_the_low_endpoint(name, sde):
    """As ``t -> 0`` the posterior mean collapses onto ``x_0``."""
    rng = np.random.default_rng(0)
    x_0 = rng.standard_normal((4, 3)).astype("float64")
    x_1 = rng.standard_normal((4, 3)).astype("float64")
    t = np.full((4,), TIME_EPS)

    mean, var = bridge_posterior_moments(sde, _t(x_0, "float64"), _t(x_1, "float64"), _t(t, "float64"))
    np.testing.assert_allclose(_np(mean), x_0, atol=2e-3)
    # The variance has not merely shrunk, it is negligible against the O(1) data.
    assert np.all(_np(var) < 1e-3), f"{name}: var at TIME_EPS = {_np(var)}"


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_the_posterior_mean_pins_x_1_at_the_high_endpoint(name, sde):
    """As ``t -> 1`` the posterior mean collapses onto ``x_1``."""
    rng = np.random.default_rng(1)
    x_0 = rng.standard_normal((4, 3)).astype("float64")
    x_1 = rng.standard_normal((4, 3)).astype("float64")
    t = np.full((4,), 1.0 - TIME_EPS)

    mean, var = bridge_posterior_moments(sde, _t(x_0, "float64"), _t(x_1, "float64"), _t(t, "float64"))
    np.testing.assert_allclose(_np(mean), x_1, atol=2e-3)
    assert np.all(_np(var) < 1e-3), f"{name}: var at 1-TIME_EPS = {_np(var)}"


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_the_posterior_variance_is_exactly_zero_at_both_raw_endpoints(name, sde):
    """``t = 0`` and ``t = 1`` are the anchored ends: no residual randomness."""
    x_0 = np.zeros((2, 3), dtype="float64")
    x_1 = np.ones((2, 3), dtype="float64")
    _, var = bridge_posterior_moments(
        sde, _t(x_0, "float64"), _t(x_1, "float64"), _t([0.0, 1.0], "float64")
    )
    np.testing.assert_allclose(_np(var), [0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_the_variance_shrinks_monotonically_towards_each_endpoint(name, sde):
    """An anti-vacuity companion: the variance is not simply small everywhere."""
    x_0 = np.zeros((1, 2), dtype="float64")
    x_1 = np.zeros((1, 2), dtype="float64")
    ladder = np.array([1e-6, 1e-4, 1e-2, 0.5, 1 - 1e-2, 1 - 1e-4, 1 - 1e-6])
    _, var = bridge_posterior_moments(
        sde,
        _t(np.repeat(x_0, len(ladder), 0), "float64"),
        _t(np.repeat(x_1, len(ladder), 0), "float64"),
        _t(ladder, "float64"),
    )
    v = _np(var)
    assert np.all(np.diff(v[:4]) > 0), f"{name}: rising half not monotone: {v}"
    assert np.all(np.diff(v[3:]) < 0), f"{name}: falling half not monotone: {v}"
    assert v[3] > 1e3 * max(v[0], v[-1]), f"{name}: midpoint variance not dominant: {v}"


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_the_sampled_x_t_concentrates_on_the_endpoints(name, sde):
    """The *sample*, not just the mean, sits on ``x_0`` / ``x_1`` at the ends."""
    rng = np.random.default_rng(2)
    x_0 = rng.standard_normal((256, 3)).astype("float32")
    x_1 = rng.standard_normal((256, 3)).astype("float32")

    for t_val, anchor in ((TIME_EPS, x_0), (1.0 - TIME_EPS, x_1)):
        t = np.full((256,), t_val, dtype="float32")
        x_t = _np(sample_bridge_x_t(sde, _t(x_0), _t(x_1), _t(t), seed=7))
        _, var = bridge_posterior_moments(sde, _t(x_0), _t(x_1), _t(t))
        sigma = float(np.sqrt(_np(var).max()))
        gap = np.abs(x_t - anchor).max()
        assert gap < 8.0 * sigma + 2e-3, (
            f"{name} at t={t_val}: max|x_t - anchor| = {gap:.3e}, sigma = {sigma:.3e}"
        )


# =====================================================================
# 2. Posterior variance non-negativity on a dense grid
# =====================================================================


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_the_posterior_variance_is_non_negative_on_a_dense_grid(name, sde):
    """Sweep ``t`` finely; the shipped (clamped) variance never goes below zero."""
    grid = np.linspace(0.0, 1.0, 2001, dtype="float32")
    x = np.zeros((grid.size, 2), dtype="float32")
    _, var = bridge_posterior_moments(sde, _t(x), _t(x), _t(grid))
    v = _np(var)
    assert np.all(v >= 0.0), f"{name}: {int((v < 0).sum())} negative variances, min={v.min():.3e}"
    assert np.all(np.isfinite(np.sqrt(v))), f"{name}: non-finite sqrt of the variance"


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_the_sampled_x_t_is_finite_on_a_dense_grid(name, sde):
    grid = np.linspace(0.0, 1.0, 1001, dtype="float32")
    x_0 = np.ones((grid.size, 2), dtype="float32")
    x_1 = -np.ones((grid.size, 2), dtype="float32")
    x_t = _np(sample_bridge_x_t(sde, _t(x_0), _t(x_1), _t(grid), seed=11))
    assert np.all(np.isfinite(x_t))


def test_the_flow_matching_variant_has_no_posterior_at_all():
    """``FlowMatchingODE`` has no ``C``; the bridge posterior must raise, not guess."""
    x = np.zeros((2, 3), dtype="float32")
    with pytest.raises(NotImplementedError):
        bridge_posterior_moments(FlowMatchingODE(), _t(x), _t(x), _t([0.3, 0.7]))
    with pytest.raises(NotImplementedError):
        sample_bridge_x_t(FlowMatchingODE(), _t(x), _t(x), _t([0.3, 0.7]), seed=0)


# =====================================================================
# 3. The clamp is LOAD-BEARING -- measured float32 negatives
# =====================================================================
#
# Found by `probes/step5_variance_negativity_search.py` (output staged beside it).
# Both cases below are real float32 round-off on ordinary constructor arguments,
# not manufactured by absurd parameters:
#
#   UniformVolatilitySDE(A=5.0, K=1.0)            at t = 1.0      -> -7.450581e-09
#   PeriodicVolatilitySDE(0.95, k=3.0, eps=1e-3)  at t = 0.99875  -> -2.980232e-08
#
# HONEST SCOPE, recorded by `test_the_shipped_defaults_show_no_float32_negative`
# below: at the three SHIPPED default parameter sets the unclamped expression
# never went negative over 2,010,002 float32 samples. The clamp is DEMONSTRATED
# for the two parameterisations above and DEFENSIVE at the defaults.

CLAMP_CASES = [
    ("uniform_OU_A5_at_t_exactly_one", UniformVolatilitySDE(A=5.0, K=1.0), 1.0),
    ("periodic_k3_eps1e-3_near_one", PeriodicVolatilitySDE(alpha=0.95, k=3.0, eps=1e-3), 0.99875),
]


@pytest.mark.parametrize("name,sde,t_val", CLAMP_CASES)
def test_the_unclamped_float32_variance_really_does_go_negative(name, sde, t_val):
    """Step 1 of the clamp proof: the negative exists, computed without the module."""
    t = _t([t_val], "float32")
    zeros, ones = keras.ops.zeros_like(t), keras.ops.ones_like(t)
    c_tt = sde.C(zeros, t, t)
    c_t1 = sde.C(zeros, t, ones)
    c_11 = sde.C(zeros, ones, ones)
    raw = _np(c_tt - c_t1 * c_t1 / c_11)
    assert keras.backend.standardize_dtype(c_tt.dtype) == "float32"
    assert raw[0] < 0.0, f"{name}: expected a negative float32 variance, got {raw[0]:.6e}"


@pytest.mark.parametrize("name,sde,t_val", CLAMP_CASES)
def test_the_module_clamps_that_negative_to_zero_and_stays_finite(name, sde, t_val):
    """Step 2: the same input through the module yields exactly 0 and a finite sample."""
    x_0 = np.array([[1.0, -2.0]], dtype="float32")
    x_1 = np.array([[0.5, 3.0]], dtype="float32")
    t = _t([t_val], "float32")
    _, var = bridge_posterior_moments(sde, _t(x_0), _t(x_1), t)
    assert _np(var)[0] == 0.0
    x_t = _np(sample_bridge_x_t(sde, _t(x_0), _t(x_1), t, seed=3))
    assert np.all(np.isfinite(x_t)), f"{name}: sqrt of a negative variance leaked NaN"


@pytest.mark.parametrize(
    "name,sde",
    [
        ("uniform_default", UniformVolatilitySDE()),
        ("periodic_default", PeriodicVolatilitySDE()),
        ("cosine_default", CosineDecayingVolatilitySDE()),
    ],
)
def test_the_shipped_defaults_show_no_float32_negative(name, sde):
    """Characterization arm: at the DEFAULT parameters the clamp is defensive.

    This records the honest scope of the two arms above. If a future change makes
    a default parameterisation go negative, this arm reddens and the comment
    block above must be corrected -- that is the point of asserting it.
    Searched: 200,000 uniform-random ``t`` (seed 0) plus a 10,002-point
    logarithmic ladder into both endpoints, all in float32.
    """
    rng = np.random.default_rng(0)
    grid = np.concatenate(
        [
            rng.random(200_000),
            1.0 - np.logspace(-12, -1, 5000),
            np.logspace(-12, -1, 5000),
            [0.0, 1.0],
        ]
    ).astype("float32")
    t = _t(grid)
    zeros, ones = keras.ops.zeros_like(t), keras.ops.ones_like(t)
    raw = _np(sde.C(zeros, t, t) - sde.C(zeros, t, ones) ** 2 / sde.C(zeros, ones, ones))
    assert raw.min() >= 0.0, (
        f"{name}: a DEFAULT parameterisation now goes negative (min={raw.min():.3e}); "
        "the clamp is no longer merely defensive here -- update the comment block."
    )


# =====================================================================
# 4. TIME_EPS is load-bearing -- the raw endpoints are genuinely singular
# =====================================================================


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_the_reverse_score_target_is_non_finite_at_t_equals_zero(name, sde):
    """``C(0,0,0) = 0`` divides the reverse target: ``t = 0`` must blow up."""
    x_t = _t(np.array([[1.0, -0.5, 2.0]], dtype="float32"))
    x_0 = _t(np.array([[0.0, 0.25, -1.0]], dtype="float32"))
    out = _np(score_target_reverse(sde, x_t, _t([0.0]), x_0))
    assert not np.all(np.isfinite(out)), f"{name}: expected non-finite at t=0, got {out}"


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_the_forward_score_target_is_non_finite_at_t_equals_one(name, sde):
    """``C(1,1,1) = 0`` divides the forward target: ``t = 1`` must blow up."""
    x_t = _t(np.array([[1.0, -0.5, 2.0]], dtype="float32"))
    x_1 = _t(np.array([[0.0, 0.25, -1.0]], dtype="float32"))
    out = _np(score_target_forward(sde, x_t, _t([1.0]), x_1))
    assert not np.all(np.isfinite(out)), f"{name}: expected non-finite at t=1, got {out}"


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_the_weightings_annihilate_the_loss_at_their_own_singular_endpoint(name, sde):
    """The weightings do NOT blow up at the endpoints -- they go to exactly zero.

    Recorded deliberately: the endpoint failure is asymmetric. The score targets
    become non-finite, while the weightings silently multiply the whole loss by
    ``0``. A test that only looked for NaN would call the weightings safe.
    """
    assert _np(dsm_weight_reverse(sde, _t([0.0])))[0] == pytest.approx(0.0, abs=1e-12)
    assert _np(dsm_weight_forward(sde, _t([1.0])))[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_both_score_targets_are_finite_inside_the_TIME_EPS_window(name, sde):
    """The clamp's own interval is safe -- so the clamp actually buys something."""
    grid = np.linspace(TIME_EPS, 1.0 - TIME_EPS, 501, dtype="float32")
    x_t = np.ones((grid.size, 2), dtype="float32")
    other = np.zeros((grid.size, 2), dtype="float32")
    assert np.all(np.isfinite(_np(score_target_reverse(sde, _t(x_t), _t(grid), _t(other)))))
    assert np.all(np.isfinite(_np(score_target_forward(sde, _t(x_t), _t(grid), _t(other)))))


# =====================================================================
# 5. The two weightings are NOT interchangeable
# =====================================================================


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_dsm_weight_reverse_is_the_kernel_variance_C_0_t_t(name, sde):
    t = _t([0.1, 0.25, 0.6, 0.8], "float64")
    expected = _np(sde.C(keras.ops.zeros_like(t), t, t))
    np.testing.assert_allclose(_np(dsm_weight_reverse(sde, t)), expected, rtol=1e-12)


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_dsm_weight_forward_is_C_t_1_1_divided_by_phi_t_1(name, sde):
    """The ``/ phi(t,1)`` factor is the whole difference from a plain variance.

    It is INVISIBLE on every driftless variant (``phi == 1``), which is why
    ``uniform_OU`` (``A = 1.5``) is in the parameter list: it is the only member
    that can see the division at all.
    """
    t = _t([0.1, 0.25, 0.6, 0.8], "float64")
    ones = keras.ops.ones_like(t)
    expected = _np(sde.C(t, ones, ones)) / _np(sde.phi(t, ones))
    np.testing.assert_allclose(_np(dsm_weight_forward(sde, t)), expected, rtol=1e-12)


def test_the_phi_division_is_visible_on_the_OU_variant_and_only_there():
    """Anti-vacuity for the arm above: ``phi != 1`` exactly on the drifted variant."""
    t = _t([0.25, 0.8], "float64")
    ones = keras.ops.ones_like(t)
    for name, sde in STOCHASTIC_SDES:
        phi = _np(sde.phi(t, ones))
        if name == "uniform_OU":
            assert np.all(np.abs(phi - 1.0) > 0.1), f"{name}: phi={phi} cannot see the division"
        else:
            np.testing.assert_allclose(phi, 1.0, rtol=1e-12)


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
@pytest.mark.parametrize("t_val", [0.15, 0.25, 0.8, 0.9])
def test_the_two_weightings_are_not_interchangeable(name, sde, t_val):
    """A refactor that unifies the two weightings reddens here.

    The ``t`` values are chosen deliberately AWAY from the coincidence point
    pinned by :func:`test_the_two_weightings_do_coincide_at_one_specific_t`.
    """
    t = _t([t_val], "float64")
    fwd = _np(dsm_weight_forward(sde, t))[0]
    rev = _np(dsm_weight_reverse(sde, t))[0]
    rel = abs(fwd - rev) / max(abs(fwd), abs(rev))
    assert rel > 0.05, f"{name} at t={t_val}: forward={fwd:.6e} reverse={rev:.6e} rel={rel:.3e}"


def test_the_two_weightings_do_coincide_at_one_specific_t():
    """The coincidence is real, so the choice of ``t`` above is not arbitrary.

    For a driftless process ``forward = G(1) - G(t)`` and ``reverse = G(t)`` with
    ``G`` the integrated ``sigma^2``; they are equal exactly where ``G(t)`` is
    half its total. For ``UniformVolatilitySDE(A=0)`` that is ``t = 0.5``. A
    guard that sampled there would pass under a full unification of the two.
    """
    sde = UniformVolatilitySDE(A=0.0, K=1.0)
    t = _t([0.5], "float64")
    np.testing.assert_allclose(
        _np(dsm_weight_forward(sde, t)), _np(dsm_weight_reverse(sde, t)), rtol=1e-12
    )


# =====================================================================
# 6. Score-target correctness by finite differences (independent oracle)
# =====================================================================


def _fd_gradient(log_density, x, h=1e-5):
    """Central finite difference of a scalar ``log_density(x)`` w.r.t. every entry."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        xp, xm = x.copy(), x.copy()
        xp[idx] += h
        xm[idx] -= h
        grad[idx] = (log_density(xp) - log_density(xm)) / (2.0 * h)
        it.iternext()
    return grad


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_the_reverse_score_target_is_the_gradient_of_log_p_x_t_given_x_0(name, sde, capsys):
    """``score = grad_{x_t} log N(x_t; phi(0,t) x_0, C(0,t,t) I)``."""
    rng = np.random.default_rng(10)
    t = np.array([0.2, 0.45, 0.75], dtype="float64")
    x_t = rng.standard_normal((3, 4))
    x_0 = rng.standard_normal((3, 4))

    zeros = np.zeros_like(t)
    phi_0t = _np(sde.phi(_t(zeros, "float64"), _t(t, "float64")))
    c_tt = _np(sde.C(_t(zeros, "float64"), _t(t, "float64"), _t(t, "float64")))

    def log_density(x):
        resid = x - phi_0t[:, None] * x_0
        return float(np.sum(-(resid ** 2) / (2.0 * c_tt[:, None])))

    numeric = _fd_gradient(log_density, x_t)
    closed = _np(score_target_reverse(sde, _t(x_t, "float64"), _t(t, "float64"), _t(x_0, "float64")))
    scale = np.maximum(np.abs(closed), 1e-8)
    max_rel = float(np.max(np.abs(numeric - closed) / scale))
    with capsys.disabled():
        print(f"\n[FD reverse] {name}: max relative error = {max_rel:.3e}")
    assert max_rel < 1e-6, f"{name}: max rel err {max_rel:.3e}"


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_the_forward_score_target_is_the_gradient_of_log_p_x_1_given_x_t(name, sde, capsys):
    """``score = grad_{x_t} log N(x_1; phi(t,1) x_t, C(t,1,1) I)``.

    Note the gradient is w.r.t. the CONDITIONING variable, which is where the
    extra ``phi`` factor comes from -- and why this target's weighting carries a
    ``1/phi`` the reverse one does not.
    """
    rng = np.random.default_rng(11)
    t = np.array([0.2, 0.45, 0.75], dtype="float64")
    x_t = rng.standard_normal((3, 4))
    x_1 = rng.standard_normal((3, 4))

    ones = np.ones_like(t)
    phi_t1 = _np(sde.phi(_t(t, "float64"), _t(ones, "float64")))
    c_11 = _np(sde.C(_t(t, "float64"), _t(ones, "float64"), _t(ones, "float64")))

    def log_density(x):
        resid = x_1 - phi_t1[:, None] * x
        return float(np.sum(-(resid ** 2) / (2.0 * c_11[:, None])))

    numeric = _fd_gradient(log_density, x_t)
    closed = _np(score_target_forward(sde, _t(x_t, "float64"), _t(t, "float64"), _t(x_1, "float64")))
    scale = np.maximum(np.abs(closed), 1e-8)
    max_rel = float(np.max(np.abs(numeric - closed) / scale))
    with capsys.disabled():
        print(f"\n[FD forward] {name}: max relative error = {max_rel:.3e}")
    assert max_rel < 1e-6, f"{name}: max rel err {max_rel:.3e}"


def test_the_two_score_targets_are_not_the_same_function():
    """Anti-vacuity: swapping them is a detectable change, not a relabelling."""
    sde = PeriodicVolatilitySDE()
    rng = np.random.default_rng(12)
    x_t = _t(rng.standard_normal((3, 4)), "float64")
    other = _t(rng.standard_normal((3, 4)), "float64")
    t = _t([0.2, 0.45, 0.75], "float64")
    fwd = _np(score_target_forward(sde, x_t, t, other))
    rev = _np(score_target_reverse(sde, x_t, t, other))
    assert np.max(np.abs(fwd - rev)) > 1.0, f"targets differ by only {np.max(np.abs(fwd - rev)):.3e}"


# =====================================================================
# 7. The bridge sampler's moments, and its seed threading
# =====================================================================


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_the_posterior_moments_match_a_closed_form_recomputed_in_the_test(name, sde):
    rng = np.random.default_rng(20)
    x_0 = rng.standard_normal((5, 3))
    x_1 = rng.standard_normal((5, 3))
    t = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    exp_mean, exp_var = _closed_form_moments(sde, x_0, x_1, t)
    mean, var = bridge_posterior_moments(
        sde, _t(x_0, "float64"), _t(x_1, "float64"), _t(t, "float64")
    )
    np.testing.assert_allclose(_np(mean), exp_mean, rtol=1e-11)
    np.testing.assert_allclose(_np(var), np.maximum(exp_var, 0.0), rtol=1e-11)


@pytest.mark.parametrize("name,sde", STOCHASTIC_SDES)
def test_the_sampler_reproduces_the_analytic_moments_statistically(name, sde):
    """The only arm that can see a sampler which computes ``var`` and ignores it."""
    n = 20000
    x_0 = np.zeros((n, 1), dtype="float32")
    x_1 = np.ones((n, 1), dtype="float32")
    t = np.full((n,), 0.4, dtype="float32")
    x_t = _np(sample_bridge_x_t(sde, _t(x_0), _t(x_1), _t(t), seed=99))
    mean, var = bridge_posterior_moments(sde, _t(x_0[:1]), _t(x_1[:1]), _t(t[:1]))
    exp_mean = _np(mean)[0, 0]
    exp_std = float(np.sqrt(_np(var)[0]))
    assert abs(x_t.mean() - exp_mean) < 5.0 * exp_std / np.sqrt(n) + 1e-6
    assert abs(x_t.std() - exp_std) < 0.05 * exp_std + 1e-6


def test_the_bridge_sampler_is_deterministic_under_an_explicit_seed():
    """No reliance on global RNG state -- step 9 calls this inside ``tf.data``."""
    sde = PeriodicVolatilitySDE()
    x_0 = _t(np.zeros((8, 3), dtype="float32"))
    x_1 = _t(np.ones((8, 3), dtype="float32"))
    t = _t(np.full((8,), 0.3, dtype="float32"))
    a = _np(sample_bridge_x_t(sde, x_0, x_1, t, seed=1234))
    b = _np(sample_bridge_x_t(sde, x_0, x_1, t, seed=1234))
    c = _np(sample_bridge_x_t(sde, x_0, x_1, t, seed=4321))
    np.testing.assert_array_equal(a, b)
    assert np.max(np.abs(a - c)) > 1e-3, "a different seed produced the same draw"


def test_the_bridge_sampler_accepts_a_seed_generator():
    """A ``SeedGenerator`` advances, so consecutive calls differ."""
    sde = PeriodicVolatilitySDE()
    gen = keras.random.SeedGenerator(seed=5)
    x_0 = _t(np.zeros((8, 3), dtype="float32"))
    x_1 = _t(np.ones((8, 3), dtype="float32"))
    t = _t(np.full((8,), 0.3, dtype="float32"))
    a = _np(sample_bridge_x_t(sde, x_0, x_1, t, seed=gen))
    b = _np(sample_bridge_x_t(sde, x_0, x_1, t, seed=gen))
    assert np.max(np.abs(a - b)) > 1e-3


@pytest.mark.parametrize("rank", [2, 4])
def test_the_bridge_math_broadcasts_a_per_sample_t_over_any_rank(rank):
    """``t`` is ``(B,)``; the bridge tensor is rank-4 in production."""
    sde = PeriodicVolatilitySDE()
    shape = (6, 3) if rank == 2 else (6, 4, 4, 2)
    x_0 = _t(np.zeros(shape, dtype="float32"))
    x_1 = _t(np.ones(shape, dtype="float32"))
    t = _t(np.linspace(0.1, 0.9, 6).astype("float32"))
    x_t = sample_bridge_x_t(sde, x_0, x_1, t, seed=0)
    assert tuple(keras.ops.shape(x_t)) == shape
    assert tuple(keras.ops.shape(score_target_reverse(sde, x_t, t, x_0))) == shape
    assert tuple(keras.ops.shape(score_target_forward(sde, x_t, t, x_1))) == shape


# =====================================================================
# 8. Never-narrow dtype
# =====================================================================


@pytest.mark.parametrize("in_dtype,expected", [("float16", "float32"), ("float32", "float32"), ("float64", "float64")])
def test_the_bridge_math_never_narrows_below_float32(in_dtype, expected):
    sde = PeriodicVolatilitySDE()
    x_0 = _t(np.zeros((3, 2)), in_dtype)
    x_1 = _t(np.ones((3, 2)), in_dtype)
    t = _t(np.array([0.2, 0.5, 0.8]), in_dtype)
    mean, var = bridge_posterior_moments(sde, x_0, x_1, t)
    assert keras.backend.standardize_dtype(mean.dtype) == expected
    assert keras.backend.standardize_dtype(var.dtype) == expected
    assert keras.backend.standardize_dtype(
        score_target_reverse(sde, x_0, t, x_1).dtype) == expected
    assert keras.backend.standardize_dtype(
        score_target_forward(sde, x_0, t, x_1).dtype) == expected
    assert keras.backend.standardize_dtype(dsm_weight_forward(sde, t).dtype) == expected
    assert keras.backend.standardize_dtype(dsm_weight_reverse(sde, t).dtype) == expected
    assert keras.backend.standardize_dtype(
        sample_bridge_x_t(sde, x_0, x_1, t, seed=0).dtype) == expected


# =====================================================================
# 9. Flow-matching interpolant and target
# =====================================================================


def test_the_flow_matching_interpolant_is_the_straight_line():
    rng = np.random.default_rng(30)
    x_0 = rng.standard_normal((4, 3))
    x_1 = rng.standard_normal((4, 3))
    t = np.array([0.0, 0.25, 0.5, 1.0])
    got = _np(flow_matching_interpolant(_t(x_0, "float64"), _t(x_1, "float64"), _t(t, "float64")))
    expected = (1.0 - t)[:, None] * x_0 + t[:, None] * x_1
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    np.testing.assert_allclose(got[0], x_0[0], rtol=1e-12)
    np.testing.assert_allclose(got[3], x_1[3], rtol=1e-12)


def test_the_flow_matching_target_is_the_endpoint_difference():
    rng = np.random.default_rng(31)
    x_0 = rng.standard_normal((4, 3))
    x_1 = rng.standard_normal((4, 3))
    got = _np(flow_matching_target(_t(x_0, "float64"), _t(x_1, "float64")))
    np.testing.assert_allclose(got, x_1 - x_0, rtol=1e-12)


def test_the_flow_matching_target_is_time_independent():
    """Both directions regress the same constant velocity -- a real upstream property.

    A ``t`` parameter here would be an invented knob: the straight-line path has
    constant velocity and upstream's ``flow_matching_loss`` weights it not at all.
    """
    import inspect

    assert "t" not in inspect.signature(flow_matching_target).parameters


# =====================================================================
# 10. Time samplers
# =====================================================================


#: The window boundaries as the float32 sampler can actually represent them.
#: ``float32(1e-4) = 9.99999975e-05`` sits just BELOW the float64 literal, so a
#: float64 comparison would reject the sampler's own exact lower bound.
LO32 = np.float32(TIME_EPS)
HI32 = np.float32(1.0 - TIME_EPS)


def test_the_uniform_time_sampler_stays_inside_the_TIME_EPS_window():
    """200k draws: plain ``uniform(0,1)`` would leave the window with probability ~1."""
    t = _np(sample_timesteps_uniform(200_000, seed=0)).astype("float32")
    assert t.min() >= LO32, f"min = {t.min():.6e} < TIME_EPS = {TIME_EPS}"
    assert t.max() <= HI32, f"max = {t.max():.6e} > 1 - TIME_EPS"
    # Anti-vacuity: the sampler really does reach towards both ends.
    assert t.min() < 1e-3 and t.max() > 1.0 - 1e-3


def test_the_uniform_time_sampler_is_actually_uniform():
    t = _np(sample_timesteps_uniform(200_000, seed=1))
    assert abs(t.mean() - 0.5) < 5e-3
    assert abs(t.std() - 1.0 / np.sqrt(12.0)) < 5e-3


def test_the_logit_normal_sampler_stays_inside_the_TIME_EPS_window():
    """``p_std = 6`` is what makes the clip observable.

    At the shipped defaults (``p_mean=0.4, p_std=0.7``) reaching ``t < 1e-4``
    needs a ``-13.7 sigma`` normal draw, so the clip NEVER fires there and a
    default-only arm would be vacuous. This arm widens the distribution until the
    clip is the only thing holding the range.
    """
    t = _np(sample_timesteps_logit_normal(200_000, p_mean=0.0, p_std=6.0, seed=0)).astype("float32")
    assert t.min() >= LO32, f"min = {t.min():.6e} < TIME_EPS"
    assert t.max() <= HI32, f"max = {t.max():.6e} > 1 - TIME_EPS"
    # Anti-vacuity: the clip is genuinely being hit, not merely available.
    n_clipped = int((t == LO32).sum() + (t == HI32).sum())
    assert n_clipped > 100, f"only {n_clipped} draws were clipped; the arm is vacuous"


def test_the_logit_normal_sampler_at_the_shipped_defaults_is_in_range():
    t = _np(sample_timesteps_logit_normal(50_000, seed=2)).astype("float32")
    assert t.min() >= LO32 and t.max() <= HI32
    assert LOGIT_NORMAL_P_MEAN == 0.4 and LOGIT_NORMAL_P_STD == 0.7


def test_the_logit_normal_p_mean_is_live():
    """Anti-vacuity for ``p_mean``: shifting it moves the distribution's mean."""
    lo = _np(sample_timesteps_logit_normal(50_000, p_mean=-2.0, p_std=0.7, seed=3)).mean()
    mid = _np(sample_timesteps_logit_normal(50_000, p_mean=0.4, p_std=0.7, seed=3)).mean()
    hi = _np(sample_timesteps_logit_normal(50_000, p_mean=2.0, p_std=0.7, seed=3)).mean()
    assert lo < mid < hi, f"p_mean is dead: {lo:.4f} {mid:.4f} {hi:.4f}"
    assert hi - lo > 0.5, f"p_mean barely moved the mean: {hi - lo:.4f}"


def test_the_logit_normal_p_std_is_live():
    narrow = _np(sample_timesteps_logit_normal(50_000, p_mean=0.0, p_std=0.2, seed=4)).std()
    wide = _np(sample_timesteps_logit_normal(50_000, p_mean=0.0, p_std=3.0, seed=4)).std()
    assert wide > 2.0 * narrow, f"p_std is dead: narrow={narrow:.4f} wide={wide:.4f}"


@pytest.mark.parametrize(
    "sampler", [sample_timesteps_uniform, sample_timesteps_logit_normal]
)
def test_the_time_samplers_are_deterministic_under_a_fixed_seed(sampler):
    a = _np(sampler(1024, seed=7))
    b = _np(sampler(1024, seed=7))
    c = _np(sampler(1024, seed=8))
    np.testing.assert_array_equal(a, b)
    assert np.max(np.abs(a - c)) > 1e-3, "a different seed produced the same draw"


@pytest.mark.parametrize(
    "sampler", [sample_timesteps_uniform, sample_timesteps_logit_normal]
)
def test_the_time_samplers_return_a_flat_batch_of_the_working_dtype(sampler):
    t = sampler(16, seed=0)
    assert tuple(keras.ops.shape(t)) == (16,)
    assert keras.backend.standardize_dtype(t.dtype) == "float32"
    t64 = sampler(16, seed=0, dtype="float64")
    assert keras.backend.standardize_dtype(t64.dtype) == "float64"


@pytest.mark.parametrize(
    "sampler", [sample_timesteps_uniform, sample_timesteps_logit_normal]
)
def test_the_time_samplers_reject_a_non_positive_batch(sampler):
    with pytest.raises(ValueError):
        sampler(0, seed=0)


def test_the_sampled_times_feed_the_score_targets_without_blowing_up():
    """End-to-end: the sampler's own output is safe for every downstream user."""
    sde = CosineDecayingVolatilitySDE()
    t = sample_timesteps_logit_normal(4096, p_mean=0.0, p_std=6.0, seed=0)
    x = _t(np.ones((4096, 2), dtype="float32"))
    assert np.all(np.isfinite(_np(score_target_reverse(sde, x, t, x))))
    assert np.all(np.isfinite(_np(score_target_forward(sde, x, t, x))))
    assert np.all(np.isfinite(_np(dsm_weight_reverse(sde, t))))
    assert np.all(np.isfinite(_np(dsm_weight_forward(sde, t))))
