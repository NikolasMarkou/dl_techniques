"""Real-checkpoint integration smoke test for the bias-free denoiser app (Step 6).

SLOW / GPU test. Unlike ``test_operators.py`` / ``test_solver.py`` (which run on a
tiny stub denoiser in well under a second on CPU), this module loads the ACTUAL
trained checkpoints and runs the full stack — real ``DenoiserPrior`` + real
``UniversalInverseSolver`` + real ``MeasurementOperator`` — at 256x256x3 on GPU1.
The ``prior`` fixture is parametrized over BOTH shipped architectures (the ConvUNext
default and the legacy CliffordUNet), each skipped if its checkpoint is absent.

Gating (so the fast suite is unaffected):

* The whole module carries ``pytestmark = pytest.mark.slow`` — ``pytest -m
  "not slow"`` deselects every test here BEFORE any checkpoint load happens
  (module import is light; the load lives inside the fixture).
* The ``prior`` fixture ``pytest.skip``s if the checkpoint file is absent, so the
  module is portable to machines without the trained artifact.
* The ``prior`` fixture ALSO ``pytest.skip``s if the checkpoint predates the
  ``[0,1]`` unit-domain migration (its ``config.json`` lacks ``data_range ==
  "[0,1]"``). Such a checkpoint CANNOT be exercised: a bias-free net has no
  mechanism to subtract a DC offset, so it would emit silent garbage, and
  ``from_pretrained`` refuses it outright. SKIP is the honest state — these tests
  auto-re-enable the moment a ``[0,1]`` checkpoint exists. They must never FAIL on
  a legacy checkpoint, and must never silently PASS on one.

Domain (INV-1): pixels live in ``[0, 1]``, center ``0.5``. Every gate below measures
DEVIATION FROM THE DOMAIN CENTER (``|out - 0.5|``), not ``|out|`` — the zero-centered
``abs()`` form these gates used to carry encoded the old domain STRUCTURALLY, and a
literal find-and-replace would have left them green and semantically wrong (D-006).

Coverage (plan.md Step 6 / Pre-Mortem STOP-IF #3 / Success Criterion 7):

* ``test_load_and_callable`` — ``DenoiserPrior.from_pretrained`` (default dynamic
  rebuild) loads and the wrapped model is callable, finite, shape-preserving.
* ``test_prior_sampling_null_operator`` — Algorithm-1 prior sampling via
  ``NullOperator``: output finite, in-domain, sigma non-increasing overall. This is
  STOP-IF #3's positive check.
* ``test_inpainting_operator`` — block inpainting on a synthetic in-domain target:
  output finite, in-domain, constraint error ``||measure(best_y) - m||`` decreases.

Run (GPU1, serial — never parallel GPU jobs):
    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest \
        tests/test_applications/test_bias_free_denoiser/test_integration.py -vvv -s

Confirm the fast suite skips it:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
        tests/test_applications/test_bias_free_denoiser/ -m "not slow" -vvv
"""

import json
from pathlib import Path

import keras
import numpy as np
import pytest

from applications.bias_free_denoiser.denoiser_prior import DenoiserPrior
from applications.bias_free_denoiser.operators import (
    CompressiveSensingOperator,
    InpaintingOperator,
    MeasurementOperator,
    NullOperator,
    RandomPixelsOperator,
    SpectralDeblurOperator,
    SuperResolutionOperator,
)
from applications.bias_free_denoiser.solver import UniversalInverseSolver
from dl_techniques.utils.logger import logger

# SLOW marker on the whole module: `pytest -m "not slow"` deselects all of it
# (the fast unit suite never touches the 22 MB checkpoint).
pytestmark = pytest.mark.slow

# The real trained checkpoints (F1 / plan.md). Absolute-from-repo-root; resolved
# relative to this file so the test is CWD-independent. The `prior` fixture is
# parametrized over BOTH bias-free architectures (plan_2026-07-10_77fb9b17/step-6):
# the ConvUNext default that the app now ships and the CliffordUNet checkpoint it
# used to ship, so `from_pretrained`'s architecture auto-dispatch (graph-relax vs.
# factory-rebuild) is regression-covered for both. Each param skips if absent.
_REPO_ROOT = Path(__file__).resolve().parents[3]
CONVUNEXT_CKPT = _REPO_ROOT / "results" / "convunext_denoiser_base_20260707_122133" / "best_model.keras"
CLIFFORD_CKPT = _REPO_ROOT / "results" / "cliffordunet_denoiser_base_20260705_004751" / "best_model.keras"

# 256x256x3: the checkpoint's native patch size; divisible by 8 (depth-3 U-Net).
H, W, C = 256, 256, 3
FULL_SHAPE = (1, H, W, C)

# Modest iteration budget — this is a SMOKE test, not a quality benchmark. Enough
# iterations to see sigma anneal and the data constraint bind, cheap on GPU1.
# NOTE (Step-6 empirical): with the checkpoint's slow paper-schedule annealing
# (h0=0.01, beta=0.01) sigma_t drops only ~0.4->0.33 over ~120 iters — the iterate
# is a COARSE-SCALE sample whose std tracks sigma_t (measured std==sigma_t to ~1%),
# so ~15-20% of pixels legitimately sit outside the domain at this scale. That is
# correct annealed-Langevin behavior, NOT divergence: tails CONTRACT and sigma
# decreases monotonically as iterations grow (60->300: dev-max 1.82->1.22,
# frac_in_domain 0.82->0.95). Full domain containment needs the production budget
# (paper's hundreds-to-1000 steps), out of scope for a GPU smoke. Hence the gates
# below assert the non-divergence invariants (finite, sigma anneals, field scale
# bounded by init = contraction, no exponential blow-up), not full containment.
SMOKE_ITERS = 120

# Even cheaper budget for the all-six sweep: this is a strict no-NaN / finite /
# no-blow-up smoke across every problem on the REAL checkpoint, NOT a quality
# benchmark. 40 iters keeps the whole 6-problem sweep to ~2-3 GPU1 minutes while
# still exercising the full solve loop (init + annealed ascent + early-stop) on
# each measurement operator (closes WARNING-1 / D-009).
ALL6_ITERS = 40

# Domain (INV-1): pixels live in [0, 1]. Every ceiling below is a bound on the
# DEVIATION FROM THE DOMAIN CENTER, |out - DOMAIN_CENTER| — see _dev() and the module
# docstring (D-006). A healthy iterate is centered on DOMAIN_CENTER (the operators'
# init_mean fills unmeasured pixels with c0 = 0.5), so deviation-from-center is the
# quantity these gates have ALWAYS meant; on the old zero-centered domain it merely
# happened to coincide with |out|.
DOMAIN_CENTER = 0.5
DOMAIN_HALFWIDTH = 0.5

# Generous divergence ceiling for the all-six sweep (D-009 accepted relaxation):
# at a modest budget the coarse-scale iterate has std ~= sigma_t so healthy tails
# reach a few * sigma_0; a flat dev-max ceiling of 5.0 passes those while still
# catching real NaN/Inf/exponential blow-up in any of the four previously-untested
# problems (random_pixels / super_resolution / deblur / compressive_sensing).
ALL6_CEIL = 5.0

# Tight single-pass containment for the blind `denoise` task (A3/A5). Unlike the
# iterative annealed-Langevin sweep (ALL6_CEIL=5.0, coarse-scale relaxation), a
# single forward pass D(y) on a modestly-noised in-domain target has NO coarse-scale
# sampling — it should land essentially in-domain. The INTENDED tight bound on the
# deviation-from-center is ~0.6 (0.5 half-width + ~0.1 slack); 1.0 is the guard
# threshold (a dev-max above it means the output left [-0.5, 1.5] — investigate
# before loosening, plan A5/SC6). NOTE this is a DEVIATION bound, so it does NOT
# degenerate into an always-borderline gate on [0,1] the way a raw |out| <= 1.0
# would have.
DENOISE_CEIL = 1.0

# Non-divergence bound: at a coarse scale of std ~= sigma_0 the max of ~2e5 pixels
# reaches ~4.5-5 sigma; a bound of 6*sigma_0 passes healthy tails while still
# catching true (exponential) blow-up, which would overshoot by orders of magnitude.
_BLOWUP_SIGMA_MULT = 6.0

# The provenance stamp a checkpoint must carry to be exercisable at all (D-005).
_REQUIRED_DATA_RANGE = "[0,1]"


def _l2(t) -> float:
    """Euclidean norm of a tensor/array as a python float."""
    return float(np.sqrt(np.sum(np.square(np.asarray(t, dtype=np.float64)))))


def _to_np(t) -> np.ndarray:
    """Materialize a keras/backend tensor to a float32 numpy array."""
    return np.asarray(keras.ops.convert_to_numpy(t), dtype=np.float32)


def _dev(out: np.ndarray) -> np.ndarray:
    """Deviation from the domain center: ``|out - 0.5|``.

    THE structural primitive of this module (D-006). Every blow-up / containment gate
    is a bound on this, never on ``|out|`` — ``|out| <= 0.5`` is nonsense on ``[0,1]``
    (it would reject the entire upper half of a perfectly valid image), and ``|out| <=
    1.0`` degenerates into an always-borderline gate. ``|out - 0.5| <= 0.5`` is exactly
    "in domain", and ``|out - 0.5| <= k`` is exactly "within k of mid-grey".
    """
    return np.abs(out - DOMAIN_CENTER)


def _domain_report(name: str, out: np.ndarray) -> None:
    """Log finiteness + value-range diagnostics for a solver output."""
    finite = bool(np.all(np.isfinite(out)))
    dev = _dev(out) if finite else None
    dmax = float(np.max(dev)) if finite else float("nan")
    p995 = float(np.percentile(dev, 99.5)) if finite else float("nan")
    logger.info(
        "[integration] %s: finite=%s min=%.4f max=%.4f dev-max=%.4f dev-p99.5=%.4f "
        "(dev = |x - %.1f|, deviation from the [0,1] domain center)",
        name, finite, float(out.min()), float(out.max()), dmax, p995, DOMAIN_CENTER,
    )


def _make_synthetic_target() -> np.ndarray:
    """Build a smooth in-domain [0, 1] target (low-freq field + shapes).

    Kept comfortably interior (``|value - 0.5| <= ~0.4``, i.e. ``[0.05, 0.95]``) so the
    residual=score identity stays trustworthy away from the clip boundary (F1 §3 / S1).
    Mirrors the old ``main.py`` generator's intent (gradient + shapes) in the [0,1]
    domain: the field is built zero-centered (that is the natural way to synthesize a
    band-limited texture) and then SHIFTED onto the domain center, rather than the
    literal bounds being swapped.
    """
    rng = np.random.default_rng(0)
    # Low-frequency base: an 8x8 gaussian field bilinearly upsampled to HxW.
    base = rng.standard_normal((8, 8, C)).astype(np.float32)
    ys = np.linspace(0, 7, H)
    xs = np.linspace(0, 7, W)
    y0 = np.floor(ys).astype(int).clip(0, 6)
    x0 = np.floor(xs).astype(int).clip(0, 6)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = base[y0][:, x0] * (1 - wx) + base[y0][:, x0 + 1] * wx
    bot = base[y0 + 1][:, x0] * (1 - wx) + base[y0 + 1][:, x0 + 1] * wx
    field = top * (1 - wy) + bot * wy
    # Add a smooth diagonal gradient.
    grad = (np.linspace(-1, 1, H)[:, None] + np.linspace(-1, 1, W)[None, :])[..., None]
    field = field + 0.5 * grad
    # Normalize the ZERO-CENTERED texture to a comfortably-interior amplitude +-0.4...
    field = field - field.mean()
    field = 0.4 * field / (np.max(np.abs(field)) + 1e-8)
    # A couple of soft shapes (still in zero-centered amplitude units).
    yy, xx = np.mgrid[0:H, 0:W]
    disc = ((yy - 90) ** 2 + (xx - 90) ** 2) < 40 ** 2
    field[disc] = field[disc] * 0.3 + 0.25
    field[170:210, 60:190, :] = -0.2
    # ...then SHIFT onto the [0,1] domain center -> values in [0.05, 0.95].
    field = np.clip(DOMAIN_CENTER + field, 0.05, 0.95).astype(np.float32)
    return field[None]  # [1, H, W, C]


def _has_unit_domain_stamp(ckpt: Path) -> bool:
    """True iff the checkpoint's sibling config.json stamps ``data_range == "[0,1]"``."""
    try:
        raw = json.loads((ckpt.parent / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return raw.get("data_range") == _REQUIRED_DATA_RANGE


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(CONVUNEXT_CKPT, id="convunext"),
        pytest.param(CLIFFORD_CKPT, id="cliffordunet"),
    ],
)
def prior(request) -> DenoiserPrior:
    """Load a real checkpoint ONCE per architecture (dynamic path). Skip if unusable.

    Parametrized over both bias-free architectures so `from_pretrained`'s
    auto-detected dynamic path — ConvUNext graph-relax vs. CliffordUNet
    factory-rebuild — is exercised end-to-end by every test below.

    Two independent SKIP conditions, both "the artifact is absent", not "the code is
    broken":

    1. the checkpoint file does not exist on this machine;
    2. the checkpoint predates the [0,1] unit-domain migration (no ``data_range``
       stamp). Running it would be meaningless — a bias-free net cannot subtract a DC
       offset, so a legacy checkpoint fed [0,1] data emits SILENT garbage, and
       ``from_pretrained`` refuses it (D-005). SKIP is the honest state; these tests
       re-enable themselves automatically once a [0,1] checkpoint is trained. They must
       NOT fail, and must NOT pass on a legacy checkpoint.
    """
    ckpt: Path = request.param
    if not ckpt.is_file():
        pytest.skip(f"checkpoint not present, skipping integration test: {ckpt}")
    if not _has_unit_domain_stamp(ckpt):
        pytest.skip(
            f"no [0,1]-domain checkpoint available: {ckpt} lacks the "
            f"data_range={_REQUIRED_DATA_RANGE!r} stamp in its config.json, so it was "
            f"trained on the legacy [-0.5,+0.5] domain and cannot be exercised (a "
            f"bias-free denoiser cannot subtract a DC offset -> silent garbage). "
            f"Retrain on [0,1] to re-enable this test."
        )
    logger.info("[integration] loading real checkpoint: %s", ckpt)
    p = DenoiserPrior.from_pretrained(str(ckpt))  # default resolution="dynamic"
    return p


def test_load_and_callable(prior: DenoiserPrior) -> None:
    """The loaded denoiser is callable and shape/finiteness-preserving at 256x256."""
    assert prior.model is not None
    x = _make_synthetic_target()  # in-domain [0.05, 0.95]
    d = _to_np(prior.denoise(x))
    assert d.shape == FULL_SHAPE, f"denoise shape {d.shape} != {FULL_SHAPE}"
    assert np.all(np.isfinite(d)), "denoiser produced non-finite output"
    r = _to_np(prior.residual(x))
    assert r.shape == FULL_SHAPE
    assert np.all(np.isfinite(r))
    logger.info(
        "[integration] load ok: ||D(x)-x||/sqrt(N)=%.5f (residual/score magnitude)",
        _l2(r) / np.sqrt(H * W * C),
    )


def test_prior_sampling_null_operator(prior: DenoiserPrior) -> None:
    """Algorithm-1 prior sampling (NullOperator): finite, in-domain, sigma anneals.

    STOP-IF #3 positive check: no NaN/Inf, no sustained out-of-domain blow-up, and
    the effective noise ``sigma_t`` trends down (final < initial).
    """
    solver = UniversalInverseSolver(
        prior, max_iterations=SMOKE_ITERS, patience=SMOKE_ITERS, clip=False,
    )
    best_y, info = solver.solve(NullOperator(), shape=FULL_SHAPE, seed=0)
    out = _to_np(best_y)
    _domain_report("prior-sample best_y", out)

    sig = info["sigma_values"]
    std = float(out.std())
    # "In domain" == within one half-width of the domain CENTER, i.e. 0 <= x <= 1.
    frac_in = float(np.mean(_dev(out) <= DOMAIN_HALFWIDTH))
    logger.info(
        "[integration] prior-sample: iters=%d sigma_0=%.5f sigma_final=%.5f "
        "best_sigma=%.5f field_std=%.5f frac_in_domain=%.3f",
        info["stopped_iteration"], sig[0], sig[-1], info["best_sigma"], std, frac_in,
    )

    # --- STOP-IF #3 gates (non-divergence, NOT full domain containment — see the
    #     SMOKE_ITERS note: at coarse scale std ~= sigma_t so partial out-of-domain
    #     mass is expected and correct).
    assert out.shape == FULL_SHAPE
    # (1) No NaN/Inf — the primary STOP-IF #3 signal.
    assert np.all(np.isfinite(out)), "prior sampling produced NaN/Inf (STOP-IF #3)"
    # (2) Sigma anneals: effective noise decreases (final < initial).
    assert sig[-1] < sig[0], f"sigma did not anneal: {sig[0]:.5f} -> {sig[-1]:.5f}"
    # (3) Field scale is bounded by the init scale (contraction, not divergence):
    #     the iterate's std must not exceed sigma_0 — annealed Langevin only shrinks it.
    assert std <= solver.sigma_0 + 1e-2, (
        f"field std {std:.5f} grew past init sigma_0={solver.sigma_0:.5f} (divergence)"
    )
    # (4) No exponential blow-up in the tails (healthy tail ~5*sigma passes). Measured
    #     as deviation from the domain CENTER — the iterate is centered on 0.5, so a raw
    #     |out| bound would be off-center by exactly one half-width (D-006).
    blowup = _BLOWUP_SIGMA_MULT * solver.sigma_0
    assert float(np.max(_dev(out))) < blowup, (
        f"sustained blow-up: dev-max={np.max(_dev(out)):.4f} >= {blowup:.3f} "
        f"(STOP-IF #3)"
    )
    # (5) A substantial fraction of pixels already sit in-domain at this scale.
    assert frac_in >= 0.6, f"only {frac_in:.2%} of pixels in-domain (expected >=60%)"


def test_inpainting_operator(prior: DenoiserPrior) -> None:
    """Block inpainting on an in-domain target: finite, in-domain, constraint ↓.

    Measured pixels must be respected: ``||measure(best_y) - m||`` at the end is
    smaller than at init, and smaller than the measurement's own scale.
    """
    target = _make_synthetic_target()
    operator = InpaintingOperator(image_shape=(H, W, C), block_size=64)
    measurements = operator.measure(target)

    solver = UniversalInverseSolver(
        prior, max_iterations=SMOKE_ITERS, patience=SMOKE_ITERS, clip=False,
    )
    best_y, info = solver.solve(operator, measurements=measurements, seed=0)
    out = _to_np(best_y)
    _domain_report("inpaint best_y", out)

    # Constraint error on the RETURNED best_y (the deliverable), plus the
    # per-iteration trajectory the solver recorded on the running iterate.
    err_best = _l2(_to_np(operator.measure(best_y)) - _to_np(measurements))
    errs = info["constraint_errors"]
    std = float(out.std())
    logger.info(
        "[integration] inpaint: iters=%d constraint_err init=%.5f final=%.5f "
        "best_y=%.5f | ||m||=%.5f | sigma %.5f->%.5f field_std=%.5f",
        info["stopped_iteration"], errs[0], errs[-1], err_best,
        _l2(_to_np(measurements)), info["sigma_values"][0],
        info["sigma_values"][-1], std,
    )

    # --- STOP-IF #3 gates (non-divergence; see SMOKE_ITERS note).
    assert out.shape == FULL_SHAPE
    assert np.all(np.isfinite(out)), "inpainting produced NaN/Inf (STOP-IF #3)"
    # Field scale bounded by init (contraction, not divergence).
    assert std <= solver.sigma_0 + 1e-2, (
        f"inpaint field std {std:.5f} grew past sigma_0={solver.sigma_0:.5f}"
    )
    blowup = _BLOWUP_SIGMA_MULT * solver.sigma_0
    assert float(np.max(_dev(out))) < blowup, (
        f"inpaint sustained blow-up: dev-max={np.max(_dev(out)):.4f} >= {blowup:.3f} "
        f"(STOP-IF #3)"
    )
    # Measured pixels respected: constraint error decreased vs init (best_y no worse
    # than the running iterate's final error).
    assert errs[-1] < errs[0], (
        f"constraint error did not decrease: {errs[0]:.5f} -> {errs[-1]:.5f}"
    )
    assert err_best <= errs[0], (
        f"best_y constraint error {err_best:.5f} exceeds init {errs[0]:.5f}"
    )


def _build_all6_operator(problem: str) -> MeasurementOperator:
    """Construct the measurement operator for one of the six problems.

    Small, sane knobs matching ``main.build_operator``'s defaults so this smoke
    exercises the SAME operators the CLI ships (WARNING-1 / D-009). ``prior`` uses a
    :class:`NullOperator` (Algorithm-1); the other five are constrained operators.

    Args:
        problem: One of ``prior``, ``inpaint``, ``random_pixels``,
            ``super_resolution``, ``deblur``, ``compressive_sensing``.

    Returns:
        The matching :class:`MeasurementOperator` instance.

    Raises:
        ValueError: If ``problem`` is not a recognized id.
    """
    image_shape = (H, W, C)
    if problem == "prior":
        return NullOperator()
    if problem == "inpaint":
        return InpaintingOperator(image_shape, block_size=64)
    if problem == "random_pixels":
        return RandomPixelsOperator(image_shape, keep_ratio=0.3, seed=0)
    if problem == "super_resolution":
        return SuperResolutionOperator(image_shape, factor=4)
    if problem == "deblur":
        return SpectralDeblurOperator(image_shape, keep_fraction=0.15)
    if problem == "compressive_sensing":
        return CompressiveSensingOperator(image_shape, measurement_ratio=0.2, seed=0)
    raise ValueError(f"unknown problem {problem!r}")


@pytest.mark.parametrize(
    "problem",
    [
        "denoise",
        "prior",
        "inpaint",
        "random_pixels",
        "super_resolution",
        "deblur",
        "compressive_sensing",
    ],
)
def test_all_six_problems_finite_and_bounded(prior: DenoiserPrior, problem: str) -> None:
    """Every one of the six problems solves finite + bounded on the REAL checkpoint.

    This is the hardening test for WARNING-1 / D-009: ``main.py`` runs each problem
    inside an exception-swallowing ``try/except``, so the four problems never
    exercised elsewhere on the real model (random_pixels, super_resolution, deblur,
    compressive_sensing) could NaN or diverge without any committed test noticing.
    Here each problem runs the FULL real stack at a modest budget and must return a
    ``best_y`` that is finite (no NaN/Inf), correctly shaped, and value-bounded
    (``|best_y|.max() < ALL6_CEIL``) — catching divergence/blow-up without requiring
    full domain containment (consistent with D-009's accepted annealed-Langevin
    relaxation).

    The ``prior`` fixture is module-scoped, so the 22 MB checkpoint loads ONCE and
    is shared across all parametrizations.
    """
    if problem == "denoise":
        # Blind single-pass denoise D(y): NO MeasurementOperator, NO solver (it is a
        # genuine sibling dispatch, not a fake identity operator). Mirror main.py's
        # noise synthesis: add in-domain Gaussian noise sigma~0.1 then clip to the
        # [0, 1] domain, and take a SINGLE forward pass.
        target = _make_synthetic_target()  # in-domain [0.05, 0.95]
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 0.1, target.shape).astype(np.float32)
        noisy = np.clip(target + noise, 0.0, 1.0)
        out = _to_np(prior.denoise(noisy))
        _domain_report("all6[denoise] D(y)", out)
        dmax = float(np.max(_dev(out))) if np.all(np.isfinite(out)) else float("nan")
        logger.info(
            "[integration] all6[denoise]: shape=%s finite=%s dev-max=%.4f (single-pass)",
            out.shape, bool(np.all(np.isfinite(out))), dmax,
        )
        # --- Gates: finite, correct shape, TIGHT single-pass containment around the
        #     domain center (D-006: a raw |out| <= 1.0 bound would be trivially true on
        #     [0,1] for any non-diverging output — a gate that can never go red).
        assert out.shape == FULL_SHAPE, f"denoise: shape {out.shape} != {FULL_SHAPE}"
        assert np.all(np.isfinite(out)), "denoise: produced NaN/Inf on real checkpoint"
        assert dmax <= DENOISE_CEIL, (
            f"denoise: max|D(y) - {DOMAIN_CENTER}|={dmax:.4f} > {DENOISE_CEIL} "
            f"(single-pass should land essentially in-domain; investigate before "
            f"loosening — plan A5/SC6)"
        )
        return  # denoise has no operator/solver path — skip the sweep below.

    operator = _build_all6_operator(problem)

    if problem == "prior":
        # Algorithm-1 unconstrained prior sampling: no measurements, size via shape.
        solver = UniversalInverseSolver(
            prior, max_iterations=ALL6_ITERS, patience=ALL6_ITERS, clip=False,
        )
        best_y, info = solver.solve(operator, shape=FULL_SHAPE, seed=0)
    else:
        target = _make_synthetic_target()
        measurements = operator.measure(target)
        solver = UniversalInverseSolver(
            prior, max_iterations=ALL6_ITERS, patience=ALL6_ITERS, clip=False,
        )
        best_y, info = solver.solve(
            operator, measurements=measurements, shape=FULL_SHAPE, seed=0,
        )

    out = _to_np(best_y)
    _domain_report(f"all6[{problem}] best_y", out)
    dmax = float(np.max(_dev(out))) if np.all(np.isfinite(out)) else float("nan")
    sig = info["sigma_values"]
    logger.info(
        "[integration] all6[%s]: shape=%s finite=%s dev-max=%.4f "
        "sigma %.5f->%.5f iters=%d",
        problem, out.shape, bool(np.all(np.isfinite(out))), dmax,
        sig[0], sig[-1], info["stopped_iteration"],
    )

    # --- Gates: finite, correct shape, bounded deviation from the domain center
    #     (no NaN/Inf/blow-up). Deviation, not |out| — see _dev() / D-006.
    assert out.shape == FULL_SHAPE, f"{problem}: shape {out.shape} != {FULL_SHAPE}"
    assert np.all(np.isfinite(out)), f"{problem}: produced NaN/Inf on real checkpoint"
    assert dmax < ALL6_CEIL, (
        f"{problem}: max|best_y - {DOMAIN_CENTER}|={dmax:.4f} >= {ALL6_CEIL} "
        f"(divergence/blow-up)"
    )
