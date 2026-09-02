"""Quirk guard: with ``learn_sigma=True`` the second channel half is a VARIANCE
LOGIT, not a second epsilon.

**The lines this file pins.**
``src/dl_techniques/models/vision_language/dit/diffusion.py``,
``GaussianDiffusion.p_mean_variance``::

    model_output, model_var_values = keras.ops.split(model_output, 2, axis=-1)
    ...
    frac = (model_var_values + 1.0) / 2.0
    model_log_variance = frac * max_log + (1.0 - frac) * min_log

transcribed from ``reference/diffusion/gaussian_diffusion.py`` (``ModelVarType.
LEARNED_RANGE``), with ``dim=1`` replaced by ``axis=-1`` for this port's
channels-last layout. ``max_log = log(betas[t])`` and
``min_log = posterior_log_variance_clipped[t]``, so ``v = +1`` selects the upper
bound and ``v = -1`` the lower one.

**The plausible WRONG alternatives this file is RED against.**

1. Treating the second half as a second epsilon prediction -- averaging the two
   halves, or reading epsilon out of the SECOND half. Both produce a
   correctly-shaped, finite, trainable sampler that denoises to the wrong mean.
2. Interpolating with ``frac = v`` rather than ``(v + 1) / 2``, i.e. reading the
   logit as if it were already in ``[0, 1]``.

**Why the existing arms do not cover this.**
``test_dit_diffusion.py``'s ``TestPMeanVarianceMatchesTheOracle`` compares every
returned key against a by-line NumPy oracle, which pins the VALUE. It does not
pin the SEPARATION -- that the mean reads only the first half and the variance
only the second. A port that mixed the halves into the mean would disagree with
the oracle only because the oracle happens to be right, and a future refactor
that changed both together would stay green. The arms here assert the
independence directly: perturb one half, watch exactly one output move.
``test_dit.py::test_learn_sigma_doubles_exactly_the_read_out_width`` owns the
width claim; this file owns the MEANING of the extra width.

**RED proof (step 10).** Two injections into ``diffusion.py``:

* ``model_output = 0.5 * (model_output + model_var_values)`` (read the second
  half as a second epsilon and average) -- **4 failed / 13 passed**:
  ``test_perturbing_the_variance_half_leaves_the_mean_bit_identical``,
  ``test_swapping_the_two_halves_changes_every_output``,
  ``test_pred_xstart_is_the_epsilon_inversion_of_the_first_half``,
  ``test_the_average_of_the_two_halves_disagrees_too``.
* ``frac = model_var_values`` (drop the ``(v + 1) / 2`` rescale) --
  **6 failed / 11 passed**: ``test_v_equals_minus_one_selects_the_clipped_posterior``,
  ``test_v_equals_zero_is_the_exact_midpoint`` and all four
  ``test_the_interpolation_is_affine_in_v`` cases.
"""

from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.dit.diffusion import GaussianDiffusion
from dl_techniques.utils.ddpm_schedule import DDPMSchedule

from ._dit_helpers import TINY, np_, built_model

#: The chain length every arm here uses. Long enough that ``betas[t]`` and the
#: clipped posterior log-variance are far apart at the sampled ``t``.
STEPS: int = 1000

#: Batch, spatial side and latent channels for the synthetic cases.
BATCH, SIZE, CHANNELS = 3, 4, 4


class ConstantModel:
    """A stub model callable returning a fixed tensor.

    Interface contract: ``(x, t, **kwargs) -> value``. It ignores its arguments
    entirely, which is the point -- these arms need to place an EXACT model
    output into ``p_mean_variance`` and a real ``DiT`` cannot be asked for one.
    ``call_count`` records how often the sampler invoked it.
    """

    def __init__(self, value: Any) -> None:
        self.value = value
        self.call_count = 0

    def __call__(self, x: Any, t: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        return self.value


def case(
    seed: int = 0, eps_scale: float = 0.5, v_fill: Any = None
) -> Tuple[GaussianDiffusion, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(process, x, t, eps_half, v_half)`` for a learned-range case.

    Interface contract: the two halves are drawn INDEPENDENTLY so an arm can
    replace one without touching the other. ``v_fill``, when given, fills the
    variance half with a constant instead of a draw -- that is how the ``v = +1``
    / ``v = -1`` endpoint arms are written.
    """
    rng = np.random.default_rng(seed)
    schedule = DDPMSchedule.from_name("linear", STEPS)
    process = GaussianDiffusion(schedule, model_var_type="learned_range")
    x = rng.normal(size=(BATCH, SIZE, SIZE, CHANNELS)).astype("float64")
    t = np.array([0, STEPS // 2, STEPS - 1], dtype="int32")
    eps = (rng.normal(size=(BATCH, SIZE, SIZE, CHANNELS)) * eps_scale).astype(
        "float64"
    )
    if v_fill is None:
        v = rng.uniform(-1.0, 1.0, size=(BATCH, SIZE, SIZE, CHANNELS))
    else:
        v = np.full((BATCH, SIZE, SIZE, CHANNELS), float(v_fill))
    return process, x, t, eps, v.astype("float64")


def run(
    process: GaussianDiffusion,
    x: np.ndarray,
    t: np.ndarray,
    eps: np.ndarray,
    v: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Feed ``concat([eps, v], -1)`` through ``p_mean_variance`` and return NumPy."""
    model_out = keras.ops.convert_to_tensor(
        np.concatenate([eps, v], axis=-1)
    )
    out = process.p_mean_variance(
        ConstantModel(model_out),
        keras.ops.convert_to_tensor(x),
        keras.ops.convert_to_tensor(t),
    )
    return {key: np_(value) for key, value in out.items()}


def gathered(table: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``table[t]`` broadcast over ``(B, 1, 1, 1)`` -- the independent lookup."""
    return table[t][:, None, None, None]


# ---------------------------------------------------------------------
# The split is where the docstring says it is
# ---------------------------------------------------------------------


class TestTheSplitIsAtTheChannelMidpoint:
    """Half one drives the mean, half two drives the variance. Nothing crosses."""

    def test_perturbing_the_variance_half_leaves_the_mean_bit_identical(self) -> None:
        """The claim: the second half is NOT a second epsilon."""
        process, x, t, eps, v = case(seed=1)
        base = run(process, x, t, eps, v)
        moved = run(process, x, t, eps, np.clip(v + 0.37, -1.0, 1.0))

        for key in ("mean", "pred_xstart"):
            np.testing.assert_allclose(moved[key], base[key], rtol=0, atol=0.0)
        assert float(np.max(np.abs(moved["log_variance"] - base["log_variance"]))) > 0.0

    def test_perturbing_the_epsilon_half_leaves_the_variance_bit_identical(
        self,
    ) -> None:
        """The mirror image, and the anti-vacuity partner of the arm above."""
        process, x, t, eps, v = case(seed=2)
        base = run(process, x, t, eps, v)
        moved = run(process, x, t, eps + 0.41, v)

        for key in ("variance", "log_variance"):
            np.testing.assert_allclose(moved[key], base[key], rtol=0, atol=0.0)
        for key in ("mean", "pred_xstart"):
            assert float(np.max(np.abs(moved[key] - base[key]))) > 0.0, key

    def test_swapping_the_two_halves_changes_every_output(self) -> None:
        """If the split direction did not matter, nothing above would mean anything."""
        process, x, t, eps, v = case(seed=3)
        straight = run(process, x, t, eps, v)
        swapped = run(process, x, t, v, eps)
        for key in ("mean", "pred_xstart", "log_variance"):
            assert float(np.max(np.abs(swapped[key] - straight[key]))) > 0.0, key


# ---------------------------------------------------------------------
# The second half is a LOGIT in [-1, 1], read through frac = (v + 1) / 2
# ---------------------------------------------------------------------


class TestTheInterpolationEndpoints:
    """``v = +1`` gives ``log(beta_t)``; ``v = -1`` gives the clipped posterior."""

    def test_v_equals_plus_one_selects_log_beta(self) -> None:
        process, x, t, eps, v = case(seed=4, v_fill=1.0)
        got = run(process, x, t, eps, v)
        expected = np.log(gathered(process.schedule.betas, t))
        np.testing.assert_allclose(
            got["log_variance"], np.broadcast_to(expected, got["log_variance"].shape),
            rtol=0, atol=1e-12,
        )

    def test_v_equals_minus_one_selects_the_clipped_posterior(self) -> None:
        process, x, t, eps, v = case(seed=5, v_fill=-1.0)
        got = run(process, x, t, eps, v)
        expected = gathered(process.schedule.posterior_log_variance_clipped, t)
        np.testing.assert_allclose(
            got["log_variance"], np.broadcast_to(expected, got["log_variance"].shape),
            rtol=0, atol=1e-12,
        )

    def test_the_two_endpoints_separate_at_small_t_and_converge_at_large_t(
        self,
    ) -> None:
        """Anti-vacuity, and a MEASURED warning about where this probe is blind.

        ``min_log`` is the clipped posterior log-variance and ``max_log`` is
        ``log(beta_t)``. They are far apart early in the chain and collapse onto
        each other late in it -- measured on the 1000-step ``linear`` schedule,
        the gap runs ``6.06e-01`` at ``t = 0`` down to ``8.24e-07`` at
        ``t = 999``. So a DIFFERENCE-style probe ("v = +1 and v = -1 give
        different variances") reads essentially zero at large ``t`` and proves
        nothing there. The endpoint arms above are written as exact VALUE
        comparisons for exactly that reason, and this arm pins the shape of the
        blindness so nobody rewrites them as a difference.
        """
        process, _, _, _, _ = case(seed=6)
        schedule = process.schedule
        gaps = np.abs(
            np.log(schedule.betas) - schedule.posterior_log_variance_clipped
        )
        assert float(gaps[0]) > 0.5
        assert float(gaps[-1]) < 1e-5
        # Monotone decay, so "small t" and "large t" are the whole story.
        probe = gaps[np.array([0, 250, 500, 750, 999])]
        assert list(probe) == sorted(probe, reverse=True), probe

    def test_v_equals_zero_is_the_exact_midpoint(self) -> None:
        """``frac = (0 + 1)/2 = 1/2``. Under ``frac = v`` this would be ``min_log``."""
        process, x, t, eps, v = case(seed=7, v_fill=0.0)
        got = run(process, x, t, eps, v)
        low = gathered(process.schedule.posterior_log_variance_clipped, t)
        high = np.log(gathered(process.schedule.betas, t))
        expected = np.broadcast_to(0.5 * high + 0.5 * low, got["log_variance"].shape)
        np.testing.assert_allclose(got["log_variance"], expected, rtol=0, atol=1e-12)

        wrong = np.broadcast_to(low, got["log_variance"].shape)
        assert not np.allclose(got["log_variance"], wrong, rtol=0, atol=1e-6)

    @pytest.mark.parametrize("v_fill", [-0.75, -0.25, 0.25, 0.75])
    def test_the_interpolation_is_affine_in_v(self, v_fill: float) -> None:
        process, x, t, eps, v = case(seed=8, v_fill=v_fill)
        got = run(process, x, t, eps, v)
        frac = (v_fill + 1.0) / 2.0
        low = gathered(process.schedule.posterior_log_variance_clipped, t)
        high = np.log(gathered(process.schedule.betas, t))
        expected = np.broadcast_to(
            frac * high + (1.0 - frac) * low, got["log_variance"].shape
        )
        np.testing.assert_allclose(got["log_variance"], expected, rtol=0, atol=1e-12)

    def test_variance_is_the_exponential_of_log_variance(self) -> None:
        process, x, t, eps, v = case(seed=9)
        got = run(process, x, t, eps, v)
        np.testing.assert_allclose(
            got["variance"], np.exp(got["log_variance"]), rtol=1e-12, atol=0.0
        )


# ---------------------------------------------------------------------
# The mean reads the first half AS EPSILON
# ---------------------------------------------------------------------


class TestTheFirstHalfIsEpsilon:
    """``x_0_hat = sqrt_recip[t] * x_t - sqrt_recipm1[t] * eps``, first half only."""

    def test_pred_xstart_is_the_epsilon_inversion_of_the_first_half(self) -> None:
        process, x, t, eps, v = case(seed=10)
        got = run(process, x, t, eps, v)
        expected = gathered(
            process.schedule.sqrt_recip_alphas_cumprod, t
        ) * x - gathered(process.schedule.sqrt_recipm1_alphas_cumprod, t) * eps
        np.testing.assert_allclose(got["pred_xstart"], expected, rtol=0, atol=1e-9)

    def test_the_same_inversion_on_the_second_half_disagrees(self) -> None:
        """Anti-vacuity: the wrong half gives a materially different ``x_0``."""
        process, x, t, eps, v = case(seed=10)
        got = run(process, x, t, eps, v)
        wrong = gathered(
            process.schedule.sqrt_recip_alphas_cumprod, t
        ) * x - gathered(process.schedule.sqrt_recipm1_alphas_cumprod, t) * v
        assert float(np.max(np.abs(got["pred_xstart"] - wrong))) > 1e-3

    def test_the_average_of_the_two_halves_disagrees_too(self) -> None:
        """The other tempting "second epsilon" reading: mix them."""
        process, x, t, eps, v = case(seed=10)
        got = run(process, x, t, eps, v)
        mixed = 0.5 * (eps + v)
        wrong = gathered(
            process.schedule.sqrt_recip_alphas_cumprod, t
        ) * x - gathered(process.schedule.sqrt_recipm1_alphas_cumprod, t) * mixed
        assert float(np.max(np.abs(got["pred_xstart"] - wrong))) > 1e-3


# ---------------------------------------------------------------------
# The model actually emits the 2C the sampler splits
# ---------------------------------------------------------------------


class TestTheModelSuppliesBothHalves:
    """``learn_sigma`` is what makes the variance half exist at all."""

    def test_learn_sigma_true_emits_two_c_channels(self) -> None:
        model = built_model(seed=0)
        assert model.out_channels == 2 * TINY["in_channels"]

    def test_learn_sigma_false_emits_c_and_cannot_be_sampled_learned_range(
        self,
    ) -> None:
        model = built_model(seed=0, learn_sigma=False)
        assert model.out_channels == TINY["in_channels"]

        process = GaussianDiffusion(
            DDPMSchedule.from_name("linear", STEPS), model_var_type="learned_range"
        )
        rng = np.random.default_rng(11)
        x = rng.normal(size=(2, SIZE, SIZE, CHANNELS)).astype("float32")
        stub = ConstantModel(
            keras.ops.convert_to_tensor(
                rng.normal(size=(2, SIZE, SIZE, CHANNELS)).astype("float32")
            )
        )
        with pytest.raises(ValueError, match="2 \\* 4 = 8 channels"):
            process.p_mean_variance(
                stub,
                keras.ops.convert_to_tensor(x),
                keras.ops.convert_to_tensor(np.zeros((2,), "int32")),
            )
