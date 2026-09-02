"""Tests for :mod:`dl_techniques.utils.ddpm_schedule`.

The oracle in this file is a LINE-BY-LINE transcription of the on-disk upstream
reference, not a re-derivation from the DDPM paper and not a second copy of the
module under test written from memory:

    plans/plan-2026-09-02T170923-1285ed83/reference/diffusion/gaussian_diffusion.py
        get_beta_schedule           (the "linear" branch)
        get_named_beta_schedule
        betas_for_alpha_bar
        GaussianDiffusion.__init__  (every derived table)
    plans/plan-2026-09-02T170923-1285ed83/reference/diffusion/respace.py
        space_timesteps
        SpacedDiffusion.__init__    (the new_betas / timestep_map derivation)

Tolerance: every table comparison below is EXACT -- ``assert_array_equal``,
i.e. ``atol=0, rtol=0``. The module performs the same operations in the same
order on the same ``float64`` inputs, so any difference at all is a defect, and
a loose tolerance here would hide exactly the kind of silent reordering or
narrowing this module exists to prevent.
"""

import math
import pytest
import numpy as np
from typing import Any, Callable, Dict, List, Sequence, Set

from dl_techniques.utils.ddpm_schedule import (
    DDPMSchedule,
    MAX_BETA,
    VALID_BETA_SCHEDULES,
    betas_for_alpha_bar,
    get_named_beta_schedule,
    space_timesteps,
)


# ---------------------------------------------------------------------
# The upstream oracle (transcribed by line -- do not "simplify")
# ---------------------------------------------------------------------


def oracle_betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """gaussian_diffusion.py :: betas_for_alpha_bar."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def oracle_get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """gaussian_diffusion.py :: get_named_beta_schedule (+ the "linear" branch of
    get_beta_schedule, inlined at its single call site)."""
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
        assert betas.shape == (num_diffusion_timesteps,)
        return betas
    elif schedule_name == "squaredcos_cap_v2":
        return oracle_betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def oracle_tables(betas) -> Dict[str, np.ndarray]:
    """gaussian_diffusion.py :: GaussianDiffusion.__init__ (the table block)."""
    # Use float64 for accuracy.
    betas = np.array(betas, dtype=np.float64)
    assert len(betas.shape) == 1, "betas must be 1-D"
    assert (betas > 0).all() and (betas <= 1).all()

    num_timesteps = int(betas.shape[0])

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)
    assert alphas_cumprod_prev.shape == (num_timesteps,)

    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    log_one_minus_alphas_cumprod = np.log(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_log_variance_clipped = np.log(
        np.append(posterior_variance[1], posterior_variance[1:])
    ) if len(posterior_variance) > 1 else np.array([])

    posterior_mean_coef1 = (
        betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coef2 = (
        (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "alphas_cumprod_next": alphas_cumprod_next,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "log_one_minus_alphas_cumprod": log_one_minus_alphas_cumprod,
        "sqrt_recip_alphas_cumprod": sqrt_recip_alphas_cumprod,
        "sqrt_recipm1_alphas_cumprod": sqrt_recipm1_alphas_cumprod,
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": posterior_log_variance_clipped,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
    }


def oracle_space_timesteps(num_timesteps, section_counts):
    """respace.py :: space_timesteps."""
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def oracle_respaced(base_alphas_cumprod, use_timesteps):
    """respace.py :: SpacedDiffusion.__init__ (the new_betas / timestep_map loop)."""
    use_timesteps = set(use_timesteps)
    timestep_map = []
    last_alpha_cumprod = 1.0
    new_betas = []
    for i, alpha_cumprod in enumerate(base_alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)
    return np.array(new_betas), timestep_map


TABLE_NAMES = tuple(oracle_tables(np.linspace(1e-4, 0.02, 8)).keys())

SCHEDULE_CASES = [
    ("linear", 1000),
    ("linear", 250),
    ("linear", 50),
    ("squaredcos_cap_v2", 1000),
    ("squaredcos_cap_v2", 100),
    ("squaredcos_cap_v2", 7),
]


# ---------------------------------------------------------------------
# Beta schedules against the oracle
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,steps", SCHEDULE_CASES)
def test_named_beta_schedule_matches_the_oracle_exactly(name: str, steps: int) -> None:
    np.testing.assert_array_equal(
        get_named_beta_schedule(name, steps),
        oracle_get_named_beta_schedule(name, steps),
    )


@pytest.mark.parametrize("name,steps", SCHEDULE_CASES)
def test_betas_are_float64(name: str, steps: int) -> None:
    assert get_named_beta_schedule(name, steps).dtype == np.float64


def test_unknown_schedule_name_raises_value_error_listing_the_valid_names() -> None:
    with pytest.raises(ValueError) as excinfo:
        get_named_beta_schedule("cosine", 1000)
    message = str(excinfo.value)
    for valid in VALID_BETA_SCHEDULES:
        assert valid in message


def test_non_positive_step_count_raises_value_error() -> None:
    with pytest.raises(ValueError):
        get_named_beta_schedule("linear", 0)


def test_the_linear_schedule_is_rescaled_for_a_short_chain() -> None:
    """RED-proof target 1: dropping ``scale = 1000 / T`` moves both endpoints.

    At ``T = 250`` the scale is exactly 4.0, so the endpoints are 4e-4 and 0.08
    rather than the 1000-step 1e-4 and 0.02.
    """
    betas = get_named_beta_schedule("linear", 250)
    assert betas[0] == pytest.approx(4e-4, rel=1e-12)
    assert betas[-1] == pytest.approx(0.08, rel=1e-12)


def test_the_linear_rescale_is_inert_at_exactly_one_thousand_steps() -> None:
    """The RED proof above CANNOT be run at T=1000: there ``scale`` is 1.0.

    Pinned so nobody re-derives the false claim that removing the rescale is
    detectable at the default step count (the step-1 meshgrid lesson, D-008).
    """
    betas = get_named_beta_schedule("linear", 1000)
    assert betas[0] == pytest.approx(1e-4, rel=1e-15)
    assert betas[-1] == pytest.approx(0.02, rel=1e-15)
    unscaled = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    np.testing.assert_array_equal(betas, unscaled)


def test_the_cosine_schedule_is_capped() -> None:
    betas = get_named_beta_schedule("squaredcos_cap_v2", 1000)
    assert betas.max() <= MAX_BETA
    assert MAX_BETA == 0.999
    # Anti-vacuity: the cap is actually reached, so the assertion above is not
    # trivially satisfied by a schedule that never approaches it.
    assert (betas >= MAX_BETA).sum() >= 1


def test_betas_for_alpha_bar_matches_the_oracle_exactly() -> None:
    fn = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    np.testing.assert_array_equal(
        betas_for_alpha_bar(64, fn), oracle_betas_for_alpha_bar(64, fn)
    )


# ---------------------------------------------------------------------
# The derived tables against the oracle
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,steps", SCHEDULE_CASES)
@pytest.mark.parametrize("table", TABLE_NAMES)
def test_every_derived_table_matches_the_oracle_exactly(
        name: str, steps: int, table: str
) -> None:
    schedule = DDPMSchedule.from_name(name, steps)
    expected = oracle_tables(oracle_get_named_beta_schedule(name, steps))[table]
    np.testing.assert_array_equal(getattr(schedule, table), expected)


@pytest.mark.parametrize("name,steps", SCHEDULE_CASES)
@pytest.mark.parametrize("table", TABLE_NAMES)
def test_every_table_stays_float64(name: str, steps: int, table: str) -> None:
    """No field may be narrowed here -- consumers cast at the point of use."""
    assert getattr(DDPMSchedule.from_name(name, steps), table).dtype == np.float64


@pytest.mark.parametrize("name,steps", SCHEDULE_CASES)
def test_every_table_has_length_num_timesteps(name: str, steps: int) -> None:
    schedule = DDPMSchedule.from_name(name, steps)
    assert schedule.num_timesteps == steps
    for table in TABLE_NAMES:
        assert getattr(schedule, table).shape == (steps,)


# ---------------------------------------------------------------------
# The explicitly pinned values
# ---------------------------------------------------------------------


def test_alphas_cumprod_first_entry_is_one_minus_the_first_beta() -> None:
    schedule = DDPMSchedule.from_name("linear", 1000)
    assert schedule.alphas_cumprod[0] == pytest.approx(0.9999, rel=1e-15)


def test_alphas_cumprod_last_entry_is_the_published_residual_signal() -> None:
    schedule = DDPMSchedule.from_name("linear", 1000)
    assert schedule.alphas_cumprod[-1] == pytest.approx(
        4.035829765375676e-05, rel=1e-12
    )


@pytest.mark.parametrize("name,steps", SCHEDULE_CASES)
def test_alphas_cumprod_prev_starts_at_exactly_one(name: str, steps: int) -> None:
    schedule = DDPMSchedule.from_name(name, steps)
    assert schedule.alphas_cumprod_prev[0] == 1.0
    np.testing.assert_array_equal(
        schedule.alphas_cumprod_prev[1:], schedule.alphas_cumprod[:-1]
    )


@pytest.mark.parametrize("name,steps", SCHEDULE_CASES)
def test_alphas_cumprod_next_ends_at_exactly_zero(name: str, steps: int) -> None:
    schedule = DDPMSchedule.from_name(name, steps)
    assert schedule.alphas_cumprod_next[-1] == 0.0
    np.testing.assert_array_equal(
        schedule.alphas_cumprod_next[:-1], schedule.alphas_cumprod[1:]
    )


@pytest.mark.parametrize("name,steps", SCHEDULE_CASES)
def test_the_posterior_log_variance_is_clipped_at_the_head_of_the_chain(
        name: str, steps: int
) -> None:
    """RED-proof target 2: ``posterior_variance[0]`` is exactly 0.

    Upstream substitutes entry 1 for entry 0 BEFORE the log, so entries 0 and 1
    are equal and finite. A plain ``np.log(posterior_variance)`` yields ``-inf``
    at entry 0.
    """
    schedule = DDPMSchedule.from_name(name, steps)
    assert schedule.posterior_variance[0] == 0.0
    plvc = schedule.posterior_log_variance_clipped
    assert plvc[0] == plvc[1]
    assert np.isfinite(plvc).all()


@pytest.mark.parametrize("name,steps", SCHEDULE_CASES)
def test_every_table_is_finite(name: str, steps: int) -> None:
    schedule = DDPMSchedule.from_name(name, steps)
    for table in TABLE_NAMES:
        assert np.isfinite(getattr(schedule, table)).all(), table


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        np.array([0.0, 0.1]),
        np.array([-0.1, 0.1]),
        np.array([0.1, 1.5]),
    ],
)
def test_betas_outside_the_unit_interval_raise_value_error(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        DDPMSchedule.from_betas(bad)


def test_a_beta_of_exactly_one_is_accepted() -> None:
    """The bound is ``betas <= 1``, not ``< 1`` -- an off-by-one here would
    reject the legal terminal step of a fully destroying schedule."""
    schedule = DDPMSchedule.from_betas(np.array([0.5, 1.0]))
    assert schedule.alphas_cumprod[-1] == 0.0


def test_non_one_dimensional_betas_raise_value_error() -> None:
    with pytest.raises(ValueError):
        DDPMSchedule.from_betas(np.zeros((2, 3)) + 0.1)


def test_empty_betas_raise_value_error() -> None:
    with pytest.raises(ValueError):
        DDPMSchedule.from_betas(np.array([]))


# ---------------------------------------------------------------------
# space_timesteps
# ---------------------------------------------------------------------


def test_space_timesteps_250_retains_exactly_250_indices() -> None:
    retained = space_timesteps(1000, "250")
    assert len(retained) == 250
    assert retained == oracle_space_timesteps(1000, "250")


def test_space_timesteps_ddim250_retains_exactly_250_indices() -> None:
    retained = space_timesteps(1000, "ddim250")
    assert len(retained) == 250
    assert retained == oracle_space_timesteps(1000, "ddim250")
    # The DDIM form is a fixed stride, unlike the section form.
    assert retained == set(range(0, 1000, 4))


def test_the_two_forms_of_250_steps_are_not_the_same_set() -> None:
    """Anti-vacuity for the pair above: they agree on the count, not the indices."""
    assert space_timesteps(1000, "250") != space_timesteps(1000, "ddim250")


@pytest.mark.parametrize("spec", ["10,15,20", "1", "500", "ddim100"])
def test_space_timesteps_matches_the_oracle(spec: str) -> None:
    assert space_timesteps(1000, spec) == oracle_space_timesteps(1000, spec)


def test_the_default_path_retains_every_timestep() -> None:
    """Upstream's empty-respacing default is ``[diffusion_steps]``
    (reference/diffusion/__init__.py). Both the list and the int form of it here
    must retain the whole chain."""
    assert space_timesteps(1000, [1000]) == set(range(1000))
    assert space_timesteps(1000, 1000) == set(range(1000))
    assert space_timesteps(1000, [1000]) == oracle_space_timesteps(1000, [1000])


def test_a_section_count_larger_than_its_section_raises_value_error() -> None:
    with pytest.raises(ValueError):
        space_timesteps(1000, "1001")
    with pytest.raises(ValueError):
        space_timesteps(100, "30,40,50")


def test_an_impossible_ddim_stride_raises_value_error() -> None:
    """No integer stride yields exactly 999 steps out of 1000: stride 1 gives
    1000 and stride 2 gives 500."""
    with pytest.raises(ValueError):
        space_timesteps(1000, "ddim999")


# ---------------------------------------------------------------------
# Respacing
# ---------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["250", "ddim250", "10,15,20"])
def test_respaced_betas_match_the_upstream_spaced_diffusion_derivation(
        spec: str,
) -> None:
    base = DDPMSchedule.from_name("linear", 1000)
    retained = space_timesteps(1000, spec)
    respaced, timestep_map = base.respaced(retained)

    expected_betas, expected_map = oracle_respaced(base.alphas_cumprod, retained)
    np.testing.assert_array_equal(respaced.betas, expected_betas)
    np.testing.assert_array_equal(timestep_map, np.array(expected_map))
    assert respaced.num_timesteps == len(retained)


def test_the_respaced_chain_reaches_the_same_cumulative_signal_levels() -> None:
    """The point of the ratio derivation: the shortened chain's alphas_cumprod
    equals the original's AT THE RETAINED INDICES. A naive subsample of betas
    would not."""
    base = DDPMSchedule.from_name("linear", 1000)
    retained = space_timesteps(1000, "ddim250")
    respaced, timestep_map = base.respaced(retained)
    np.testing.assert_allclose(
        respaced.alphas_cumprod,
        base.alphas_cumprod[timestep_map],
        rtol=1e-12,
        atol=0.0,
    )
    naive = DDPMSchedule.from_betas(base.betas[timestep_map])
    assert not np.allclose(
        naive.alphas_cumprod, base.alphas_cumprod[timestep_map], rtol=1e-3
    )


def test_the_timestep_map_is_increasing_and_indexes_the_original_chain() -> None:
    base = DDPMSchedule.from_name("linear", 1000)
    _, timestep_map = base.respaced(space_timesteps(1000, "250"))
    assert timestep_map.dtype == np.int64
    assert (np.diff(timestep_map) > 0).all()
    assert timestep_map[0] >= 0 and timestep_map[-1] < 1000


def test_respacing_to_the_full_chain_reproduces_the_original_betas() -> None:
    base = DDPMSchedule.from_name("linear", 50)
    respaced, timestep_map = base.respaced(range(50))
    np.testing.assert_allclose(respaced.betas, base.betas, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(timestep_map, np.arange(50))


def test_respacing_with_an_out_of_range_index_raises_value_error() -> None:
    base = DDPMSchedule.from_name("linear", 50)
    with pytest.raises(ValueError):
        base.respaced([0, 50])
    with pytest.raises(ValueError):
        base.respaced([-1, 3])


def test_respacing_with_no_retained_timesteps_raises_value_error() -> None:
    base = DDPMSchedule.from_name("linear", 50)
    with pytest.raises(ValueError):
        base.respaced([])


# ---------------------------------------------------------------------
# The value object itself
# ---------------------------------------------------------------------


def test_the_schedule_is_frozen() -> None:
    schedule = DDPMSchedule.from_name("linear", 1000)
    with pytest.raises(Exception):
        schedule.betas = np.zeros(10)


def test_two_schedules_from_the_same_name_are_bit_identical() -> None:
    a = DDPMSchedule.from_name("squaredcos_cap_v2", 100)
    b = DDPMSchedule.from_name("squaredcos_cap_v2", 100)
    for table in TABLE_NAMES:
        np.testing.assert_array_equal(getattr(a, table), getattr(b, table))
