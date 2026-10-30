"""The Phase B corruption functions are exact, keyed by (seed, condition) and arm-free.

Plan plan-2026-10-30T025213-f75ab4e9 step 1. The comparison between arms is paired only
if every arm scores the very same occluded and noisy images, so the masks and the noise
are pinned here: exact patch counts, determinism, independence of the arm, and the noise
statistics before clipping.
"""

import subprocess
import sys

import numpy as np
import pytest

from train.mean_shift import corruption

GRID = 8
PATCH = 4
SIZE = GRID * PATCH


def _images(n, dtype=np.float32, value=None, seed=0):
    """A batch of [0, 1] images with no exact zeros, so a zeroed pixel is an occlusion."""
    if value is not None:
        return np.full((n, SIZE, SIZE, 3), value, dtype=dtype)
    rng = np.random.default_rng(seed)
    return (0.1 + 0.8 * rng.random((n, SIZE, SIZE, 3))).astype(dtype)


def _occluded_patches(x, y):
    """Boolean (N, 8, 8) grid of patches that were zeroed, checking each is all-or-nothing."""
    zeroed = (y == 0).all(axis=-1)
    blocks = zeroed.reshape(len(x), GRID, PATCH, GRID, PATCH)
    per_patch = blocks.sum(axis=(2, 4))
    assert set(np.unique(per_patch)) <= {0, PATCH * PATCH}, "a patch was partly occluded"
    return per_patch == PATCH * PATCH


# ---------------------------------------------------------------------------
# occlude
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fraction, expected", [(0.25, 16), (0.5, 32), (0.0, 0), (1.0, 64)])
@pytest.mark.parametrize("n", [1, 7, 64, 130])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_every_image_loses_exactly_the_asked_number_of_patches(fraction, expected, n, dtype):
    x = _images(n, dtype)
    y = corruption.occlude(x, fraction, seed=3, condition="occ")
    patches = _occluded_patches(x, y)
    assert patches.reshape(n, -1).sum(axis=1).tolist() == [expected] * n


def test_the_untouched_patches_keep_their_pixels():
    x = _images(20)
    y = corruption.occlude(x, 0.25, seed=1, condition="occ25")
    keep = ~np.kron(_occluded_patches(x, y), np.ones((PATCH, PATCH), dtype=bool))
    assert np.array_equal(y[keep], x[keep])


def test_the_masks_differ_between_images():
    x = _images(32)
    patches = _occluded_patches(x, corruption.occlude(x, 0.25, seed=1, condition="occ25"))
    assert len({p.tobytes() for p in patches}) > 24


def test_each_patch_is_chosen_about_equally_often():
    x = _images(2000)
    patches = _occluded_patches(x, corruption.occlude(x, 0.25, seed=2, condition="occ25"))
    rate = patches.mean(axis=0)
    assert np.all(np.abs(rate - 0.25) < 0.05), rate


def test_occlusion_is_deterministic_per_seed_and_condition():
    x = _images(16)
    a = corruption.occlude(x, 0.5, seed=5, condition="occ50")
    b = corruption.occlude(x.copy(), 0.5, seed=5, condition="occ50")
    assert np.array_equal(a, b)


def test_the_mask_does_not_depend_on_the_pixels():
    """The arm never enters: the mask is a function of (seed, condition, count) only."""
    a = _occluded_patches(_images(16, seed=1), corruption.occlude(
        _images(16, seed=1), 0.25, seed=5, condition="occ25"))
    b = _occluded_patches(_images(16, seed=9), corruption.occlude(
        _images(16, seed=9), 0.25, seed=5, condition="occ25"))
    assert np.array_equal(a, b)


def test_different_conditions_draw_different_masks():
    x = _images(16)
    a = corruption.occlude(x, 0.25, seed=5, condition="occ25")
    b = corruption.occlude(x, 0.25, seed=5, condition="another")
    assert not np.array_equal(a, b)


def test_different_seeds_draw_different_masks():
    x = _images(16)
    a = corruption.occlude(x, 0.25, seed=5, condition="occ25")
    b = corruption.occlude(x, 0.25, seed=6, condition="occ25")
    assert not np.array_equal(a, b)


def test_occlude_does_not_mutate_its_input_and_keeps_shape_and_dtype():
    x = _images(8, np.float64)
    before = x.copy()
    y = corruption.occlude(x, 0.5, seed=0, condition="occ50")
    assert np.array_equal(x, before)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert not np.shares_memory(x, y)


def test_a_fraction_of_zero_returns_a_copy():
    x = _images(4)
    y = corruption.occlude(x, 0.0, seed=0, condition="none")
    assert np.array_equal(x, y) and not np.shares_memory(x, y)


@pytest.mark.parametrize("fraction", [-0.1, 1.5, float("nan")])
def test_a_fraction_outside_the_unit_interval_is_refused(fraction):
    with pytest.raises(ValueError, match="fraction"):
        corruption.occlude(_images(2), fraction, seed=0, condition="x")


def test_an_image_that_is_not_a_whole_patch_grid_is_refused():
    with pytest.raises(ValueError, match="multiple"):
        corruption.occlude(np.zeros((2, 30, 30, 3), np.float32), 0.25, seed=0, condition="x")


def test_a_non_batch_array_is_refused():
    with pytest.raises(ValueError, match="N, H, W, C"):
        corruption.occlude(np.zeros((32, 32, 3), np.float32), 0.25, seed=0, condition="x")


def test_a_fraction_that_is_not_a_whole_patch_count_rounds_to_the_nearest():
    x = _images(4)
    y = corruption.occlude(x, 0.3, seed=0, condition="x")
    assert _occluded_patches(x, y).reshape(4, -1).sum(axis=1).tolist() == [round(0.3 * 64)] * 4


# ---------------------------------------------------------------------------
# add_noise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.05, 0.1, 0.2])
def test_the_noise_std_matches_sigma_before_clipping(sigma):
    """Mid-grey pixels stay far from the clip bounds at these sigmas (>= 2.5 sigma)."""
    x = _images(400, value=0.5)
    y = corruption.add_noise(x, sigma, seed=1, condition=f"noise{sigma}")
    delta = (y - x).astype(np.float64)
    if sigma <= 0.1:
        assert abs(delta.mean()) < 5 * sigma / np.sqrt(delta.size)
        assert abs(delta.std() - sigma) < 0.01 * sigma
    else:
        # sigma 0.2 clips at +-0.5 = 2.5 sigma, so the std is the clipped one: just below sigma.
        assert 0.97 * sigma < delta.std() < sigma


def test_the_raw_noise_is_gaussian_with_the_asked_std(monkeypatch):
    """Unclipped view of the same draw: sigma 0.2 on grey pixels, tails included."""
    monkeypatch.setattr(corruption, "_clip_unit", lambda values: values)
    x = _images(400, value=0.5)
    raw = corruption.add_noise(x, 0.2, seed=1, condition="n") - x
    assert abs(raw.std() - 0.2) < 0.002
    assert abs(raw.mean()) < 0.002


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_output_is_clipped_to_the_unit_interval(dtype):
    x = _images(64, dtype)
    y = corruption.add_noise(x, 0.5, seed=1, condition="big")
    assert y.min() >= 0.0 and y.max() <= 1.0
    assert (y == 0).any() and (y == 1).any(), "sigma 0.5 on [0.1, 0.9] must reach both bounds"


def test_noise_is_deterministic_per_seed_and_condition_and_arm_free():
    x = _images(16)
    a = corruption.add_noise(x, 0.1, seed=5, condition="noise0.1")
    b = corruption.add_noise(x.copy(), 0.1, seed=5, condition="noise0.1")
    assert np.array_equal(a, b)


def test_the_noise_does_not_depend_on_the_pixels(monkeypatch):
    monkeypatch.setattr(corruption, "_clip_unit", lambda values: values)
    a = corruption.add_noise(_images(8, value=0.5), 0.1, seed=5, condition="n") - 0.5
    b = corruption.add_noise(_images(8, value=0.4), 0.1, seed=5, condition="n") - 0.4
    assert np.allclose(a, b, atol=1e-6)


def test_different_conditions_and_seeds_draw_different_noise():
    x = _images(8)
    base = corruption.add_noise(x, 0.1, seed=5, condition="noise0.1")
    assert not np.array_equal(base, corruption.add_noise(x, 0.1, seed=5, condition="other"))
    assert not np.array_equal(base, corruption.add_noise(x, 0.1, seed=6, condition="noise0.1"))


def test_the_same_noise_reaches_float32_and_float64_inputs():
    x = _images(8, np.float64)
    wide = corruption.add_noise(x, 0.1, seed=2, condition="n")
    narrow = corruption.add_noise(x.astype(np.float32), 0.1, seed=2, condition="n")
    assert np.allclose(wide, narrow, atol=1e-6)


def test_add_noise_does_not_mutate_its_input_and_keeps_shape_and_dtype():
    x = _images(8)
    before = x.copy()
    y = corruption.add_noise(x, 0.1, seed=0, condition="n")
    assert np.array_equal(x, before)
    assert y.shape == x.shape and y.dtype == x.dtype


def test_sigma_zero_returns_the_input_values():
    x = _images(4)
    assert np.array_equal(corruption.add_noise(x, 0.0, seed=0, condition="n"), x)


@pytest.mark.parametrize("sigma", [-0.1, float("nan")])
def test_a_bad_sigma_is_refused(sigma):
    with pytest.raises(ValueError, match="sigma"):
        corruption.add_noise(_images(2), sigma, seed=0, condition="n")


def test_an_input_outside_the_unit_domain_is_refused():
    """Standardised images would be silently clipped to a different picture."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        corruption.add_noise(_images(2) * 3 - 1, 0.1, seed=0, condition="n")


# ---------------------------------------------------------------------------
# the condition table
# ---------------------------------------------------------------------------

def test_the_six_conditions_are_named_and_clean_is_the_identity():
    assert list(corruption.CONDITIONS) == [
        "clean", "occ25", "occ50", "noise0.05", "noise0.1", "noise0.2"]
    x = _images(4)
    assert np.array_equal(corruption.apply_condition(x, "clean", seed=0), x)


def test_apply_condition_uses_the_condition_name_as_the_key():
    x = _images(16)
    via_table = corruption.apply_condition(x, "occ25", seed=7)
    direct = corruption.occlude(x, 0.25, seed=7, condition="occ25")
    assert np.array_equal(via_table, direct)
    noisy = corruption.apply_condition(x, "noise0.1", seed=7)
    assert np.array_equal(noisy, corruption.add_noise(x, 0.1, seed=7, condition="noise0.1"))
    assert not np.array_equal(corruption.apply_condition(x, "occ50", seed=7),
                              corruption.occlude(x, 0.5, seed=7, condition="occ25"))


def test_an_unknown_condition_is_refused():
    with pytest.raises(KeyError, match="nosuch"):
        corruption.apply_condition(_images(2), "nosuch", seed=0)


def test_two_arms_see_identical_corrupted_inputs():
    """The function takes no arm; calling it per arm gives byte-identical inputs."""
    x = _images(32)
    per_arm = [{name: corruption.apply_condition(x, name, seed=11) for name in corruption.CONDITIONS}
               for _ in range(2)]
    for name in corruption.CONDITIONS:
        assert per_arm[0][name].tobytes() == per_arm[1][name].tobytes()


# ---------------------------------------------------------------------------
# import hygiene
# ---------------------------------------------------------------------------

def test_the_corruption_module_does_not_import_tensorflow():
    code = ("import sys, train.mean_shift.corruption; "
            "raise SystemExit(int('tensorflow' in sys.modules or 'keras' in sys.modules))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-500:]
