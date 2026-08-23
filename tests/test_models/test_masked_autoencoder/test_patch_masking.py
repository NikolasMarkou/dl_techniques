"""``PatchMasking`` on its own: the mask it REPORTS must be the mask it APPLIED.

Why this file exists
--------------------
``PatchMasking`` returns three things -- the masked image, a flat ``(B, P)`` mask
and the patch count -- and the MAE's loss is computed against the *reported*
mask while the encoder only ever sees the *applied* one. Nothing in the package
tied those two together: the existing suites drive the layer only through
``MaskedAutoencoder``, where an inverted, mis-ordered or transposed mask still
produces a plausible reconstruction loss. A layer that reports patch 3 as masked
while having actually blanked patch 1 would pass every end-to-end test here and
train against the wrong pixels.

So the arms below never assert on shape alone. They recover the applied mask
FROM THE OUTPUT PIXELS (with ``mask_value="zero"`` over strictly-positive inputs,
so a zeroed patch is unambiguous) and require exact set equality with the
returned mask, plus the exact achieved count -- ``int(P * mask_ratio)``, a
floor, not a rounding.

The row-major patch indexing pinned here is the same one
``test_the_loss_is_confined_to_masked_patches.py`` re-derives at the model level;
this file pins it at the layer, which is where it is decided.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.masked_autoencoder import PatchMasking

IMAGE_SIZE, PATCH_SIZE, CHANNELS = 32, 8, 3
GRID = IMAGE_SIZE // PATCH_SIZE          # 4
NUM_PATCHES = GRID * GRID                # 16
SEED = 20260823


def _positive_images(batch: int = 3) -> np.ndarray:
    """Strictly positive pixels, so "this patch is exactly 0" means "blanked"."""
    rng = np.random.default_rng(SEED)
    return rng.uniform(1.0, 2.0, (batch, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)).astype(
        "float32"
    )


def _applied_mask(original: np.ndarray, masked: np.ndarray) -> np.ndarray:
    """Recover the mask actually applied, per (batch, patch), from the pixels.

    Row-major over the patch grid, matching the layer's own reshape order.
    Returns a ``(B, P)`` float array of 1.0 (blanked) / 0.0 (untouched).
    """
    batch = original.shape[0]
    out = np.zeros((batch, NUM_PATCHES), dtype="float32")
    for b in range(batch):
        for p in range(NUM_PATCHES):
            r, c = (p // GRID) * PATCH_SIZE, (p % GRID) * PATCH_SIZE
            tile = masked[b, r:r + PATCH_SIZE, c:c + PATCH_SIZE, :]
            src = original[b, r:r + PATCH_SIZE, c:c + PATCH_SIZE, :]
            if np.all(tile == 0.0):
                out[b, p] = 1.0
            else:
                assert np.array_equal(tile, src), (
                    f"patch {p} of sample {b} is neither blanked nor untouched: "
                    f"max|delta| = {np.max(np.abs(tile - src)):.6e}. A visible "
                    f"patch must pass through bit-identically."
                )
    return out


class TestTheConstructorRefusesImpossibleGeometry:
    """The validation at ``patch_masking.py:28-30`` and the build-time divisibility check."""

    @pytest.mark.parametrize("patch_size", [0, -1, -16])
    def test_a_non_positive_patch_size_is_refused(self, patch_size):
        with pytest.raises(ValueError, match="patch_size must be positive"):
            PatchMasking(patch_size=patch_size)

    @pytest.mark.parametrize("mask_ratio", [-0.01, 1.01, 2.0])
    def test_a_mask_ratio_outside_the_unit_interval_is_refused(self, mask_ratio):
        with pytest.raises(ValueError, match=r"mask_ratio must be in \[0, 1\]"):
            PatchMasking(mask_ratio=mask_ratio)

    @pytest.mark.parametrize("mask_ratio", [0.0, 1.0])
    def test_the_closed_endpoints_are_legal(self, mask_ratio):
        """The control: the check is ``0 <= r <= 1``, so both ends must construct."""
        assert PatchMasking(mask_ratio=mask_ratio).mask_ratio == mask_ratio

    def test_an_indivisible_image_is_refused_at_build(self):
        layer = PatchMasking(patch_size=8, mask_ratio=0.5)
        with pytest.raises(ValueError, match="not divisible by patch_size"):
            layer.build((None, 30, 32, CHANNELS))


class TestTheAchievedRatio:
    """``num_masked`` is ``int(P * ratio)`` -- a FLOOR, and identical for every sample."""

    @pytest.mark.parametrize(
        "mask_ratio, expected",
        [
            (0.75, 12),   # 16 * 0.75  = 12.0  -> 12
            (0.5, 8),     # 16 * 0.5   =  8.0  ->  8
            (0.3, 4),     # 16 * 0.3   =  4.8  ->  4, NOT 5 -- floor, not round
            (0.99, 15),   # 16 * 0.99  = 15.84 -> 15, NOT 16
            (1.0, 16),
        ],
    )
    def test_the_masked_count_is_the_floor_of_the_request(self, mask_ratio, expected):
        keras.utils.set_random_seed(SEED)
        layer = PatchMasking(
            patch_size=PATCH_SIZE, mask_ratio=mask_ratio, mask_value="zero"
        )
        images = _positive_images()
        _, mask, num_patches = layer(images, training=True)

        assert int(num_patches) == NUM_PATCHES
        counts = np.asarray(ops.sum(mask, axis=-1))
        assert np.array_equal(counts, np.full(counts.shape, float(expected))), (
            f"mask_ratio={mask_ratio} over {NUM_PATCHES} patches masked "
            f"{counts.tolist()} patches per sample; the layer's own arithmetic is "
            f"int({NUM_PATCHES} * {mask_ratio}) = {expected}, applied identically "
            f"to every sample in the batch."
        )

    def test_a_zero_ratio_masks_nothing_and_is_the_identity(self):
        """The control for the whole file: ratio 0 must leave the image alone."""
        layer = PatchMasking(patch_size=PATCH_SIZE, mask_ratio=0.0, mask_value="zero")
        images = _positive_images()
        masked, mask, _ = layer(images, training=True)

        assert float(ops.sum(mask)) == 0.0
        assert np.max(np.abs(np.asarray(masked) - images)) == 0.0


class TestTheReportedMaskIsTheAppliedMask:
    """The claim this file exists for."""

    @pytest.mark.parametrize("mask_ratio", [0.25, 0.5, 0.75])
    def test_the_blanked_patches_are_exactly_the_reported_ones(self, mask_ratio):
        keras.utils.set_random_seed(SEED)
        layer = PatchMasking(
            patch_size=PATCH_SIZE, mask_ratio=mask_ratio, mask_value="zero"
        )
        images = _positive_images()
        masked, reported, _ = layer(images, training=True)

        reported = np.asarray(reported)
        applied = _applied_mask(images, np.asarray(masked))

        assert np.array_equal(applied, reported), (
            "the mask the layer RETURNS is not the mask it APPLIED. Reported "
            f"indices {np.argwhere(reported == 1.0).tolist()[:8]}, blanked "
            f"indices {np.argwhere(applied == 1.0).tolist()[:8]}. The MAE loss "
            "reads the returned mask, so a mismatch trains against the wrong "
            "pixels while every end-to-end reconstruction test stays green."
        )

    def test_different_samples_get_different_masks(self):
        """Anti-vacuity: a per-batch-broadcast mask would satisfy the arm above."""
        keras.utils.set_random_seed(SEED)
        layer = PatchMasking(patch_size=PATCH_SIZE, mask_ratio=0.5, mask_value="zero")
        _, mask, _ = layer(_positive_images(batch=8), training=True)
        rows = {tuple(row) for row in np.asarray(mask).tolist()}
        assert len(rows) > 1, (
            "every sample in the batch drew the identical mask; the noise is "
            "being broadcast instead of sampled per sample."
        )


class TestInferenceIsNotMasking:
    """``training=False`` must be the identity -- the MAE relies on it for eval."""

    def test_inference_returns_the_input_bit_identically(self):
        layer = PatchMasking(patch_size=PATCH_SIZE, mask_ratio=0.75, mask_value="zero")
        images = _positive_images()
        masked, mask, _ = layer(images, training=False)

        assert float(ops.sum(mask)) == 0.0
        delta = float(np.max(np.abs(np.asarray(masked) - images)))
        assert delta == 0.0, f"inference perturbed the image by {delta:.6e}"

    def test_training_is_not_the_identity(self):
        """The paired 'something changed' arm."""
        keras.utils.set_random_seed(SEED)
        layer = PatchMasking(patch_size=PATCH_SIZE, mask_ratio=0.75, mask_value="zero")
        images = _positive_images()
        masked, _, _ = layer(images, training=True)
        assert float(np.max(np.abs(np.asarray(masked) - images))) > 0.0


class TestTheMaskValuePolicies:
    """``learnable`` / ``zero`` / ``noise`` / a constant are four different fills."""

    def test_the_learnable_token_is_a_trainable_weight_of_the_right_shape(self):
        layer = PatchMasking(
            patch_size=PATCH_SIZE, mask_ratio=0.5, mask_value="learnable"
        )
        layer.build((None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        assert layer.mask_token is not None
        assert tuple(layer.mask_token.shape) == (1, PATCH_SIZE, PATCH_SIZE, CHANNELS)
        assert layer.mask_token.trainable
        assert len(layer.trainable_weights) == 1

    def test_a_fixed_mask_value_allocates_no_weight(self):
        """The control: only ``learnable`` may add a parameter."""
        for value in ("zero", "noise", 0.5):
            layer = PatchMasking(
                patch_size=PATCH_SIZE, mask_ratio=0.5, mask_value=value
            )
            layer.build((None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
            assert layer.mask_token is None, value
            assert layer.weights == [], value

    def test_the_learnable_token_is_what_lands_in_the_masked_patches(self):
        """Perturb the token off its zero init; the blanked tiles must BE it."""
        keras.utils.set_random_seed(SEED)
        layer = PatchMasking(
            patch_size=PATCH_SIZE, mask_ratio=0.5, mask_value="learnable"
        )
        layer.build((None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        rng = np.random.default_rng(SEED)
        token = rng.normal(
            0.0, 1.0, (1, PATCH_SIZE, PATCH_SIZE, CHANNELS)
        ).astype("float32")
        layer.mask_token.assign(token)

        images = _positive_images(batch=2)
        masked, mask, _ = layer(images, training=True)
        masked, mask = np.asarray(masked), np.asarray(mask)

        checked = 0
        for b in range(images.shape[0]):
            for p in range(NUM_PATCHES):
                if mask[b, p] != 1.0:
                    continue
                r, c = (p // GRID) * PATCH_SIZE, (p % GRID) * PATCH_SIZE
                tile = masked[b, r:r + PATCH_SIZE, c:c + PATCH_SIZE, :]
                assert np.max(np.abs(tile - token[0])) == 0.0, (
                    f"masked patch {p} of sample {b} does not hold the mask "
                    f"token; max|delta| = {np.max(np.abs(tile - token[0])):.6e}"
                )
                checked += 1
        assert checked == images.shape[0] * (NUM_PATCHES // 2), checked

    def test_a_constant_mask_value_fills_with_that_constant(self):
        keras.utils.set_random_seed(SEED)
        layer = PatchMasking(patch_size=PATCH_SIZE, mask_ratio=0.5, mask_value=-3.5)
        images = _positive_images(batch=1)
        masked, mask, _ = layer(images, training=True)
        masked, mask = np.asarray(masked), np.asarray(mask)

        for p in np.flatnonzero(mask[0] == 1.0):
            r, c = (p // GRID) * PATCH_SIZE, (p % GRID) * PATCH_SIZE
            tile = masked[0, r:r + PATCH_SIZE, c:c + PATCH_SIZE, :]
            assert np.all(tile == -3.5), tile.ravel()[:4]


class TestSerialization:
    def test_the_config_round_trips(self):
        layer = PatchMasking(patch_size=4, mask_ratio=0.6, mask_value="noise")
        clone = PatchMasking.from_config(layer.get_config())
        assert (clone.patch_size, clone.mask_ratio, clone.mask_value) == (
            4,
            0.6,
            "noise",
        )

    def test_the_output_shape_matches_the_input_shape(self):
        layer = PatchMasking(patch_size=PATCH_SIZE, mask_ratio=0.75)
        shape = (None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        assert layer.compute_output_shape(shape) == shape
