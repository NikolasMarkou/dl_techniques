"""Segmentation transforms: the augmentation moves an image and its mask together, and invents no class.

The images here are built so that the colour of a pixel IS its class (``PALETTE[mask]``, void
grey), so after any transform the class a pixel's colour names must equal the mask at that
pixel: if the image and the mask were augmented independently, with two draws or with one op on one
of them, that equality breaks within a few seeds. The claims:

* over 200 seeds and both presets, a mask that is a function of the image stays that function, the
  mask values stay a subset of the originals (void included) and the constant mask of a class
  stays constant (nothing is padded, so no pixel turns into class 0);
* the presets really move pixels (all 2 flips and all 8 dihedral variants appear), ``none`` is no
  hook, ``dihedral`` is refused on a non-square size and ``flip`` is not;
* through the pipeline, the fit split is augmented and keeps the function, the selection and test
  splits are untouched, two builds of one seed give the same epochs and the next epoch differs;
* the resize is bilinear for the image and nearest for the mask: an upsampled gradient has
  intermediate values, an upsampled mask has none;
* ``SegTransformArgs`` takes an int or a pair, refuses the rest by dotted path, fills ``"auto"`` from
  the recipe, refuses dihedral on a non-square size whether the size or the preset came from the
  recipe, and leaves a field the recipe lacks as ``"auto"`` for ``assert_resolved`` to report.
"""

from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf

import train.common.segmentation_data as sd
from train.api import segmentation_args as sa
from train.api.args import AutoContext
from train.api.config import assert_resolved, validate
from train.api.errors import ConfigError, ConfigValidationError, UnresolvedAutoError

from tests.test_train.test_api._segmentation_fixtures import (
    PALETTE,
    blocks,
    class_of,
    listing_of,
    load,
    picture,
    write_pair,
)

VOID = 255


def void_class_of(image: np.ndarray) -> np.ndarray:
    """The class a decoded pixel names, with the grey of void mapped back to 255."""
    named = class_of(np.round(image * 255).astype(np.uint8))
    return np.where(named == len(PALETTE), VOID, named)


def asymmetric_mask(height: int, width: int, seed: int) -> np.ndarray:
    """A mask of 4x4-pixel blocks that none of the 8 symmetries of the square (or 2 of the rectangle) maps onto itself."""
    rng = np.random.default_rng(seed)
    for _ in range(100):
        small = rng.integers(0, 3, size=(height // 4, width // 4))
        mask = np.kron(small, np.ones((4, 4), np.int64)).astype(np.int32)
        variants = [mask[:, ::-1], mask[::-1, :], mask[::-1, ::-1]]
        if height == width:
            variants += [mask.T, mask.T[::-1], mask.T[:, ::-1], mask.T[::-1, ::-1]]
        if not any(np.array_equal(mask, v) for v in variants):
            return mask
    raise AssertionError("no asymmetric fixture found")


def make_pair(height: int, width: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    mask = asymmetric_mask(height, width, seed)
    mask[:4, :4] = VOID  # void in a corner, so a transform that moves it must move its colour
    return picture(np.where(mask == VOID, 9, mask).astype(np.uint8)).astype(np.float32) / 255.0, mask


def apply(hook, image: np.ndarray, mask: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    out_image, out_mask = hook(tf.constant([seed, seed + 1], tf.int64), tf.constant(image), tf.constant(mask))
    return out_image.numpy(), out_mask.numpy()


# ---------------------------------------------------------------------
# The hook
# ---------------------------------------------------------------------

@pytest.mark.parametrize("preset,size", [("flip", (16, 16)), ("flip", (16, 24)), ("dihedral", (16, 16))])
def test_a_mask_that_is_a_function_of_the_image_stays_that_function(preset, size):
    hook = sd.build_segmentation_augment(preset, size)
    image, mask = make_pair(*size, seed=1)
    assert np.array_equal(void_class_of(image), mask), "the fixture must start as a function"
    originals = set(np.unique(mask))
    for seed in range(200):
        out_image, out_mask = apply(hook, image, mask, seed)
        assert out_image.shape == image.shape and out_mask.shape == mask.shape
        assert out_mask.dtype == np.int32 and out_image.dtype == np.float32
        assert np.array_equal(void_class_of(out_image), out_mask), f"seed {seed}: image and mask moved apart"
        assert set(np.unique(out_mask)) <= originals


@pytest.mark.parametrize("preset,size,variants", [("flip", (16, 24), 2), ("dihedral", (16, 16), 8)])
def test_the_presets_really_move_pixels(preset, size, variants):
    hook = sd.build_segmentation_augment(preset, size)
    image, mask = make_pair(*size, seed=2)
    seen = {apply(hook, image, mask, seed)[1].tobytes() for seed in range(200)}
    assert len(seen) == variants, f"{preset} should produce {variants} distinct arrangements, got {len(seen)}"


@pytest.mark.parametrize("preset,size", [("flip", (16, 24)), ("dihedral", (16, 16))])
def test_nothing_is_padded_so_a_constant_mask_stays_constant(preset, size):
    hook = sd.build_segmentation_augment(preset, size)
    image = np.full((*size, 3), 0.5, np.float32)
    for klass in (1, 2, VOID):
        mask = np.full(size, klass, np.int32)
        for seed in range(50):
            _, out_mask = apply(hook, image, mask, seed)
            assert np.all(out_mask == klass), f"class {klass} became {np.unique(out_mask)} at seed {seed}"


def test_the_draw_is_a_function_of_the_seed_alone():
    hook = sd.build_segmentation_augment("dihedral", (16, 16))
    image, mask = make_pair(16, 16, seed=3)
    for seed in (0, 7, 99):
        a, b = apply(hook, image, mask, seed), apply(hook, image, mask, seed)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_the_preset_table():
    assert sd.SEG_AUGMENTS == ("none", "flip", "dihedral")
    assert sd.build_segmentation_augment("none", (16, 24)) is None
    for preset in ("flip", "dihedral"):
        hook = sd.build_segmentation_augment(preset, (16, 16))
        assert callable(hook) and getattr(hook, "seeded", False)
    assert sd.build_segmentation_augment("flip", (16, 48)) is not None
    with pytest.raises(ValueError, match=r"swaps the sides of a 16x24 .*use flip"):
        sd.build_segmentation_augment("dihedral", (16, 24))
    with pytest.raises(ValueError, match="Unknown segmentation augmentation 'rotate'"):
        sd.build_segmentation_augment("rotate", (16, 16))


# ---------------------------------------------------------------------
# Through the pipeline
# ---------------------------------------------------------------------

def epochs(data: sd.SegmentationData, count: int = 2):
    out = []
    for _ in range(count):
        out.extend(data.train_ds.as_numpy_iterator())
    return out


@pytest.mark.parametrize("preset,size", [("flip", (16, 24)), ("dihedral", (16, 16))])
def test_the_fit_split_is_augmented_and_keeps_the_function_the_others_are_untouched(tmp_path, preset, size):
    listing = listing_of(tmp_path, count=12, size=size)  # source size = target size: no resize blur
    plain = load(listing, size=size, seed=3)
    augmented = load(listing, size=size, seed=3, augment=sd.build_segmentation_augment(preset, size))
    moved = 0
    for (image, mask), (plain_image, plain_mask) in zip(epochs(augmented), epochs(plain)):
        assert image.shape == plain_image.shape
        for one_image, one_mask in zip(image, mask):
            assert np.array_equal(class_of(np.round(one_image * 255).astype(np.uint8)), one_mask)
        moved += int(not np.array_equal(mask, plain_mask))
    assert moved > 0, "the augmentation changed no batch"
    for split in ("select_ds",):
        for (a_image, a_mask), (p_image, p_mask) in zip(getattr(augmented, split), getattr(plain, split)):
            assert np.array_equal(a_image.numpy(), p_image.numpy()) and np.array_equal(a_mask.numpy(), p_mask.numpy())


def test_two_builds_of_one_seed_give_the_same_augmented_epochs_and_the_next_epoch_differs(tmp_path):
    listing = listing_of(tmp_path, count=12, size=(16, 16))
    hook = lambda: sd.build_segmentation_augment("dihedral", (16, 16))
    one = epochs(load(listing, size=(16, 16), seed=5, augment=hook()), 2)
    two = epochs(load(listing, size=(16, 16), seed=5, augment=hook()), 2)
    assert len(one) == len(two) == 8
    assert all(np.array_equal(a[1], b[1]) and np.array_equal(a[0], b[0]) for a, b in zip(one, two))
    assert not all(np.array_equal(a[1], b[1]) for a, b in zip(one[:4], one[4:])), "an epoch repeated another"


def test_void_pixels_survive_the_pipeline_augmentation(tmp_path):
    images, masks = tmp_path / "i", tmp_path / "m"
    for index in range(12):
        mask = blocks(16, 16, classes=3, seed=index)
        mask[:8, :8] = VOID
        write_pair(images, masks, f"p{index:02d}", mask)
    listing = sd.scan_segmentation_folder(str(images), str(masks))
    data = load(listing, size=(16, 16), ignore_index=VOID,
                augment=sd.build_segmentation_augment("dihedral", (16, 16)))
    for _, mask in data.train_ds.as_numpy_iterator():
        assert np.all((mask == VOID).sum(axis=(1, 2)) == 64), "the 8x8 void block keeps its 64 pixels"
        assert set(np.unique(mask)) <= {0, 1, 2, VOID}


# ---------------------------------------------------------------------
# The resize
# ---------------------------------------------------------------------

def test_the_image_is_resized_bilinearly_and_the_mask_by_nearest_neighbour():
    gradient = np.tile(np.linspace(0, 255, 4).astype(np.uint8), (4, 1))
    image = np.stack([gradient] * 3, -1)
    mask = np.tile(np.asarray([0, 1, 2, VOID], np.int32), (4, 1))
    out_image, out_mask = sd.resize_pair(tf.constant(image), tf.constant(mask), (8, 16))
    assert out_image.shape == (8, 16, 3) and out_mask.shape == (8, 16)
    assert len(np.unique(np.round(out_image.numpy() * 255))) > 4, "bilinear makes intermediate grey levels"
    assert set(np.unique(out_mask.numpy())) == {0, 1, 2, VOID}, "nearest neighbour makes no new class"
    down_image, down_mask = sd.resize_pair(tf.constant(np.repeat(np.repeat(image, 2, 0), 2, 1)),
                                           tf.constant(np.repeat(np.repeat(mask, 2, 0), 2, 1)), (4, 4))
    assert set(np.unique(down_mask.numpy())) <= {0, 1, 2, VOID}


def test_the_inputs_are_in_unit_range_and_not_standardized(tmp_path):
    data = load(listing_of(tmp_path, count=12))
    assert np.all(data.mean == 0) and np.all(data.std == 1)
    image, _ = next(iter(data.train_ds))
    assert 0.0 <= float(tf.reduce_min(image)) and float(tf.reduce_max(image)) <= 1.0


# ---------------------------------------------------------------------
# The transform group
# ---------------------------------------------------------------------

def context(recipe=None):
    return AutoContext(family="convunext", variant="tiny", dataset="oxford_pet", num_classes=3,
                       image_size=0, defaults={"transform_args": recipe} if recipe is not None else {})


@pytest.mark.parametrize("value", [64, [128, 256], [32, 32]])
def test_a_size_is_an_int_or_a_pair(value):
    assert sa.SegTransformArgs(image_size=value).problems("transform_args") == []
    assert validate(sa.SegTransformArgs, {"image_size": value}).image_size == value


@pytest.mark.parametrize("value", [0, -4, [1, 2, 3], [64], [0, 5], [5, -1], True, "big", [64.5, 64]])
def test_a_bad_size_is_refused_by_dotted_path(value):
    with pytest.raises(ConfigValidationError) as caught:
        validate(sa.SegTransformArgs, {"image_size": value})
    assert "image_size" in "\n".join(caught.value.errors)
    if not isinstance(value, str) and value is not True and value != [64.5, 64]:
        assert "expected one positive int (a square) or [height, width] of two" in "\n".join(caught.value.errors)


@pytest.mark.parametrize("value", [True, [True, 8], [8, False], 0, [8], (8, 8, 8), 2.5])
def test_a_size_built_without_validate_is_held_to_the_same_rule(value):
    lines = sa.SegTransformArgs(image_size=value).problems("transform_args")
    assert len(lines) == 1 and "transform_args.image_size" in lines[0]


def test_auto_takes_the_recipe_and_an_explicit_value_is_untouched():
    recipe = {"image_size": 128, "augment": "flip"}
    resolved = sa.SegTransformArgs().resolve_auto(context(recipe))
    assert (resolved.image_size, resolved.augment) == (128, "flip")
    explicit = sa.SegTransformArgs(image_size=[64, 128], augment="none").resolve_auto(context(recipe))
    assert (explicit.image_size, explicit.augment) == ([64, 128], "none")
    equal = sa.SegTransformArgs(image_size=128, augment="flip").resolve_auto(context(recipe))
    assert (equal.image_size, equal.augment) == (128, "flip")


def test_a_field_the_recipe_lacks_stays_auto_and_assert_resolved_names_it():
    resolved = sa.SegTransformArgs().resolve_auto(context({"augment": "flip"}))
    assert resolved.image_size == "auto" and resolved.augment == "flip"
    with pytest.raises(UnresolvedAutoError, match="image_size"):
        assert_resolved(resolved)


def test_a_recipe_with_a_key_that_is_no_field_is_refused():
    with pytest.raises(ConfigValidationError, match="Family recipe for 'transform_args'"):
        sa.SegTransformArgs().resolve_auto(context({"normalize": "unit"}))


def test_an_unknown_augment_is_refused_whoever_gave_it():
    with pytest.raises(ConfigError, match=r"'transform_args\.augment': 'rotate' is not a segmentation preset"):
        sa.SegTransformArgs(augment="rotate").resolve_auto(context())
    with pytest.raises(ConfigError, match="not a segmentation preset"):
        sa.SegTransformArgs().resolve_auto(context({"augment": "autoaugment"}))


@pytest.mark.parametrize("given,recipe", [
    ({"augment": "dihedral", "image_size": [128, 256]}, {}),
    ({"augment": "dihedral"}, {"image_size": [128, 256]}),
    ({"image_size": [128, 256]}, {"augment": "dihedral"}),
    ({}, {"augment": "dihedral", "image_size": [96, 128]}),
])
def test_dihedral_is_refused_on_a_non_square_size_wherever_either_came_from(given, recipe):
    with pytest.raises(ConfigError, match=r"'transform_args\.augment': dihedral .*128x256|96x128|swaps the sides"):
        sa.SegTransformArgs(**given).resolve_auto(context(recipe))


def test_dihedral_on_a_square_size_and_flip_on_a_pair_are_fine():
    assert sa.SegTransformArgs(augment="dihedral", image_size=96).resolve_auto(context()).augment == "dihedral"
    assert sa.SegTransformArgs(augment="dihedral", image_size=[96, 96]).resolve_auto(context()).augment == "dihedral"
    assert sa.SegTransformArgs(augment="flip", image_size=[96, 128]).resolve_auto(context()).augment == "flip"


def test_the_group_has_no_normalize_key():
    with pytest.raises(ConfigValidationError, match="normalize"):
        validate(sa.SegTransformArgs, {"normalize": "unit"})
    assert [field for field in sa.SegTransformArgs.__dataclass_fields__] == ["image_size", "augment"]
