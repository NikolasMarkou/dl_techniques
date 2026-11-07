"""``train.common.image_augment``: the seeded, size-agnostic augmentation every trainer shares.

The operations used to live (unseeded and hard-coded to 32 px) in ``train_cliffordnet``, with
siblings in vit, alexnet, the SOM and the CLIP paths. Nothing pinned the ops themselves, so
these tests do: the AutoAugment table, magnitude mapping and fill against LITERALS copied from
torchvision's ``AutoAugment(AutoAugmentPolicy.CIFAR10)`` (D-040: ``transforms/autoaugment.py``,
sha256 11b61381b9aff782f45bdc408f8453e21a2cd8e013a1bcb46039bf5b70c0d7e5, whose 25 rows are Table 7
of Cubuk et al., arXiv:1805.09501), the erasing geometry, the photometric ranges, determinism per
seed, and the presets.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

from dl_techniques.datasets.vision import augment_hook  # noqa: E402
from train.common import image_augment as ia  # noqa: E402

SEED = tf.constant([11, 22], dtype=tf.int32)
OTHER = tf.constant([33, 44], dtype=tf.int32)


def _noise(h=32, w=32, c=3, seed=0) -> tf.Tensor:
    return tf.constant(np.random.default_rng(seed).random((h, w, c)), tf.float32)


def _same(a, b) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


class _Positive:
    """A stand-in generator whose coin always lands on the positive sign."""

    def random(self):
        return 0.0


class _Negative:
    """A stand-in generator whose coin always lands on the negative sign (``random() > 0.5``)."""

    def random(self):
        return 0.9


class _Scripted:
    """A generator that replays a list of draws: ``integers`` pops the sub-policy, ``random`` the rest."""

    def __init__(self, row, *draws):
        self.row, self.draws = row, list(draws)

    def integers(self, n):
        return self.row

    def random(self):
        return self.draws.pop(0)


# ---------------------------------------------------------------------
# AutoAugment
# ---------------------------------------------------------------------


# torchvision's ``AutoAugmentPolicy.CIFAR10`` (``_get_policies``), copied as a literal: the same 25
# rows as Table 7 of arXiv:1805.09501, an operation without a magnitude carrying ``None``.
TORCHVISION_CIFAR10_POLICY = (
    (("Invert", 0.1, None), ("Contrast", 0.2, 6)),
    (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
    (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
    (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
    (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
    (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
    (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
    (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
    (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
    (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
    (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
    (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
    (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
    (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
    (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
    (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
    (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
    (("Color", 0.9, 9), ("Equalize", 0.6, None)),
    (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
    (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
    (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
    (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
    (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
    (("Equalize", 0.8, None), ("Invert", 0.1, None)),
    (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
)

# torchvision's ``_augmentation_space(10, (height=32, width=48))``, printed from torchvision 0.29.2
# (float32 tensors, nine significant digits): (magnitude of bin 0 ... bin 9, signed).
TORCHVISION_SPACE_H32_W48 = {
    "ShearX": ((0, 0.0333333351, 0.0666666701, 0.100000009, 0.13333334, 0.166666672, 0.200000003, 0.233333349, 0.266666681, 0.300000012), True),
    "ShearY": ((0, 0.0333333351, 0.0666666701, 0.100000009, 0.13333334, 0.166666672, 0.200000003, 0.233333349, 0.266666681, 0.300000012), True),
    "TranslateX": ((0, 2.41691852, 4.83383703, 7.25075531, 9.66767406, 12.0845919, 14.5015106, 16.9184284, 19.3353481, 21.7522659), True),
    "TranslateY": ((0, 1.61127901, 3.22255802, 4.83383703, 6.44511604, 8.05639458, 9.66767311, 11.2789526, 12.8902321, 14.5015106), True),
    "Rotate": ((0, 3.33333325, 6.66666651, 10, 13.333333, 16.6666679, 20, 23.333334, 26.666666, 30), True),
    "Brightness": ((0, 0.099999994, 0.199999988, 0.299999982, 0.399999976, 0.5, 0.600000024, 0.699999988, 0.799999952, 0.899999976), True),
    "Color": ((0, 0.099999994, 0.199999988, 0.299999982, 0.399999976, 0.5, 0.600000024, 0.699999988, 0.799999952, 0.899999976), True),
    "Contrast": ((0, 0.099999994, 0.199999988, 0.299999982, 0.399999976, 0.5, 0.600000024, 0.699999988, 0.799999952, 0.899999976), True),
    "Sharpness": ((0, 0.099999994, 0.199999988, 0.299999982, 0.399999976, 0.5, 0.600000024, 0.699999988, 0.799999952, 0.899999976), True),
    "Posterize": ((8, 8, 7, 7, 6, 6, 5, 5, 4, 4), False),
    "Solarize": ((255, 226.666672, 198.333328, 170, 141.666656, 113.333336, 85, 56.6666679, 28.333334, 0), False),
}


def test_the_cifar10_policy_is_torchvisions_25_rows_and_the_bins_are_10() -> None:
    assert ia.CIFAR10_POLICY == TORCHVISION_CIFAR10_POLICY
    assert ia.NUM_MAGNITUDE_BINS == 10
    assert ia.AUTOAUGMENT_POLICIES["cifar10"] is ia.CIFAR10_POLICY
    no_magnitude = {"Invert", "AutoContrast", "Equalize"}
    for sub in ia.CIFAR10_POLICY:
        assert len(sub) == 2
        for name, probability, magnitude in sub:
            assert 0.0 <= probability <= 1.0
            if name in no_magnitude:
                assert magnitude is None, "an operation without a magnitude carries None"
            else:
                assert 0 <= magnitude < ia.NUM_MAGNITUDE_BINS


def test_the_magnitude_of_every_operation_at_every_bin_is_torchvisions() -> None:
    space = ia._augmentation_space(width=48, height=32)
    assert set(space) == set(TORCHVISION_SPACE_H32_W48)
    for name, (expected, signed) in TORCHVISION_SPACE_H32_W48.items():
        magnitudes, got_signed = space[name]
        assert got_signed is signed, name
        assert len(magnitudes) == 10, name
        # torchvision's tensors are float32: agree to its precision.
        np.testing.assert_allclose(magnitudes, expected, rtol=2e-7, atol=1e-7, err_msg=name)


def test_translation_is_a_fraction_of_the_side_of_its_own_axis_and_truncated() -> None:
    # TranslateX takes the WIDTH and TranslateY the HEIGHT (torchvision: image_size[1], image_size[0]),
    # and the pixel count is int(magnitude), truncated towards zero, never rounded. A positive
    # magnitude moves content towards larger x or y, as torchvision's does; what leaves is lost.
    width, height = 64, 16
    columns = (20, 40)
    pixels = np.zeros((height, width, 3), np.uint8)
    pixels[:, list(columns)] = 255
    image = Image.fromarray(pixels)
    for bin_index, moved in ((9, 29), (8, 25), (1, 3)):
        assert int(150.0 / 331.0 * width * bin_index / 9) == moved
        out = np.asarray(ia._apply_autoaugment_op(image, "TranslateX", bin_index, _Positive()))
        assert np.flatnonzero(out[0, :, 0] == 255).tolist() == [
            c + moved for c in columns if c + moved < width]
        assert (out[0, :moved, 0] == 0).all(), "the uncovered strip is black"
        out = np.asarray(ia._apply_autoaugment_op(image, "TranslateX", bin_index, _Negative()))
        assert np.flatnonzero(out[0, :, 0] == 255).tolist() == [
            c - moved for c in columns if c - moved >= 0]
    rows = (4, 11)
    pixels = np.zeros((height, width, 3), np.uint8)
    pixels[list(rows), :] = 255
    image = Image.fromarray(pixels)
    moved = int(150.0 / 331.0 * height)
    assert moved == 7, "the height, not the width (which would give 29)"
    out = np.asarray(ia._apply_autoaugment_op(image, "TranslateY", 9, _Positive()))
    assert np.flatnonzero(out[:, 0, 0] == 255).tolist() == [r + moved for r in rows if r + moved < height]
    out = np.asarray(ia._apply_autoaugment_op(image, "TranslateY", 9, _Negative()))
    assert np.flatnonzero(out[:, 0, 0] == 255).tolist() == [r - moved for r in rows if r - moved >= 0]


def test_solarize_and_posterize_follow_torchvisions_magnitude_mapping() -> None:
    original = np.arange(256, dtype=np.uint8).reshape(16, 16)
    ramp = Image.fromarray(original)
    # bin 5 of 10: the threshold is 255 * (1 - 5 / 9) = 113.33, a float: pixel 113 is below it and kept.
    solar = np.asarray(ia._apply_autoaugment_op(ramp, "Solarize", 5, _Positive()))
    np.testing.assert_array_equal(solar[original <= 113], original[original <= 113])
    np.testing.assert_array_equal(solar[original >= 114], 255 - original[original >= 114])
    # bin 0 is the threshold 255: only the pixel 255 is inverted (PIL's ``i < threshold`` is kept).
    first = np.asarray(ia._apply_autoaugment_op(ramp, "Solarize", 0, _Positive()))
    np.testing.assert_array_equal(first[original < 255], original[original < 255])
    assert first[original == 255].tolist() == [0]
    # Posterize keeps 8 - round(bin / 2.25) bits: 8, 8, 7, 7, 6, 6, 5, 5, 4, 4.
    for bin_index, bits in enumerate((8, 8, 7, 7, 6, 6, 5, 5, 4, 4)):
        post = np.asarray(ia._apply_autoaugment_op(ramp, "Posterize", bin_index, _Positive()))
        assert len(np.unique(post)) == 2 ** bits, bin_index


def test_what_a_rotation_shear_or_translation_uncovers_is_black() -> None:
    constant = np.full((32, 32, 3), 200, np.uint8)
    assert ia._AUTOAUGMENT_FILL == 0
    for name in ("Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY"):
        for coin in (_Positive(), _Negative()):
            out = np.asarray(ia._apply_autoaugment_op(Image.fromarray(constant), name, 9, coin))
            assert set(np.unique(out)) == {0, 200}, f"{name}: {np.unique(out)}"
            assert (out == 0).all(axis=-1).any(), f"{name} uncovered nothing"
    grey = np.asarray(ia._apply_autoaugment_op(
        Image.fromarray(constant[..., 0]), "Rotate", 9, _Positive()))
    assert grey[0, 0] == 0 and grey[16, 16] == 200, "an L image is filled with 0 as well"


def test_rotation_turns_counter_clockwise_by_the_binned_degrees() -> None:
    pixels = np.zeros((33, 33), np.uint8)
    pixels[16, 25] = 255            # right of the centre
    out = np.asarray(ia._apply_autoaugment_op(Image.fromarray(pixels), "Rotate", 9, _Positive()))
    row, col = np.argwhere(out == 255)[0]
    assert row < 16 and col < 25, "30 degrees counter-clockwise lifts a point on the right"
    out = np.asarray(ia._apply_autoaugment_op(Image.fromarray(pixels), "Rotate", 9, _Negative()))
    row, col = np.argwhere(out == 255)[0]
    assert row > 16 and col < 25


def _where(op_name, coin, row=20, col=16, size=33):
    """``(row, col)`` of the one lit pixel after ``op_name`` at bin 9 (magnitude 0.3 for a shear), or ``None`` when it left the image."""
    pixels = np.zeros((size, size), np.uint8)
    pixels[row, col] = 255
    out = np.asarray(ia._apply_autoaugment_op(Image.fromarray(pixels), op_name, 9, coin))
    lit = np.argwhere(out == 255)
    return None if len(lit) == 0 else tuple(int(v) for v in lit[0])


def test_shear_x_moves_a_point_along_the_row_and_shear_y_along_the_column() -> None:
    """Which axis a shear acts on, and in which direction (review of iteration 2, WARNING 2: swapping the two affine matrices
    survived every test, and the CIFAR-10 policy holds ShearY rows only).  PIL samples the input at ``(x + m*y, y)`` for ShearX and
    at ``(x, m*x + y)`` for ShearY, so the point at (row 20, col 16) lands at col ``16 - 0.3 * 20 = 10`` with its row kept
    (ShearX), or at row ``20 - 0.3 * 16 = 15.2 -> 15`` with its column kept (ShearY); the negative sign mirrors both."""
    assert _where("ShearX", _Positive()) == (20, 10)
    assert _where("ShearX", _Negative()) == (20, 22)
    assert _where("ShearY", _Positive()) == (15, 16)
    assert _where("ShearY", _Negative()) == (25, 16)


class _Counting:
    """A seeded generator that remembers every ``random()`` draw, so the sign an operation took can be read back."""

    def __init__(self, seed):
        self.rng, self.draws = np.random.default_rng(seed), []

    def random(self):
        self.draws.append(self.rng.random())
        return self.draws[-1]


def test_a_signed_operation_goes_negative_half_of_the_time() -> None:
    """torchvision flips the sign with probability 0.5 (``torch.randint(2)``); a biased coin bends every rotation, shear and
    translation of the policy one way (review of iteration 2, WARNING 2: a coin at 0.2 survived).  4,000 draws have a standard
    deviation of 0.0079 on the share, so the bound 0.5 +- 0.04 is five of them."""
    coin = _Counting(12345)
    went_down = 0
    for _ in range(4000):
        pixels = np.zeros((33, 33), np.uint8)
        pixels[16, 25] = 255
        out = np.asarray(ia._apply_autoaugment_op(Image.fromarray(pixels), "Rotate", 9, coin))
        went_down += int(np.argwhere(out == 255)[0][0] > 16)    # a negative angle turns clockwise: a point on the right goes down
    assert len(coin.draws) == 4000, "one coin per signed operation"
    assert went_down == sum(1 for draw in coin.draws if draw > 0.5), "the sign is the coin's"
    assert abs(went_down / 4000 - 0.5) < 0.04, went_down / 4000


def test_an_operation_is_applied_when_its_draw_is_at_most_its_probability() -> None:
    # torchvision: ``if probs[i] <= p``. A draw equal to the probability applies; above it skips.
    policy = ((("Invert", 0.5, None), ("Rotate", 0.5, 9)),)
    image = np.full((8, 8, 3), 0.25, np.float32)
    out = ia._autoaugment_array(image, policy, _Scripted(0, 0.5, 0.5001))
    np.testing.assert_allclose(out, 0.75, atol=1 / 255)
    out = ia._autoaugment_array(image, policy, _Scripted(0, 0.5001, 0.5001))
    np.testing.assert_allclose(out, 0.25, atol=1 / 255)
    # a signed operation draws its sign after its probability, an unsigned one draws none
    spent = _Scripted(0, 0.0, 0.0, 0.9)
    ia._autoaugment_array(image, policy, spent)
    assert spent.draws == []


def test_rotate_shear_and_enhancers_keep_the_size_and_a_bad_operation_or_bin_raises() -> None:
    img = Image.fromarray((np.random.default_rng(0).random((20, 30, 3)) * 255).astype(np.uint8))
    for name in ("Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY", "Brightness", "Color",
                 "Contrast", "Sharpness", "Solarize", "Posterize"):
        for bin_index in (0, 9):
            assert ia._apply_autoaugment_op(
                img, name, bin_index, np.random.default_rng(0)).size == img.size
    for name in ("Invert", "AutoContrast", "Equalize"):
        assert ia._apply_autoaugment_op(img, name, None, np.random.default_rng(0)).size == img.size
    with pytest.raises(ValueError, match="Unknown AutoAugment operation"):
        ia._apply_autoaugment_op(img, "Cutout", 3, _Positive())
    for bad in (None, -1, ia.NUM_MAGNITUDE_BINS):
        with pytest.raises(ValueError, match="magnitude index"):
            ia._apply_autoaugment_op(img, "Rotate", bad, _Positive())


CIFAR_CACHE = Path.home() / ".keras" / "datasets" / "cifar-10-batches-py-target"


@pytest.mark.skipif(not CIFAR_CACHE.exists(), reason="CIFAR-10 is not in the local Keras cache")
def test_the_policy_changes_cifar10_images_as_strongly_as_torchvisions_does() -> None:
    # Measured with torchvision's table, 10 bins and fill 0 on these 3,000 images:
    # scratch_explore_parity/logs/measure_autoaugment_strength.out, 0.1293 mean absolute pixel
    # change (this module, six seed sets: 0.1274 to 0.1367, sd 0.0038). The earlier table, 31 bins and
    # grey fill read 0.1091, torchvision's table with 31 bins and grey fill 0.0731, the earlier table
    # with 10 bins and fill 0 0.1752: the band below tells all three from it.
    import keras

    (train_images, _), _ = keras.datasets.cifar10.load_data()
    images = train_images[:3000].astype(np.float32) / 255.0
    change = [
        float(np.abs(ia._autoaugment_array(image, ia.CIFAR10_POLICY, np.random.default_rng([i, 7]))
                     - image).mean())
        for i, image in enumerate(images)
    ]
    assert 0.121 < float(np.mean(change)) < 0.137, float(np.mean(change))


def test_an_unknown_policy_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown AutoAugment policy"):
        ia.autoaugment("imagenet")


@pytest.mark.parametrize("shape", [(32, 32, 3), (64, 48, 3), (40, 40, 1)])
def test_autoaugment_is_size_agnostic_deterministic_and_stays_in_unit_range(shape) -> None:
    op = ia.autoaugment("cifar10")
    image = _noise(*shape)
    first = op(SEED, image)
    assert first.shape == image.shape and first.dtype == tf.float32
    assert float(tf.reduce_min(first)) >= 0.0 and float(tf.reduce_max(first)) <= 1.0
    assert _same(first, op(SEED, image))
    assert any(not _same(first, op(tf.constant([k, k + 1], tf.int32), image)) for k in range(8))


def test_autoaugment_with_every_probability_zero_returns_the_image(monkeypatch) -> None:
    monkeypatch.setitem(ia.AUTOAUGMENT_POLICIES, "never", ((("Invert", 0.0, 5), ("Rotate", 0.0, 5)),))
    image = _noise()
    out = ia.autoaugment("never")(SEED, image)
    np.testing.assert_allclose(out, image, atol=1 / 255)


def test_autoaugment_inside_a_parallel_map_is_a_function_of_the_seed() -> None:
    images = tf.stack([_noise(seed=k) for k in range(6)])
    seeds = tf.constant([[k, 7] for k in range(6)], tf.int64)
    op = ia.autoaugment("cifar10")
    runs = [
        np.stack([x for x in tf.data.Dataset.from_tensor_slices((seeds, images))
                  .map(op, num_parallel_calls=4).as_numpy_iterator()])
        for _ in range(2)
    ]
    np.testing.assert_array_equal(*runs)


# ---------------------------------------------------------------------
# Random erasing
# ---------------------------------------------------------------------


def test_erasing_with_probability_zero_or_no_fitting_box_is_the_identity() -> None:
    image = _noise()
    assert _same(ia.random_erasing(0.0)(SEED, image), image)
    tiny = _noise(1, 1, 3)
    assert _same(ia.random_erasing(1.0)(SEED, tiny), tiny)


@pytest.mark.parametrize("size", [32, 128])
def test_erasing_writes_one_rectangle_of_the_fill_inside_the_area_range(size) -> None:
    fill = (0.25, 0.5, 0.75)
    op = ia.random_erasing(1.0, fill)
    image = tf.ones((size, size, 3)) * 0.1
    fractions = []
    for k in range(40):
        out = op(tf.constant([k, 5], tf.int32), image).numpy()
        erased = (out != 0.1).any(axis=-1)
        rows, cols = np.flatnonzero(erased.any(axis=1)), np.flatnonzero(erased.any(axis=0))
        assert erased.sum() == len(rows) * len(cols), "the erased pixels form one rectangle"
        np.testing.assert_allclose(out[erased], np.tile(fill, (erased.sum(), 1)))
        fractions.append(erased.mean())
    assert 0.015 <= min(fractions) and max(fractions) <= 0.36
    assert 0.08 < np.mean(fractions) < 0.27


def test_erasing_fires_at_the_requested_rate_and_is_deterministic() -> None:
    op = ia.random_erasing(0.25, (0.0,))
    image = tf.ones((32, 32, 1))
    fired = [bool((op(tf.constant([k, 9], tf.int32), image).numpy() != 1.0).any())
             for k in range(400)]
    assert 0.17 < np.mean(fired) < 0.33
    assert _same(op(SEED, image), op(SEED, image))


def test_erasing_refuses_a_bad_probability_or_range() -> None:
    with pytest.raises(ValueError, match="prob must be in"):
        ia.random_erasing(1.5)
    with pytest.raises(ValueError, match="bad erasing ranges"):
        ia.random_erasing(0.5, area=(0.5, 0.1))


# ---------------------------------------------------------------------
# Flip, pad-crop, photometric
# ---------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(32, 32, 3), (96, 64, 3), (28, 28, 1)])
@pytest.mark.parametrize("mode", ["CONSTANT", "REFLECT"])
def test_flip_pad_crop_keeps_the_shape_is_a_function_of_the_seed_and_varies(shape, mode) -> None:
    op = ia.flip_pad_crop(4, mode=mode)
    image = _noise(*shape)
    out = op(SEED, image)
    assert out.shape == image.shape and _same(out, op(SEED, image))
    assert any(not _same(out, op(tf.constant([k, 1], tf.int32), image)) for k in range(8))


def test_flip_alone_either_keeps_or_mirrors_the_image() -> None:
    image = _noise()
    outs = {tuple(np.asarray(ia.random_flip()(tf.constant([k, 2], tf.int32), image)).ravel()[:5])
            for k in range(20)}
    assert outs == {tuple(np.asarray(image).ravel()[:5]),
                    tuple(np.asarray(tf.image.flip_left_right(image)).ravel()[:5])}


def test_photometric_without_amounts_is_the_identity_and_amounts_are_bounded() -> None:
    image = tf.ones((8, 8, 3)) * 0.5
    assert _same(ia.photometric()(SEED, image), image)
    deltas = [float(tf.reduce_max(tf.abs(
        ia.photometric(brightness=0.1)(tf.constant([k, 3], tf.int32), image) - image)))
        for k in range(30)]
    assert max(deltas) <= 0.1 + 1e-6 and max(deltas) > 0.02
    out = ia.photometric(contrast=0.1)(SEED, _noise())
    assert out.shape == (32, 32, 3)


def test_photometric_applies_colour_ops_to_three_channels_and_skips_them_on_one() -> None:
    op = ia.photometric(0.1, 0.1, 0.5, 0.2)
    colour = _noise()
    assert not _same(op(SEED, colour), ia.photometric(0.1, 0.1)(SEED, colour))
    grey = _noise(c=1)
    assert op(SEED, grey).shape == grey.shape


@pytest.mark.parametrize("kwargs", [dict(brightness=-0.1), dict(contrast=1.0),
                                    dict(saturation=1.5), dict(hue=0.9)])
def test_photometric_refuses_an_impossible_amount(kwargs) -> None:
    with pytest.raises(ValueError, match="photometric amounts"):
        ia.photometric(**kwargs)


def test_compose_applies_in_order_and_needs_an_operation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ia.compose()
    add_one = lambda seed, image: image + 1.0  # noqa: E731
    double = lambda seed, image: image * 2.0  # noqa: E731
    image = tf.zeros((4, 4, 1))
    assert float(ia.compose(add_one, double)(SEED, image)[0, 0, 0]) == 2.0
    assert float(ia.compose(double, add_one)(SEED, image)[0, 0, 0]) == 1.0
    assert ia.compose(add_one) is add_one


# ---------------------------------------------------------------------
# Hooks, presets, flags
# ---------------------------------------------------------------------


def test_as_hook_is_seeded_keeps_the_label_and_matches_the_op() -> None:
    hook = ia.as_hook(ia.random_flip())
    assert augment_hook.is_seeded(hook)
    image = _noise()
    out, label = hook(SEED, image, tf.constant(7))
    assert int(label) == 7 and _same(out, ia.random_flip()(SEED, image))


def test_every_preset_builds_runs_at_two_sizes_and_none_is_none() -> None:
    assert ia.build_op("none") is None and ia.build_augment("none") is None
    for name in ia.PRESETS[1:]:
        hook = ia.build_augment(name)
        assert augment_hook.is_seeded(hook)
        for h, w in ((32, 32), (80, 64)):
            out, _ = hook(SEED, _noise(h, w), tf.constant(1))
            assert out.shape == (h, w, 3)


def test_the_unknown_preset_and_a_misplaced_erasing_probability_are_refused() -> None:
    with pytest.raises(ValueError, match="Unknown augmentation preset"):
        ia.build_op("mixup")
    with pytest.raises(ValueError, match="does not apply to the 'flip' preset"):
        ia.build_op("flip", random_erasing_prob=0.25)


def test_cifar_autoaugment_erases_with_the_mean_fill_so_normalization_maps_it_to_zero() -> None:
    """The reference erased AFTER normalization with 0; erasing before it with the mean is the same."""
    mean = np.array([0.4914, 0.4822, 0.4465], np.float32)
    std = np.array([0.247, 0.243, 0.261], np.float32)
    op = ia.build_op("cifar_autoaugment", random_erasing_prob=1.0, fill=mean)
    out = op(SEED, _noise()).numpy()
    erased = np.isclose(out, mean).all(axis=-1)
    assert erased.sum() > 0
    np.testing.assert_allclose(((out - mean) / std)[erased], 0.0, atol=1e-5)


def test_cifar_light_is_vits_recipe_without_clipping() -> None:
    out = ia.build_op("cifar_light")(SEED, tf.ones((32, 32, 3)))
    assert float(tf.reduce_max(out)) > 1.0, "brightness may leave [0, 1]: no clip_by_value (D-006)"


def test_unseeded_draws_its_seed_from_the_global_generator() -> None:
    op = ia.unseeded(ia.random_flip())
    image = _noise()
    tf.random.set_seed(5)
    first = op(image)
    tf.random.set_seed(5)
    assert _same(first, op(image))


def test_the_augment_flag_offers_the_presets_and_defaults_as_asked() -> None:
    parser = argparse.ArgumentParser()
    ia.add_augmentation_arguments(parser)
    assert parser.parse_args([]).augment is None
    assert parser.parse_args(["--augment", "cifar_light"]).augment == "cifar_light"
    with pytest.raises(SystemExit):
        parser.parse_args(["--augment", "mixup"])
    pinned = argparse.ArgumentParser()
    ia.add_augmentation_arguments(pinned, default="flip")
    assert pinned.parse_args([]).augment == "flip"


# ---------------------------------------------------------------------
# The hook protocol
# ---------------------------------------------------------------------


def test_with_hook_runs_plain_and_seeded_hooks_and_none_is_the_dataset() -> None:
    ds = tf.data.Dataset.from_tensor_slices((tf.zeros((6, 4, 4, 1)), tf.range(6)))
    assert augment_hook.with_hook(ds, None, 0) is ds
    plain = augment_hook.with_hook(ds, lambda image, label: (image + 1.0, label), 0)
    assert all(float(x.sum()) == 16.0 for x, _ in plain.as_numpy_iterator())

    @augment_hook.seeded
    def stamp(seed, image, label):
        return image + tf.cast(seed[0] % 1000, tf.float32), label

    runs = [[float(x[0, 0, 0]) for x, _ in augment_hook.with_hook(ds, stamp, 3).as_numpy_iterator()]
            for _ in range(2)]
    assert runs[0] == runs[1] and len(set(runs[0])) > 1


# ---------------------------------------------------------------------
# Random affine
# ---------------------------------------------------------------------


def test_the_affine_with_no_amounts_is_the_identity() -> None:
    image = _noise(24, 40, 3)
    np.testing.assert_allclose(ia.random_affine()(SEED, image), image, atol=1e-5)


@pytest.mark.parametrize("shape", [(32, 32, 3), (24, 40, 3), (28, 28, 1)])
def test_the_affine_keeps_the_shape_is_a_function_of_the_seed_and_varies(shape) -> None:
    op = ia.random_affine(rotation=20, zoom=0.1, translation=0.1)
    image = _noise(*shape)
    out = op(SEED, image)
    assert out.shape == image.shape and _same(out, op(SEED, image))
    assert any(not _same(out, op(tf.constant([k, 3], tf.int32), image)) for k in range(6))


def test_the_affine_translation_moves_the_image_by_the_drawn_fraction() -> None:
    image = np.zeros((20, 20, 1), np.float32)
    image[:, 10] = 1.0
    op = ia.random_affine(translation=0.25, fill_mode="CONSTANT")
    shifts = set()
    for k in range(20):
        out = op(tf.constant([k, 1], tf.int32), tf.constant(image)).numpy()
        column = int(np.argmax(out[10, :, 0]))
        shifts.add(column - 10)
        assert abs(column - 10) <= 5 + 1
    assert len(shifts) > 3, "the shift is drawn per seed"


def test_the_affine_zoom_scales_about_the_centre() -> None:
    image = np.zeros((21, 21, 1), np.float32)
    image[10, 10] = 1.0
    out = ia.random_affine(zoom=0.5, fill_mode="CONSTANT")(SEED, tf.constant(image)).numpy()
    assert out[10, 10, 0] > 0.0, "the centre stays put"


def test_the_affine_refuses_a_bad_amount_or_fill() -> None:
    with pytest.raises(ValueError, match="random_affine amounts"):
        ia.random_affine(zoom=1.0)
    with pytest.raises(ValueError, match="Unknown fill_mode"):
        ia.random_affine(fill_mode="MIRROR")


def test_the_affine_rotates_about_the_centre_by_at_most_the_asked_angle() -> None:
    image = np.zeros((41, 41, 1), np.float32)
    image[:, 20] = 1.0
    op = ia.random_affine(rotation=30, fill_mode="CONSTANT")
    slopes = []
    for k in range(30):
        out = op(tf.constant([k, 9], tf.int32), tf.constant(image)).numpy()[..., 0]
        top, bottom = int(np.argmax(out[5])), int(np.argmax(out[35]))
        slopes.append(np.degrees(np.arctan2(bottom - top, 30)))
        assert out[20].argmax() in (19, 20, 21), "the centre row keeps the line at the centre"
    assert max(np.abs(slopes)) <= 30 + 3 and max(np.abs(slopes)) > 10
    assert min(slopes) < 0 < max(slopes), "both directions are drawn"


# ---------------------------------------------------------------------
# Crop and dihedral, and the trainers that use them
# ---------------------------------------------------------------------


def test_random_crop_takes_a_window_of_the_image_and_refuses_a_bad_size() -> None:
    image = _noise(40, 48, 3)
    out = ia.random_crop(32, 24)(SEED, image)
    assert out.shape == (32, 24, 3) and _same(out, ia.random_crop(32, 24)(SEED, image))
    found = any(np.array_equal(np.asarray(image)[t:t + 32, l:l + 24], np.asarray(out))
                for t in range(9) for l in range(25))
    assert found, "the crop is a window of the image"
    with pytest.raises(ValueError, match="crop size must be positive"):
        ia.random_crop(0, 8)


def test_dihedral_yields_every_symmetry_of_a_square_patch() -> None:
    image = tf.constant(np.arange(16, dtype=np.float32).reshape(4, 4, 1))
    seen = {tuple(np.asarray(ia.random_dihedral()(tf.constant([k, 6], tf.int32), image)).ravel())
            for k in range(200)}
    assert len(seen) == 8, "the dihedral group of the square"
    assert all(sorted(s) == list(range(16)) for s in seen)


def test_augment_patch_is_the_dihedral_op_on_the_global_generator() -> None:
    from train.common.augment import augment_patch

    image = tf.constant(np.arange(16, dtype=np.float32).reshape(4, 4, 1))
    outs = {tuple(np.asarray(augment_patch(image)).ravel()) for _ in range(100)}
    assert 4 <= len(outs) <= 8
    tf.random.set_seed(3)
    a = augment_patch(image)
    tf.random.set_seed(3)
    assert _same(a, augment_patch(image))


def test_the_clip_loaders_augmentation_runs_on_the_unit_range_and_stays_in_pixels() -> None:
    from train.clip.train_clip import CLIPDataLoader

    class _Tok:
        def encode(self, text):
            return [1]

    loader = CLIPDataLoader(tokenizer=_Tok(), augmentation=True)
    image = tf.fill((16, 16, 3), 100.0)
    deltas = [float(tf.reduce_max(tf.abs(loader._augment_image(image) - image))) for _ in range(30)]
    assert 0.0 < max(deltas) < 100.0, "a 0.1 delta of the unit range is about 25 levels, not 0.1"
    assert float(tf.reduce_min(loader._augment_image(tf.zeros((16, 16, 3))))) >= 0.0


def test_the_image_text_training_branch_crops_and_flips_and_eval_is_untouched() -> None:
    from train.common import image_text

    image = tf.cast(tf.reshape(tf.range(20 * 20 * 3) % 251, (20, 20, 3)), tf.uint8)
    train = image_text.augment_and_normalize(image, 16, True)
    evaluated = image_text.augment_and_normalize(image, 20, False)
    assert train.shape == (16, 16, 3) and evaluated.shape == (20, 20, 3)
    again = [image_text.augment_and_normalize(image, 16, True) for _ in range(10)]
    assert any(not _same(train, other) for other in again)
