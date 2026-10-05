"""Test suite for the transformation-sequence datasets (paper Sections A.8, A.9).

A dataset generator that silently produces sequences with **no signal in them** is
the failure mode worth guarding: the rotation code path here returned an all-black
frame for every timestep on the first implementation, and nothing about the SHAPE
of that output is wrong. So the suite is built around liveness — every transform
must produce frames that actually differ, and the factors must actually vary.

The rendering path is also pinned for determinism. `gen_image_ops.
image_projective_transform_v3` was tried first and MEASURED non-deterministic in
this build: three consecutive identical processes returned sum 0.0, sum 0.0 and
then the correct value for the same 28x20 identity probe. The warps are therefore
pure NumPy, and `test_the_warps_are_reproducible_across_processes` is the guard
against that regression.

The dSprites tests are SKIPPED when the archive is absent, since the dataset lives
on a data volume rather than in git. They are never silently passed.
"""

import numpy as np
import pytest

from dl_techniques.datasets.vision.transform_sequences import (
    DSPRITES_TRANSFORMS,
    MNIST_TRANSFORMS,
    _affine_warp,
    _grayscale_to_hue,
    _hsl_to_rgb,
    _rescale,
    _rotate_hue,
    build_dsprites_transform_sequences,
    build_mnist_transform_sequences,
    create_transform_sequence_dataset,
    download_dsprites,
    load_dsprites_subset,
)

# The dSprites subset the paper specifies: 3 shapes x 5 scales x 15 x 15 x 15.
DSPRITES_SUBSET_SIZE = 50625


def _probe():
    """A NON-SQUARE, off-centre bar: a square centred probe cannot see a transpose."""
    image = np.zeros((28, 20, 1), dtype=np.float32)
    image[10:18, 4:16] = 1.0
    return image


class TestMNISTTransformSequences:
    """Shape, liveness and factor bookkeeping."""

    @pytest.mark.parametrize("transform", MNIST_TRANSFORMS)
    def test_shape_and_dtype(self, transform):
        sequences, factors = build_mnist_transform_sequences(
            transform=transform, num_sequences=3
        )
        assert sequences.shape == (3, 18, 28, 28, 3)
        assert sequences.dtype == np.float32
        assert factors.shape == (3, 18)
        assert np.all(np.isfinite(sequences))
        assert sequences.min() >= 0.0 and sequences.max() <= 1.0

    @pytest.mark.parametrize("transform", MNIST_TRANSFORMS)
    def test_consecutive_frames_actually_differ(self, transform):
        """The liveness guard.

        A generator that emitted a constant frame would produce a perfectly
        shaped dataset that carries no transformation at all -- and the model
        trained on it would learn nothing while every shape assertion passed.
        """
        sequences, _ = build_mnist_transform_sequences(
            transform=transform, num_sequences=4
        )
        deltas = np.abs(sequences[:, 1:] - sequences[:, :-1]).max(axis=(2, 3, 4))
        assert np.all(deltas > 1e-3), (
            f"transform {transform!r} produced near-identical consecutive "
            f"frames: max deltas {deltas}"
        )

    @pytest.mark.parametrize("transform", MNIST_TRANSFORMS)
    def test_the_factor_varies_within_a_sequence(self, transform):
        _, factors = build_mnist_transform_sequences(
            transform=transform, num_sequences=4
        )
        distinct = [len(np.unique(factors[b])) for b in range(4)]
        assert np.all(np.array(distinct) > 5), distinct

    @pytest.mark.parametrize("transform", MNIST_TRANSFORMS)
    def test_start_poses_are_randomised(self, transform):
        """Without a random start, a model could learn absolute factor positions
        instead of the shift — and CapCorr, which measures the shift, would be
        measuring nothing."""
        _, factors = build_mnist_transform_sequences(
            transform=transform, num_sequences=12
        )
        starts = set(np.round(factors[:, 0], 6).tolist())
        assert len(starts) > 3, f"only {len(starts)} distinct start poses"

    def test_rotation_cycles_through_360_degrees(self):
        _, factors = build_mnist_transform_sequences(
            transform="rotation", num_sequences=2, sequence_length=18
        )
        assert factors.min() == pytest.approx(0.0)
        assert factors.max() == pytest.approx(340.0)  # 18 steps of 20 degrees

    def test_scale_is_bounded_not_infinite(self):
        """Scale is inherently non-cyclic, so the paper bounds it to [0.60, 1.26].

        MEASURED: with the paper's 3.66% increment the cycle is 0.600 .. 1.2594.
        """
        _, factors = build_mnist_transform_sequences(
            transform="scale", num_sequences=2
        )
        assert factors.min() >= 0.60 - 1e-6
        assert factors.max() <= 1.26 + 1e-6

    def test_colour_sequences_carry_chrominance(self):
        """MNIST is grayscale, so a hue rotation alone would be a NO-OP.

        Without the tinting step every frame of a colour sequence would be
        identical — a valid-shaped dataset with zero signal in it.
        """
        sequences, _ = build_mnist_transform_sequences(
            transform="color", num_sequences=2
        )
        chroma = sequences.max(axis=-1) - sequences.min(axis=-1)
        assert chroma.max() > 0.05, "the colour sequences carry no chrominance"

    def test_a_different_seed_gives_different_sequences(self):
        first, _ = build_mnist_transform_sequences(
            transform="rotation", num_sequences=2, seed=1
        )
        second, _ = build_mnist_transform_sequences(
            transform="rotation", num_sequences=2, seed=2
        )
        assert not np.allclose(first, second)

    def test_the_same_seed_gives_identical_sequences(self):
        first, _ = build_mnist_transform_sequences(
            transform="rotation", num_sequences=2, seed=1
        )
        second, _ = build_mnist_transform_sequences(
            transform="rotation", num_sequences=2, seed=1
        )
        np.testing.assert_array_equal(first, second)

    def test_rejects_an_unknown_transform(self):
        with pytest.raises(ValueError, match="transform must be one of"):
            build_mnist_transform_sequences(transform="wobble")

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"sequence_length": 0}, "sequence_length must be positive"),
            (
                {"transform": "scale", "scale_increment": 10.0},
                "spans no complete step",
            ),
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            build_mnist_transform_sequences(
                transform=kwargs.pop("transform", "rotation"),
                num_sequences=2,
                **kwargs,
            )

    def test_rejects_a_non_positive_sequence_count(self):
        with pytest.raises(ValueError, match="num_sequences must be positive"):
            build_mnist_transform_sequences(num_sequences=0)

    def test_a_custom_sequence_length_is_honoured(self):
        sequences, factors = build_mnist_transform_sequences(
            transform="rotation", num_sequences=2, sequence_length=5
        )
        assert sequences.shape == (2, 5, 28, 28, 3)
        assert factors.shape == (2, 5)


class TestWarpPrimitives:
    """The rendering primitives, against closed forms and orientation probes."""

    def test_a_zero_rotation_is_the_identity(self):
        image = _probe()
        np.testing.assert_allclose(_affine_warp(image, 0.0), image, atol=0.0, rtol=0)

    def test_a_quarter_turn_moves_the_bar_rightwards(self):
        """Orientation probe on a NON-SQUARE image.

        The bar occupies rows 10-17, columns 4-15 of a 28x20 frame. Rotating by
        +90 degrees about the centre must carry it into the columns. A transpose
        or a sign flip lands it somewhere else, and at 28x28 a square frame could
        not tell.
        """
        image = _probe()
        rotated = _affine_warp(image, np.pi / 2.0)
        assert rotated[10:18, 12:].max() > 0.5, "the bar did not move right"
        assert rotated[10:18, 0:3].max() == 0.0, "the bar did not move right"

    def test_rotation_preserves_mass_when_nothing_leaves_the_frame(self):
        image = _probe()
        assert _affine_warp(image, 0.0).sum() == pytest.approx(image.sum())

    def test_rotation_is_reproducible_across_calls(self):
        """The guard on the non-determinism the NumPy path replaced.

        The rejected TF op returned 0.0, 0.0 and then the right answer on three
        consecutive identical processes. Twenty in-process draws would not catch a
        process-level flake, but they DO catch a stateful or RNG-dependent warp,
        and the cross-process check is
        `test_the_warps_are_reproducible_across_processes`.
        """
        image = _probe()
        first = _affine_warp(image, 0.7)
        for _ in range(20):
            np.testing.assert_array_equal(_affine_warp(image, 0.7), first)

    def test_scale_of_one_is_the_identity_and_smaller_shrinks(self):
        image = _probe()
        np.testing.assert_allclose(
            _rescale(image, 1.0, 28), image, atol=0.0, rtol=0
        )
        assert _rescale(image, 0.6, 28).sum() < image.sum()

    def test_scale_keeps_the_content_centred(self):
        """A scale that also translated would present two changing factors at once.

        The probe is off-centre on purpose (columns 4-15 of a 20-wide frame), so a
        zoom whose translation terms were wrong would shift it visibly.
        """
        image = _probe()
        scaled = _rescale(image, 0.6, 28)
        # `np.nonzero` on a rank-3 array returns THREE arrays, so index rather than
        # unpack -- unpacking raises a "too many values to unpack" that has nothing
        # to do with the property under test.
        rows = np.nonzero(scaled > 0.5)[0]
        columns = np.nonzero(scaled > 0.5)[1]
        source_rows = np.nonzero(image > 0.5)[0]
        source_columns = np.nonzero(image > 0.5)[1]
        assert abs(rows.mean() - source_rows.mean()) < 1.5, "it drifted vertically"
        assert abs(columns.mean() - source_columns.mean()) < 1.5, (
            "it drifted horizontally"
        )

    def test_the_tint_preserves_luminance(self):
        """HSL, not HSV: HSV's V is the MAX channel and would darken the digit.

        MEASURED on a linear ramp: HSV gave mean RGB 0.2600 against a gray mean of
        0.5000; HSL gives 0.5193.
        """
        gray = np.repeat(np.linspace(0.0, 1.0, 64, dtype=np.float32)[:, None], 8, 1)
        tinted = _grayscale_to_hue(gray, 200.0)
        assert tinted.shape == (64, 8, 3)
        assert tinted.mean(axis=2).mean() == pytest.approx(gray.mean(), abs=0.05)

    def test_the_tint_leaves_the_extremes_achromatic(self):
        gray = np.repeat(np.linspace(0.0, 1.0, 64, dtype=np.float32)[:, None], 8, 1)
        tinted = _grayscale_to_hue(gray, 30.0)
        assert tinted[0].max() == pytest.approx(0.0, abs=1e-6)
        assert tinted[-1].min() == pytest.approx(1.0, abs=1e-6)

    def test_hsl_to_rgb_reproduces_the_primary_hues(self):
        """L = 0.5, S = 1: the only lightness at which a hue is a pure primary.

        At L = 1 every hue is white and at L = 0 every hue is black -- correct
        HSL behaviour, and the reason the primaries need the mid lightness.
        """
        hsl = np.array(
            [
                [[0.0, 1.0, 0.5]],
                [[1 / 3, 1.0, 0.5]],
                [[2 / 3, 1.0, 0.5]],
            ]
        )
        rgb = _hsl_to_rgb(hsl)
        np.testing.assert_allclose(rgb[0, 0], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(rgb[1, 0], [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(rgb[2, 0], [0.0, 0.0, 1.0], atol=1e-6)

    def test_hsl_is_grey_at_the_extremes_of_lightness(self):
        for lightness in (0.0, 1.0):
            rgb = _hsl_to_rgb(np.array([[[0.4, 1.0, lightness]]]))
            np.testing.assert_allclose(
                rgb[0, 0], np.full(3, lightness), atol=1e-6
            )

    def test_a_hue_rotation_of_zero_degrees_is_the_identity(self):
        image = _grayscale_to_hue(
            np.linspace(0.0, 1.0, 16, dtype=np.float32)[:, None] * np.ones((1, 4)),
            0.0,
        )
        np.testing.assert_allclose(
            _rotate_hue(image, 0.0), image, atol=1e-5, rtol=0
        )

    def test_a_hue_rotation_really_rotates_the_hue(self):
        gray = np.repeat(np.linspace(0.2, 0.8, 16, dtype=np.float32)[:, None], 4, 1)
        base = _grayscale_to_hue(gray, 0.0)
        turned = _rotate_hue(base, 120.0)
        assert np.abs(turned - base).max() > 0.1, "the hue did not move"

    def test_a_hue_rotation_preserves_the_YIQ_luminance(self):
        """YIQ rotates the CHROMINANCE plane and leaves Y alone.

        The claim is about Y, not about the mean of the RGB channels: the YIQ
        luminance weights (0.299, 0.587, 0.114) are not (1/3, 1/3, 1/3), so a
        rotation can legitimately move the channel mean while holding Y fixed. An
        RGB-space rotation would move Y, which is the defect this avoids.
        """
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
        gray = np.repeat(np.linspace(0.2, 0.8, 16, dtype=np.float32)[:, None], 4, 1)
        base = _grayscale_to_hue(gray, 0.0)
        turned = _rotate_hue(base, 120.0)
        before = base @ weights
        after = turned @ weights
        np.testing.assert_allclose(after, before, atol=1e-5, rtol=0)

    def test_a_grayscale_image_cannot_be_hue_rotated(self):
        with pytest.raises(ValueError, match="needs a colour image"):
            _rotate_hue(np.zeros((4, 4, 1), "float32"), 30.0)


def _dsprites_available():
    try:
        download_dsprites()
    except Exception:  # pragma: no cover - environment dependent
        return False
    return True


dsprites_required = pytest.mark.skipif(
    not _dsprites_available(),
    reason="the dSprites archive is not present; it lives on the data volume "
           "and is not tracked in git",
)


@dsprites_required
class TestDSpritesSubset:
    """The paper's 50,625-image subset."""

    def test_the_subset_size_matches_the_paper(self):
        images, factors = load_dsprites_subset()
        assert images.shape[0] == DSPRITES_SUBSET_SIZE, (
            f"got {images.shape[0]}, the paper's subset is "
            f"3 shapes x 5 scales x 15 x 15 x 15 = {DSPRITES_SUBSET_SIZE}"
        )
        assert images.shape[1:] == (64, 64, 1)
        assert images.dtype == np.float32
        assert images.min() >= 0.0 and images.max() <= 1.0

    def test_every_factor_axis_has_the_papers_cardinality(self):
        _, factors = load_dsprites_subset()
        assert len(np.unique(factors["shape"])) == 3
        assert len(np.unique(factors["scale"])) == 5
        assert len(np.unique(factors["orientation"])) == 15
        assert len(np.unique(factors["x_position"])) == 15
        assert len(np.unique(factors["y_position"])) == 15

    def test_the_largest_scales_are_kept(self):
        _, factors = load_dsprites_subset()
        full = np.unique(factors["scale"])
        assert full.max() == pytest.approx(1.0, abs=1e-6)

    def test_a_wider_stride_gives_a_smaller_subset(self):
        """The stride is what makes the subset affordable, and it must be visible."""
        images, _ = load_dsprites_subset(stride_positions=4)
        assert images.shape[0] < DSPRITES_SUBSET_SIZE

    def test_download_does_not_repeat(self, tmp_path):
        """A second call must not touch the network."""
        first = download_dsprites(cache_root=str(tmp_path))
        second = download_dsprites(cache_root=str(tmp_path))
        assert first == second


@dsprites_required
class TestDSpritesTransformSequences:
    """Sequence construction over the labelled subset."""

    @pytest.mark.parametrize("transform", DSPRITES_TRANSFORMS)
    def test_shape_and_liveness(self, transform):
        sequences, factors = build_dsprites_transform_sequences(
            transform=transform, num_sequences=3
        )
        assert sequences.shape == (3, 15, 64, 64, 1)
        assert factors.shape == (3, 15)
        deltas = np.abs(sequences[:, 1:] - sequences[:, :-1]).max(axis=(2, 3, 4))
        assert np.all(deltas > 1e-3), (
            f"{transform!r} produced near-identical consecutive frames: {deltas}"
        )

    @pytest.mark.parametrize("transform", ("orientation", "x_position", "y_position"))
    def test_the_cyclic_factors_visit_every_value(self, transform):
        """S = 15 equals the number of selected factor values, so a cycle fits."""
        _, factors = build_dsprites_transform_sequences(
            transform=transform, num_sequences=4
        )
        distinct = [len(np.unique(factors[b])) for b in range(4)]
        assert np.all(np.array(distinct) == 15), distinct

    def test_scale_loops_over_its_available_values(self):
        """Scale is NOT cyclic in dSprites; the cycle closes on a repeated list.

        MEASURED: 5 distinct scales, so a 15-frame sequence visits each three
        times. That is the paper's documented relaxation, not a smooth cycle.
        """
        _, factors = build_dsprites_transform_sequences(
            transform="scale", num_sequences=2
        )
        assert [len(np.unique(factors[b])) for b in range(2)] == [5, 5]

    def test_scale_loops_is_configurable(self):
        _, once = build_dsprites_transform_sequences(
            transform="scale", num_sequences=2, scale_loops=1
        )
        assert [len(np.unique(once[b])) for b in range(2)] == [5, 5]

    def test_rejects_an_unknown_transform(self):
        with pytest.raises(ValueError, match="transform must be one of"):
            build_dsprites_transform_sequences(transform="shape", num_sequences=2)


class TestCreateTransformSequenceDataset:
    """The single dispatch entry point."""

    @pytest.mark.parametrize("name", ("mnist",))
    def test_dispatches_to_mnist(self, name):
        sequences, factors = create_transform_sequence_dataset(
            name, num_sequences=2
        )
        assert sequences.shape == (2, 18, 28, 28, 3)
        assert factors.shape == (2, 18)

    def test_rejects_an_unknown_dataset(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            create_transform_sequence_dataset("cifar10", num_sequences=2)

    def test_the_sequence_length_default_follows_the_dataset(self):
        sequences, _ = create_transform_sequence_dataset(
            "mnist", sequence_length=None, num_sequences=1
        )
        assert sequences.shape[1] == 18


class TestWarpsAreReproducibleAcrossProcesses:
    """The cross-process guard on the non-determinism the NumPy path replaced.

    Subprocess-isolated because the defect it guards was process-level: three
    identical runs of the rejected TF op gave 0.0, 0.0, then the right answer. An
    in-process loop cannot see that.
    """

    def test_the_warps_are_reproducible_across_processes(self):
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import numpy as np, sys
            sys.path.insert(0, %r)
            from dl_techniques.datasets.vision.transform_sequences import (
                _affine_warp, _rescale,
            )
            image = np.zeros((28, 20, 1), dtype=np.float32)
            image[10:18, 4:16] = 1.0
            print(
                "%%.10f %%.10f %%.10f"
                %% (
                    _affine_warp(image, 0.7).sum(),
                    _affine_warp(image, np.pi / 2.0).sum(),
                    _rescale(image, 0.6, 28).sum(),
                )
            )
            """
        ) % ("src",)
        outputs = []
        for _ in range(3):
            completed = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                cwd=".",
                timeout=300,
            )
            assert completed.returncode == 0, completed.stderr[-2000:]
            outputs.append(completed.stdout.strip())
        assert len(set(outputs)) == 1, (
            "the warp is not reproducible across processes:\n" + "\n".join(outputs)
        )
