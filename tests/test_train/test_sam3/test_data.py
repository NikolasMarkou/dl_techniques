"""Tests for ``train/sam3/data.py`` -- the synthetic text-prompted source.

**A bug in this file's subject is indistinguishable from a model bug**, which is
why this suite runs BEFORE any learnability claim. The guards are shaped so that
none of them can pass by construction:

- **The box oracle is derived from the RASTERIZED MASK, never from the drawn
  parameters.** The pipeline computes each box ANALYTICALLY from the sampled
  extent; the oracle here recomputes it from the emitted 64x64 binary mask's own
  extents. The two are genuinely different computations, so a rasterization or
  placement bug is visible instead of cancelling out. The oracle's own RED arm
  (a deliberately displaced box) proves it discriminates.
- **Prompt-vs-target correspondence is checked against the RENDERED IMAGE.**
  Connected components of the composited image are segmented and classified by
  shape statistics, with no reference to the record's target arrays. A pipeline
  that ignored the prompt and returned every instance would satisfy every
  box/mask assertion and fails this one on the COUNT.
- **The zero-instance rate is measured, not assumed**, with both degenerate
  rates (0.0 and 1.0) as liveness arms.
- **The packed width is compared against the model's own
  ``packed_target_spec()``** at both ``include_masks`` values.
"""

import hashlib
import os
import tempfile

import cv2
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.sam3_detection_loss import (
    Sam3DetectionLoss,
    packed_channel_count,
    unpack_targets,
)
from dl_techniques.models.sam3 import Sam3Image
from dl_techniques.models.sam3.training_model import (
    Sam3TrainingModel,
    compile_sam3_trainer,
)
from train.sam3.data import (
    CATEGORIES,
    CATEGORY_PHRASES,
    PAD_ID,
    WORD_TO_ID,
    _downsample,
    _rasterize,
    _sample_extent,
    build_sam3_dataset,
    encode_phrase,
    synthetic_prompt_samples,
)

#: The shipped geometry the learnability run will use (`small`): 224 px images,
#: a 64x64 mask grid, `context_length=32`.
IMAGE_SIZE = 224
MASK_GRID = (64, 64)
CONTEXT = 32
N_MAX = 8

#: Box-vs-mask tolerance, in MASK CELLS, converted to the normalized frame the
#: boxes live in. Rasterizing at 224 px and area-downsampling to 64 moves each
#: edge by up to about one cell, so a width or height can move by two.
#: MEASURED over 570 instances: worst component deviation 1.40 cells. Pinned at
#: 2.5 cells, which still leaves a placement bug (tens of cells) and a
#: width/height swap (>= 9 cells for a `bar`) far outside.
BOX_TOLERANCE_CELLS = 2.5
BOX_TOLERANCE = BOX_TOLERANCE_CELLS / MASK_GRID[0]

#: Shape-statistic decision boundaries used ONLY by the image-side oracle below.
#: MEASURED per-instance ranges at 224 px are disjoint with wide gaps:
#: triangle fill [.504, .524], circle [.753, .797], square/bar 1.000; aspect
#: 1.0 for the first three and [2.29, 2.53] for `bar`. The boundaries sit in the
#: middle of those gaps.
FILL_TRIANGLE_MAX = 0.62
FILL_CIRCLE_MAX = 0.90
ASPECT_BAR_MIN = 1.60

#: The sha256 of the first batch at ``seed=99`` on the ``tiny`` geometry,
#: computed in a SEPARATE process (twice, under two different
#: ``PYTHONHASHSEED`` values, which agreed). See
#: ``test_the_first_batch_is_bit_reproducible_across_processes``.
PINNED_FIRST_BATCH_SHA256 = (
    "36c573f038bbf43d48be67869d1178bca8c1c3c766ba10d055467f1ba87e0f76")


# ---------------------------------------------------------------------
# Oracles. Neither calls anything in the module under test.
# ---------------------------------------------------------------------
def box_from_mask(mask: np.ndarray) -> np.ndarray:
    """Normalized ``cxcywh`` from a binary mask's OWN extents.

    This is the M1 oracle: it reads the rasterized geometry, not the parameters
    the pipeline drew from and not the pipeline's own box expression.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.size and cols.size, "empty mask has no box"
    height, width = mask.shape
    x_1, x_2 = float(cols[0]), float(cols[-1] + 1)
    y_1, y_2 = float(rows[0]), float(rows[-1] + 1)
    return np.asarray([
        (x_1 + x_2) * 0.5 / width,
        (y_1 + y_2) * 0.5 / height,
        (x_2 - x_1) / width,
        (y_2 - y_1) / height,
    ], dtype="float64")


def mask_statistics(component: np.ndarray) -> np.ndarray:
    """``(fill ratio, aspect ratio)`` of a binary component."""
    rows = np.flatnonzero(component.any(axis=1))
    cols = np.flatnonzero(component.any(axis=0))
    height = float(rows[-1] - rows[0] + 1)
    width = float(cols[-1] - cols[0] + 1)
    return np.asarray(
        [float(component.sum()) / (width * height), width / height])


def shape_statistics(category: str, downsample: bool,
                     count: int = 120, seed: int = 3) -> np.ndarray:
    """``(count, 2)`` statistics from DIRECT rasterization of one category.

    The reference distribution, produced without the sample generator, so a
    comparison against it is not a comparison of the pipeline with itself.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        mask = _rasterize(
            category, _sample_extent(category, IMAGE_SIZE, rng), IMAGE_SIZE)
        if downsample:
            mask = _downsample(mask, *MASK_GRID)
        rows.append(mask_statistics(mask))
    return np.asarray(rows)


def classify_component(component: np.ndarray) -> str:
    """Name a shape from its binary component, by fill ratio and aspect."""
    fill, aspect = mask_statistics(component)
    if fill < FILL_TRIANGLE_MAX:
        return "triangle"
    if fill < FILL_CIRCLE_MAX:
        return "circle"
    return "bar" if aspect > ASPECT_BAR_MIN else "square"


def components_of(image: np.ndarray) -> list:
    """Segment the composited image into per-instance binary components.

    Instances are drawn NON-OVERLAPPING and on a dark background (0.05-0.25 of
    full scale) in bright colours (0.55-1.0), so a mid-grey threshold plus
    connected components recovers them exactly -- with no reference to the
    record's target arrays.
    """
    grey = image.max(axis=-1)
    binary = (grey > 128.0).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary)
    return [(labels == index) for index in range(1, count)]


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def records():
    """200 records at the shipped geometry, one fixed seed."""
    return list(synthetic_prompt_samples(
        num_samples=200, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
        context_length=CONTEXT, max_instances=N_MAX,
        zero_instance_rate=0.25, seed=7))


@pytest.fixture(scope="module")
def tiny_model():
    return Sam3TrainingModel(Sam3Image.from_variant("tiny"))


# ---------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------
class TestVocabulary:
    def test_the_map_is_injective_over_the_closed_set(self):
        encodings = {name: tuple(encode_phrase(name, CONTEXT)[0].tolist())
                     for name in CATEGORIES}
        assert len(set(encodings.values())) == len(CATEGORIES)

    def test_the_same_phrase_always_maps_to_the_same_ids(self):
        for name in CATEGORIES:
            first = encode_phrase(name, CONTEXT)[0]
            second = encode_phrase(name, CONTEXT)[0]
            np.testing.assert_array_equal(first, second)

    def test_padding_id_is_zero_and_the_mask_marks_exactly_it(self):
        assert PAD_ID == 0
        for name in CATEGORIES:
            ids, mask = encode_phrase(name, CONTEXT)
            assert mask.dtype == np.bool_
            np.testing.assert_array_equal(mask, ids == PAD_ID)
            # True AT PADDING -- the key-padding polarity, not a keep mask.
            assert not bool(mask[0]) and bool(mask[-1])

    def test_the_padding_mask_is_not_a_constant_across_the_set(self):
        # A constant mask would make the padding channel untestable: the
        # phrases deliberately differ in length (2 words vs 3).
        widths = {int((~encode_phrase(name, CONTEXT)[1]).sum())
                  for name in CATEGORIES}
        assert len(widths) > 1

    def test_every_id_is_inside_the_small_variants_vocabulary(self):
        assert max([2] + list(WORD_TO_ID.values())) < 512

    def test_a_phrase_that_does_not_fit_raises_rather_than_truncating(self):
        longest = max(len(p.split()) for p in CATEGORY_PHRASES.values())
        with pytest.raises(ValueError, match="Truncation is refused"):
            encode_phrase("bar", longest + 1)


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------
class TestEmittedGeometryAgreesWithTheDrawnShapes:
    def test_every_emitted_box_matches_the_box_of_its_own_mask(self, records):
        deviations = []
        for record in records:
            for slot in range(N_MAX):
                if record["target_valid"][slot] == 0.0:
                    continue
                oracle = box_from_mask(record["target_masks"][slot])
                emitted = record["target_boxes"][slot].astype("float64")
                deviations.append(np.abs(oracle - emitted))
        assert len(deviations) > 100, "not enough instances to be a gate"
        worst = np.max(np.stack(deviations))
        assert worst < BOX_TOLERANCE, (
            f"worst box deviation {worst:.5f} exceeds {BOX_TOLERANCE:.5f} "
            f"({worst * MASK_GRID[0]:.2f} mask cells)")

    def test_that_oracle_goes_red_on_a_displaced_box(self, records):
        # M3 liveness arm. Without this, a tolerance loose enough to absorb the
        # rasterization jitter could also absorb a real placement bug.
        displaced = []
        for record in records:
            for slot in range(N_MAX):
                if record["target_valid"][slot] == 0.0:
                    continue
                oracle = box_from_mask(record["target_masks"][slot])
                wrong = record["target_boxes"][slot].astype("float64").copy()
                wrong[0] += 4.0 / MASK_GRID[0]   # four cells, not two
                displaced.append(np.max(np.abs(oracle - wrong)))
        assert min(displaced) > BOX_TOLERANCE

    def test_the_mask_area_matches_each_shapes_ANALYTIC_area(self):
        # A pure-geometry oracle: pi/4, 1, 1/2, 1 are the fill ratios of a
        # disc, a square, an isoceles triangle and a rectangle. Nothing about
        # the rasterizer or the downsampler enters it, which is why this is the
        # arm that sees a downsample-threshold mutation -- the box oracle above
        # does NOT (measured: mutation M-9 was INERT against it).
        analytic = {"circle": np.pi / 4.0, "square": 1.0,
                    "triangle": 0.5, "bar": 1.0}
        errors = {name: [] for name in CATEGORIES}
        for record in synthetic_prompt_samples(
                num_samples=150, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
                context_length=CONTEXT, max_instances=N_MAX,
                zero_instance_rate=0.0, seed=31):
            name = record["prompt_category"]
            for slot in range(N_MAX):
                if record["target_valid"][slot] == 0.0:
                    continue
                _, _, width, height = record["target_boxes"][slot]
                expected = (analytic[name] * width * MASK_GRID[1]
                            * height * MASK_GRID[0])
                observed = float(record["target_masks"][slot].sum())
                errors[name].append(abs(observed - expected) / expected)
        # MEASURED: per-category mean relative error 0.032-0.079 on the correct
        # code and 0.209-0.411 with the downsample threshold mutated to 0.0.
        # The 0.15 boundary sits between the two with ~2x margin either side.
        for name in CATEGORIES:
            assert len(errors[name]) >= 20, name
            assert float(np.mean(errors[name])) < 0.15, (
                f"{name}: mean relative area error "
                f"{float(np.mean(errors[name])):.3f}")

    def test_every_emitted_mask_is_binary_and_non_empty_where_valid(
            self, records):
        for record in records:
            for slot in range(N_MAX):
                mask = record["target_masks"][slot]
                assert set(np.unique(mask)).issubset({0.0, 1.0})
                if record["target_valid"][slot] > 0.0:
                    assert mask.sum() > 0.0
                else:
                    assert mask.sum() == 0.0

    def test_padded_rows_carry_exactly_zero_boxes(self, records):
        for record in records:
            padded = record["target_valid"] == 0.0
            assert np.all(record["target_boxes"][padded] == 0.0)

    def test_boxes_are_normalized_cxcywh_inside_the_unit_square(self, records):
        for record in records:
            valid = record["target_valid"] > 0.0
            boxes = record["target_boxes"][valid]
            if boxes.size == 0:
                continue
            assert np.all(boxes[:, 2:] > 0.0)
            assert np.all(boxes[:, 0] - boxes[:, 2] * 0.5 >= 0.0)
            assert np.all(boxes[:, 1] - boxes[:, 3] * 0.5 >= 0.0)
            assert np.all(boxes[:, 0] + boxes[:, 2] * 0.5 <= 1.0)
            assert np.all(boxes[:, 1] + boxes[:, 3] * 0.5 <= 1.0)

    @pytest.mark.parametrize("downsample", [False, True])
    def test_the_categories_are_separable_at_both_resolutions(self, downsample):
        ranges = {name: shape_statistics(name, downsample=downsample)
                  for name in CATEGORIES}
        # At the IMAGE resolution -- what the model classifies from -- every
        # pair must be separated on at least one axis with NO range overlap.
        # At the mask grid the same is asserted on the MEANS only, because
        # circle/triangle fill ranges do overlap on a 6-cell instance (a
        # measured property of the supervision target, recorded in data.py).
        for first in CATEGORIES:
            for second in CATEGORIES:
                if first >= second:
                    continue
                gaps = []
                for axis in (0, 1):
                    a, b = ranges[first][:, axis], ranges[second][:, axis]
                    if downsample:
                        gaps.append(abs(a.mean() - b.mean())
                                    > 3.0 * max(a.std(), b.std()))
                    else:
                        gaps.append(a.min() > b.max() or b.min() > a.max())
                assert any(gaps), f"{first} vs {second} not separable"


# ---------------------------------------------------------------------
# Prompt-vs-target correspondence
# ---------------------------------------------------------------------
class TestThePromptSelectsTheTargets:
    @pytest.mark.parametrize("seed", [0, 1, 2, 11])
    def test_targets_are_the_prompted_category_and_none_of_the_distractors(
            self, seed):
        distractor_images = 0
        checked = 0
        for record in synthetic_prompt_samples(
                num_samples=40, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
                context_length=CONTEXT, max_instances=N_MAX,
                zero_instance_rate=0.25, seed=seed):
            prompt = record["prompt_category"]
            drawn = [classify_component(c)
                     for c in components_of(record["image"])]
            expected = sum(1 for name in drawn if name == prompt)
            emitted = int(record["target_valid"].sum())
            # THE discriminating assertion: a pipeline that ignored the prompt
            # and returned every instance would give `emitted == len(drawn)`.
            assert emitted == expected, (
                f"seed {seed}: prompt {prompt!r}, image holds {drawn}, "
                f"pipeline emitted {emitted} target(s)")
            if any(name != prompt for name in drawn):
                distractor_images += 1
            checked += 1
        assert checked == 40
        # Without distractors the text channel would not be load-bearing.
        assert distractor_images >= 30

    def test_the_emitted_masks_carry_the_prompted_categorys_statistics(self):
        # An AGGREGATE oracle, deliberately, because at the 64x64 mask grid the
        # circle and triangle per-instance fill ranges do overlap at their
        # extremes (measured; recorded in data.py). The reference means below
        # come from DIRECT rasterization, never from the pipeline.
        reference = {name: shape_statistics(name, downsample=True).mean(axis=0)
                     for name in CATEGORIES}
        emitted = {name: [] for name in CATEGORIES}
        for record in synthetic_prompt_samples(
                num_samples=200, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
                context_length=CONTEXT, max_instances=N_MAX,
                zero_instance_rate=0.0, seed=5):
            for slot in range(N_MAX):
                if record["target_valid"][slot] == 0.0:
                    continue
                emitted[record["prompt_category"]].append(
                    mask_statistics(record["target_masks"][slot]))
        for name in CATEGORIES:
            assert len(emitted[name]) >= 20, name
            centre = np.mean(np.asarray(emitted[name]), axis=0)
            distances = {other: float(np.linalg.norm(centre - reference[other]))
                         for other in CATEGORIES}
            nearest = min(distances, key=distances.get)
            assert nearest == name, (
                f"masks emitted for prompt {name!r} sit closest to "
                f"{nearest!r}: {distances}")

    def test_the_token_ids_name_the_prompted_category(self, records):
        for record in records:
            expected, mask = encode_phrase(
                record["prompt_category"], CONTEXT)
            np.testing.assert_array_equal(record["token_ids"], expected)
            np.testing.assert_array_equal(record["token_padding_mask"], mask)


# ---------------------------------------------------------------------
# The zero-instance case
# ---------------------------------------------------------------------
class TestZeroInstanceIsGenuinelySampled:
    def test_the_observed_rate_tracks_the_configured_rate(self):
        total = 600
        rate = 0.25
        observed = sum(
            1 for record in synthetic_prompt_samples(
                num_samples=total, image_size=IMAGE_SIZE, mask_grid=MASK_GRID,
                context_length=CONTEXT, max_instances=N_MAX,
                zero_instance_rate=rate, seed=21)
            if record["target_valid"].sum() == 0.0) / total
        # 3 sigma of a Binomial(600, 0.25) proportion is 0.053; the tolerance
        # is stated rather than tuned to the number that came out.
        assert abs(observed - rate) < 0.06, f"observed {observed:.4f}"

    @pytest.mark.parametrize("rate,expect_zero", [(0.0, False), (1.0, True)])
    def test_the_degenerate_rates_are_liveness_arms(self, rate, expect_zero):
        counts = [record["target_valid"].sum() == 0.0
                  for record in synthetic_prompt_samples(
                      num_samples=60, image_size=IMAGE_SIZE,
                      mask_grid=MASK_GRID, context_length=CONTEXT,
                      max_instances=N_MAX, zero_instance_rate=rate, seed=4)]
        assert all(counts) if expect_zero else not any(counts)

    def test_keep_loss_is_zero_on_exactly_the_zero_instance_samples(
            self, tiny_model):
        dataset = build_sam3_dataset(
            tiny_model, num_samples=64, batch_size=8, max_instances=N_MAX,
            zero_instance_rate=0.5, seed=13)
        zeros = 0
        for _, packed in dataset:
            fields = unpack_targets(packed, include_masks=False)
            keep = np.asarray(fields["keep_loss"]).reshape(-1)
            valid = np.asarray(fields["target_valid"]).sum(axis=-1)
            np.testing.assert_array_equal(keep, (valid > 0.0).astype("float32"))
            np.testing.assert_array_equal(
                np.asarray(fields["num_boxes"]), valid)
            zeros += int((keep == 0.0).sum())
        # Both arms present: neither all-zero nor all-one would be a gate.
        assert 0 < zeros < 64

    def test_is_exhaustive_is_one_on_every_row(self, tiny_model):
        dataset = build_sam3_dataset(
            tiny_model, num_samples=16, batch_size=8, max_instances=N_MAX,
            seed=2)
        for _, packed in dataset:
            fields = unpack_targets(packed, include_masks=False)
            np.testing.assert_array_equal(
                np.asarray(fields["is_exhaustive"]), np.ones(8, "float32"))


# ---------------------------------------------------------------------
# The three-way width contract, from the pipeline's side
# ---------------------------------------------------------------------
class TestThePackedWidthIsDerived:
    @pytest.mark.parametrize("include_masks", [False, True])
    def test_the_emitted_width_equals_the_models_own_spec(self, include_masks):
        model = Sam3TrainingModel(
            Sam3Image.from_variant("tiny"), include_masks=include_masks)
        dataset = build_sam3_dataset(
            model, num_samples=8, batch_size=4, max_instances=N_MAX, seed=1)
        spec = model.packed_target_spec(N_MAX)
        for _, packed in dataset:
            assert tuple(packed.shape[1:]) == tuple(spec)
            # And the width is the LAYOUT's, not merely the model's.
            assert packed.shape[-1] == packed_channel_count(model.mask_size)

    def test_the_input_dict_is_exactly_the_models_contract(self, tiny_model):
        dataset = build_sam3_dataset(
            tiny_model, num_samples=4, batch_size=4, max_instances=N_MAX,
            seed=1)
        inputs, _ = next(iter(dataset))
        assert set(inputs) == {"image", "token_ids", "token_padding_mask"}
        assert inputs["token_padding_mask"].dtype == tf.bool

    def test_the_batch_axis_is_static(self, tiny_model):
        dataset = build_sam3_dataset(
            tiny_model, num_samples=10, batch_size=4, max_instances=N_MAX,
            seed=1)
        assert dataset.element_spec[1].shape[0] == 4
        # `drop_remainder=True`: 10 samples at batch 4 gives 2 batches, not 3.
        assert sum(1 for _ in dataset) == 2


# ---------------------------------------------------------------------
# Hygiene and reproducibility
# ---------------------------------------------------------------------
class TestPipelineHygiene:
    def test_nothing_is_written_to_disk(self, tiny_model):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as workdir:
            os.chdir(workdir)
            try:
                dataset = build_sam3_dataset(
                    tiny_model, num_samples=8, batch_size=4,
                    max_instances=N_MAX, seed=1)
                for _ in dataset:
                    pass
                leftovers = sorted(os.listdir(workdir))
            finally:
                os.chdir(original)
        assert leftovers == [], f"the source wrote {leftovers}"

    def test_the_first_batch_is_bit_reproducible_across_processes(self):
        # The expected digest was computed in a DIFFERENT process. An
        # in-process "two datasets agree" check cannot see a per-process hash
        # salt or a parallel stateful RNG; a pinned constant can.
        # `include_masks=True` deliberately: at the default the packed target
        # carries no mask block and the digest would be blind to the masks.
        model = Sam3TrainingModel(
            Sam3Image.from_variant("tiny"), include_masks=True)
        dataset = build_sam3_dataset(
            model, num_samples=8, batch_size=4, max_instances=N_MAX,
            seed=99)
        inputs, packed = next(iter(dataset))
        digest = hashlib.sha256()
        for tensor in (inputs["image"], inputs["token_ids"],
                       inputs["token_padding_mask"], packed):
            digest.update(np.asarray(tensor).tobytes())
        assert digest.hexdigest() == PINNED_FIRST_BATCH_SHA256

    def test_two_seeds_do_not_produce_the_same_data(self, tiny_model):
        # Liveness arm for the digest above: a source that ignored its seed
        # would satisfy the pin trivially.
        first = next(iter(build_sam3_dataset(
            tiny_model, num_samples=4, batch_size=4, seed=99)))[0]["image"]
        second = next(iter(build_sam3_dataset(
            tiny_model, num_samples=4, batch_size=4, seed=100)))[0]["image"]
        assert not np.array_equal(np.asarray(first), np.asarray(second))


# ---------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------
class TestEndToEnd:
    def test_one_batch_through_the_model_and_loss_gives_a_finite_loss(
            self, tiny_model):
        dataset = build_sam3_dataset(
            tiny_model, num_samples=8, batch_size=4, max_instances=N_MAX,
            zero_instance_rate=0.5, seed=17)
        loss_fn = Sam3DetectionLoss(include_masks=False)
        inputs, packed = next(iter(dataset))
        predictions = tiny_model(inputs, training=False)
        value = float(loss_fn(packed, predictions))
        assert np.isfinite(value) and value > 0.0

    def test_a_stock_fit_step_completes_on_this_dataset(self):
        model = Sam3TrainingModel(Sam3Image.from_variant("tiny"))
        compile_sam3_trainer(model, optimizer="sgd")
        dataset = build_sam3_dataset(
            model, num_samples=8, batch_size=4, max_instances=N_MAX,
            zero_instance_rate=0.5, seed=23)
        history = model.fit(dataset, epochs=1, verbose=0)
        assert np.isfinite(history.history["loss"][0])
        assert model.jit_compile is False
