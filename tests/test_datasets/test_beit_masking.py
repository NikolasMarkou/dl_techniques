"""Tests for BEiT's block-wise masking generator and its tf.data map fn.

Covers SC-8 of plan-2026-08-11T012340-f63796dc: the generator reproduces the
reference's STRUCTURE (budget ceiling, strict block rejection, silent under-fill,
log-uniform aspect sampling) and ``make_beit_mim_map_fn`` emits a
``sample_weight`` that is exactly the mask, with fully static shapes, inside a
real ``tf.data`` pipeline.

The oracle for every structural claim here is
``plans/.../findings/beit-pretraining-details-web.md`` section 1, which carries
the reference ``masking_generator.py`` verbatim -- not the port under test.

Numbers are the paper's real geometry (14x14 = 196 patches, budget 75, minimum
block area 16) wherever the test does not specifically need a pathological grid.
"""

import math
import random

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.vision.beit_masking import (
    BEIT_MIN_MASK_PATCHES_PER_BLOCK,
    BEIT_NUM_MASK_PATCHES,
    BeitMaskingGenerator,
    make_beit_mim_map_fn,
)

# BEiT-Base/16 @224 geometry. Do not shrink these.
GRID = 14
NUM_PATCHES = GRID * GRID  # 196
BUDGET = BEIT_NUM_MASK_PATCHES  # 75
MIN_BLOCK = BEIT_MIN_MASK_PATCHES_PER_BLOCK  # 16

IMAGE_SIZE = 224
PATCH_SIZE = 16
CHANNELS = 3


def _make(seed: int = 0, **overrides) -> BeitMaskingGenerator:
    kwargs = dict(
        input_size=GRID,
        num_masking_patches=BUDGET,
        min_num_patches=MIN_BLOCK,
        rng=random.Random(seed),
    )
    kwargs.update(overrides)
    return BeitMaskingGenerator(**kwargs)


# ---------------------------------------------------------------------
# Constants and construction
# ---------------------------------------------------------------------


class TestBeitDefaults:
    """The exported constants are BEiT v1's real CLI defaults, not invented."""

    def test_the_cli_defaults_are_the_papers_numbers(self):
        assert BEIT_NUM_MASK_PATCHES == 75
        assert BEIT_MIN_MASK_PATCHES_PER_BLOCK == 16
        # 75 / 196 = 38.3%, which the paper rounds to "roughly 40%".
        assert 0.38 < BEIT_NUM_MASK_PATCHES / NUM_PATCHES < 0.39

    def test_scalar_input_size_is_a_square_grid(self):
        gen = _make()
        assert gen.get_shape() == (GRID, GRID)
        assert gen.num_patches == NUM_PATCHES

    def test_non_square_grid_is_height_width(self):
        gen = _make(input_size=(6, 9), num_masking_patches=20, min_num_patches=4)
        assert gen.get_shape() == (6, 9)
        assert gen.num_patches == 54
        assert gen().shape == (6, 9)

    def test_max_num_patches_defaults_to_the_budget(self):
        # Reference: `self.max_num_patches = num_masking_patches if max_num_patches
        # is None else max_num_patches`.
        assert _make().max_num_patches == BUDGET
        assert _make(min_num_patches=4, max_num_patches=9).max_num_patches == 9

    def test_max_aspect_defaults_to_the_reciprocal_and_the_range_is_logged(self):
        gen = _make(min_aspect=0.3)
        assert gen.max_aspect == pytest.approx(1.0 / 0.3)
        assert gen.log_aspect_ratio == (
            pytest.approx(math.log(0.3)),
            pytest.approx(math.log(1.0 / 0.3)),
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"input_size": 0},
            {"input_size": (0, 4)},
            {"input_size": (1, 2, 3)},
            {"num_masking_patches": 0},
            {"num_masking_patches": NUM_PATCHES + 1},
            {"min_num_patches": 0},
            {"min_num_patches": 80, "max_num_patches": 40},
            {"min_aspect": 0.0},
            {"min_aspect": 2.0, "max_aspect": 1.0},
        ],
    )
    def test_invalid_configuration_raises_at_construction(self, overrides):
        with pytest.raises(ValueError):
            _make(**overrides)


# ---------------------------------------------------------------------
# The mask itself
# ---------------------------------------------------------------------


class TestMaskStructure:
    """Shape / dtype / values / budget, over many draws."""

    def test_shape_dtype_and_binary_values(self):
        gen = _make(seed=7)
        mask = gen()
        assert mask.shape == (GRID, GRID)
        # np.int64 explicitly: the reference's `np.int` was REMOVED in numpy
        # >= 1.24 and would raise here.
        assert mask.dtype == np.int64
        assert set(np.unique(mask)).issubset({0, 1})

    def test_mask_count_never_exceeds_the_budget(self):
        # 200 independent draws from one seeded stream.
        gen = _make(seed=11)
        counts = [int(gen().sum()) for _ in range(200)]
        assert max(counts) <= BUDGET, f"budget breached: max={max(counts)}"
        # And it is not trivially empty -- an all-zero generator would satisfy
        # the ceiling above while masking nothing.
        assert min(counts) > 0
        assert np.mean(counts) > 0.5 * BUDGET

    def test_blocks_are_contiguous_not_iid(self):
        """A block-wise mask is not an i.i.d. mask; prove it is clustered.

        Under i.i.d. masking at p = 75/196 the expected fraction of
        horizontally-adjacent equal pairs is p^2 + (1-p)^2 = 0.53. Block-wise
        masking is far more clustered than that.
        """
        gen = _make(seed=13)
        agreements = []
        for _ in range(50):
            m = gen()
            agreements.append(float(np.mean(m[:, :-1] == m[:, 1:])))
        assert np.mean(agreements) > 0.75, np.mean(agreements)

    def test_no_single_block_ever_spans_a_full_grid_dimension(self):
        """The reference's rejection is STRICT: ``w < width and h < height``.

        This is asserted per BLOCK, on a single ``_mask`` round starting from an
        empty grid, because it is NOT a property of the finished mask: two
        legally-placed blocks side by side can jointly fill a row, and asserting
        on the union would fail against a correct port.

        A port that used ``<=`` would place full-width / full-height stripes,
        which this catches on the first such draw.
        """
        height, width = 6, 7
        gen = BeitMaskingGenerator(
            input_size=(height, width), num_masking_patches=25,
            min_num_patches=4, rng=random.Random(3),
        )
        rounds_with_a_block = 0
        for _ in range(400):
            grid = np.zeros((height, width), dtype=np.int64)
            if gen._mask(grid, 25) == 0:
                continue
            rounds_with_a_block += 1
            rows = np.flatnonzero(grid.sum(axis=1))
            cols = np.flatnonzero(grid.sum(axis=0))
            block_h = rows[-1] - rows[0] + 1
            block_w = cols[-1] - cols[0] + 1
            # A single accepted block is a solid rectangle.
            assert grid.sum() == block_h * block_w, f"not a rectangle:\n{grid}"
            assert block_h < height, f"block spans the full height:\n{grid}"
            assert block_w < width, f"block spans the full width:\n{grid}"
        assert rounds_with_a_block > 300, rounds_with_a_block

    def test_under_fill_returns_short_and_does_not_raise(self):
        """A grid too small to admit ANY block must return, not raise.

        Construction: on a 3x3 grid with min_num_patches == num_masking_patches
        == 9, every candidate draws ``target_area = 9`` and produces ``h, w``
        with ``h * w ~ 9``. Both dimensions can only be < 3 if ``h * w <= 4``,
        which rounding cannot reach from 9, so EVERY candidate is rejected by the
        strict ``h < 3, w < 3`` test, ``_mask`` returns delta == 0, and
        ``__call__`` breaks out with an all-zero mask.

        THIS IS THE BEHAVIOUR MOST LIKELY TO BE "HELPFULLY" FIXED by a future
        editor -- by raising, by retrying forever, or by topping the mask up with
        i.i.d. cells. All three would diverge from the reference
        (``masking_generator.py`` has no error path and no fallback), and a
        retry-forever fix would hang training rather than fail it. If this test
        fails, the port changed, not the test.
        """
        gen = BeitMaskingGenerator(
            input_size=3, num_masking_patches=9, min_num_patches=9,
            rng=random.Random(5),
        )
        mask = gen()  # must not raise
        assert mask.shape == (3, 3)
        assert int(mask.sum()) < 9
        assert int(mask.sum()) == 0

    def test_partial_under_fill_is_also_silent(self):
        """The softer case: some cells placed, budget still not reached."""
        # Budget 8 on a 4x4 grid with a minimum block area of 6: blocks are
        # comparatively large relative to the grid, so rounds fail often.
        gen = BeitMaskingGenerator(
            input_size=4, num_masking_patches=8, min_num_patches=6,
            rng=random.Random(17),
        )
        counts = [int(gen().sum()) for _ in range(200)]
        assert max(counts) <= 8
        assert min(counts) < 8, "expected at least one under-filled draw"


# ---------------------------------------------------------------------
# Aspect-ratio sampling
# ---------------------------------------------------------------------


class _RecordingRandom(random.Random):
    """A ``random.Random`` that records the aspect-ratio draws it served.

    The aspect draw is identified by its ARGUMENTS -- it is the only
    ``uniform`` call made with the generator's ``log_aspect_ratio`` pair -- so
    this observes the sampling without reaching into the port's internals.
    """

    def __init__(self, seed, log_range):
        super().__init__(seed)
        self._log_range = log_range
        self.aspects = []

    def uniform(self, a, b):
        value = super().uniform(a, b)
        if (a, b) == self._log_range:
            self.aspects.append(math.exp(value))
        return value


class TestAspectRatioSampling:
    def test_sampling_reaches_both_extremes_and_is_log_uniform(self):
        min_aspect, max_aspect = 0.3, 1.0 / 0.3
        log_range = (math.log(min_aspect), math.log(max_aspect))
        rec = _RecordingRandom(23, log_range)
        gen = BeitMaskingGenerator(
            input_size=GRID, num_masking_patches=BUDGET,
            min_num_patches=MIN_BLOCK, min_aspect=min_aspect, rng=rec,
        )
        for _ in range(300):
            gen()

        aspects = np.array(rec.aspects)
        assert aspects.size > 1000, aspects.size
        assert aspects.min() >= min_aspect
        assert aspects.max() <= max_aspect
        # Both extremes are reached (within 5% of the range ends).
        span = max_aspect - min_aspect
        assert aspects.min() < min_aspect + 0.05 * span
        assert aspects.max() > max_aspect - 0.05 * span

        # LOG-uniform, not uniform: the median of a log-uniform draw over
        # [0.3, 1/0.3] is sqrt(0.3 * 1/0.3) == 1.0, whereas a uniform draw over
        # the same interval has median (0.3 + 3.333)/2 == 1.817. This assertion
        # separates the two.
        assert np.median(aspects) == pytest.approx(1.0, abs=0.15)
        assert np.mean(np.log(aspects)) == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------
# The rng injection (an addition over the reference, not a reference property)
# ---------------------------------------------------------------------


class TestRngInjection:
    def test_same_injected_seed_gives_identical_masks(self):
        a = _make(seed=1234)
        b = _make(seed=1234)
        for _ in range(20):
            np.testing.assert_array_equal(a(), b())

    def test_different_injected_seeds_diverge(self):
        a = _make(seed=1)
        b = _make(seed=2)
        assert not np.array_equal(a(), b())

    def test_no_determinism_is_claimed_without_an_injected_rng(self):
        """Default ``rng=None`` draws from the GLOBAL ``random`` module.

        The reference does the same, and this port makes NO seeding promise for
        that path: two default generators share the process-global stream, so
        their draws interleave rather than repeat. Seeding the global module
        makes them reproducible -- that is a property of ``random.seed``, not a
        contract of this class, and it is asserted here only to pin what the
        default path actually is.
        """
        a = BeitMaskingGenerator(GRID, BUDGET, min_num_patches=MIN_BLOCK)
        b = BeitMaskingGenerator(GRID, BUDGET, min_num_patches=MIN_BLOCK)
        assert a._rng is random and b._rng is random

        random.seed(99)
        first = a()
        random.seed(99)
        second = b()
        np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------
# The tf.data map fn
# ---------------------------------------------------------------------


def _fake_tokenizer(image: tf.Tensor) -> tf.Tensor:
    """A stand-in visual tokenizer: one code id per patch, in ``[0, 8192)``.

    Built from TF ops only, because the real one runs inside a tf.data graph.
    Returns a ``(GRID, GRID)`` grid, exercising the map fn's reshape contract.

    The reduction is deliberately INTEGER: a float patch-mean is
    order-of-summation dependent, so recomputing it eagerly (possibly on GPU) to
    build an oracle disagrees with the tf.data (CPU) evaluation in the last ulp,
    which then flips an occasional id after the cast. Integer sums are exact and
    device-independent, so the oracle is an oracle rather than a coin flip.
    """
    ints = tf.cast(image * 255.0, tf.int32)
    grid = tf.reshape(ints, (GRID, PATCH_SIZE, GRID, PATCH_SIZE, CHANNELS))
    ids = tf.reduce_sum(grid, axis=[1, 3, 4])  # (GRID, GRID)
    return tf.math.floormod(ids, 8192)


@pytest.fixture()
def images() -> tf.Tensor:
    rng = np.random.default_rng(20260811)
    return tf.constant(
        rng.uniform(size=(8, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)).astype("float32")
    )


class TestMakeBeitMimMapFn:
    def test_element_spec_is_fully_static_after_the_py_function_wrap(self, images):
        """A ``numpy_function`` returns UNKNOWN shape; the fn must re-pin it.

        Without the ``ensure_shape`` the mask's spec is ``(None,)``, batching
        still "works", and the breakage surfaces much later as a shape error
        deep inside the model. Assert the spec directly.
        """
        map_fn = make_beit_mim_map_fn(
            tokenizer_fn=_fake_tokenizer, grid_size=GRID,
            num_masking_patches=BUDGET, min_num_patches=MIN_BLOCK,
            rng=random.Random(0),
        )
        ds = tf.data.Dataset.from_tensor_slices(images).map(map_fn)
        (img_spec, mask_spec), ids_spec, w_spec = ds.element_spec

        for spec in (img_spec, mask_spec, ids_spec, w_spec):
            assert None not in spec.shape.as_list(), spec

        assert img_spec.shape.as_list() == [IMAGE_SIZE, IMAGE_SIZE, CHANNELS]
        assert mask_spec.shape.as_list() == [NUM_PATCHES]
        assert ids_spec.shape.as_list() == [NUM_PATCHES]
        assert w_spec.shape.as_list() == [NUM_PATCHES]
        assert mask_spec.dtype == tf.bool
        assert ids_spec.dtype == tf.int32
        assert w_spec.dtype == tf.float32

    def test_runs_inside_a_real_map_batch_pipeline(self, images):
        """Eager-callable is not enough -- a py_function can still break under map."""
        map_fn = make_beit_mim_map_fn(
            tokenizer_fn=_fake_tokenizer, grid_size=GRID,
            num_masking_patches=BUDGET, min_num_patches=MIN_BLOCK,
            rng=random.Random(1),
        )
        ds = (
            tf.data.Dataset.from_tensor_slices(images)
            .map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(4)
        )
        batches = list(ds)
        assert len(batches) == 2
        for (img, mask), ids, weight in batches:
            assert img.shape == (4, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
            assert mask.shape == (4, NUM_PATCHES)
            assert ids.shape == (4, NUM_PATCHES)
            assert weight.shape == (4, NUM_PATCHES)

    def test_sample_weight_is_elementwise_equal_to_the_mask(self, images):
        map_fn = make_beit_mim_map_fn(
            tokenizer_fn=_fake_tokenizer, grid_size=GRID,
            num_masking_patches=BUDGET, min_num_patches=MIN_BLOCK,
            rng=random.Random(2),
        )
        ds = tf.data.Dataset.from_tensor_slices(images).map(map_fn).batch(8)
        (_, mask), _, weight = next(iter(ds))

        np.testing.assert_array_equal(
            weight.numpy(), mask.numpy().astype("float32")
        )
        # 1.0 exactly at masked positions, 0.0 exactly elsewhere -- NOT rescaled
        # by N / n_masked the way ``masked_patches`` does it.
        assert set(np.unique(weight.numpy())).issubset({0.0, 1.0})
        assert weight.numpy().sum(axis=-1).max() <= BUDGET

    def test_the_mask_is_drawn_per_element_not_frozen_once(self, images):
        """A non-stateful py_function would be folded to ONE constant mask."""
        map_fn = make_beit_mim_map_fn(
            tokenizer_fn=_fake_tokenizer, grid_size=GRID,
            num_masking_patches=BUDGET, min_num_patches=MIN_BLOCK,
            rng=random.Random(4),
        )
        ds = tf.data.Dataset.from_tensor_slices(images).map(map_fn).batch(8)
        (_, mask), _, _ = next(iter(ds))
        rows = {row.tobytes() for row in mask.numpy()}
        assert len(rows) == 8, "the same mask was reused across elements"

    def test_targets_come_from_the_supplied_tokenizer(self, images):
        """The map fn must not invent ids -- they are the tokenizer's, reshaped."""
        map_fn = make_beit_mim_map_fn(
            tokenizer_fn=_fake_tokenizer, grid_size=GRID,
            num_masking_patches=BUDGET, min_num_patches=MIN_BLOCK,
            rng=random.Random(6),
        )
        ds = tf.data.Dataset.from_tensor_slices(images).map(map_fn)
        (_, _), ids, _ = next(iter(ds))

        expected = tf.reshape(_fake_tokenizer(images[0]), (NUM_PATCHES,))
        np.testing.assert_array_equal(ids.numpy(), expected.numpy())

    def test_a_flat_tokenizer_output_is_accepted_too(self, images):
        def flat_tokenizer(image):
            return tf.reshape(_fake_tokenizer(image), (-1,))

        map_fn = make_beit_mim_map_fn(
            tokenizer_fn=flat_tokenizer, grid_size=GRID,
            num_masking_patches=BUDGET, min_num_patches=MIN_BLOCK,
            rng=random.Random(8),
        )
        ds = tf.data.Dataset.from_tensor_slices(images).map(map_fn)
        (_, _), ids, _ = next(iter(ds))
        assert ids.shape == (NUM_PATCHES,)

    def test_invalid_config_raises_when_the_pipeline_is_built(self):
        with pytest.raises(ValueError):
            make_beit_mim_map_fn(
                tokenizer_fn=_fake_tokenizer,
                grid_size=GRID,
                num_masking_patches=NUM_PATCHES + 1,
            )

    def test_the_element_feeds_a_stock_masked_cross_entropy(self, images):
        """H-8: the loss sees ONLY masked positions, with no custom train_step.

        Stock ``SparseCategoricalCrossentropy(from_logits=True)`` + the emitted
        ``sample_weight`` must equal a hand-computed mean over masked positions
        only (Keras' ``sum_over_batch_size`` divides by B * N, so the reference
        value divides by the same denominator).
        """
        vocab = 32

        def small_tokenizer(image):
            return tf.math.floormod(_fake_tokenizer(image), vocab)

        map_fn = make_beit_mim_map_fn(
            tokenizer_fn=small_tokenizer, grid_size=GRID,
            num_masking_patches=BUDGET, min_num_patches=MIN_BLOCK,
            rng=random.Random(10),
        )
        ds = tf.data.Dataset.from_tensor_slices(images).map(map_fn).batch(4)
        (_, mask), ids, weight = next(iter(ds))

        rng = np.random.default_rng(0)
        logits = tf.constant(
            rng.normal(size=(4, NUM_PATCHES, vocab)).astype("float32")
        )
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        got = float(loss_fn(ids, logits, sample_weight=weight))

        per_token = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )(ids, logits).numpy()
        m = mask.numpy()
        expected = float((per_token * m).sum() / m.size)

        assert got == pytest.approx(expected, rel=1e-5)
        # And the guard has teeth: the unmasked positions really are excluded.
        assert expected < float(per_token.mean())
