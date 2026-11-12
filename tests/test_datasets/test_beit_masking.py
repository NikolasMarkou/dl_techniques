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

import hashlib
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf

import dl_techniques
from dl_techniques.datasets.vision.beit_masking import (
    BEIT_MIN_MASK_PATCHES_PER_BLOCK,
    BEIT_NUM_MASK_PATCHES,
    BeitMaskingGenerator,
    make_beit_mim_map_fn,
    make_stateless_beit_mim_map_fn,
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


# ---------------------------------------------------------------------
# An explicit rng on one call
# ---------------------------------------------------------------------


class _CountingRandom(random.Random):
    """A ``random.Random`` that counts the draws it served."""

    def __init__(self, seed):
        super().__init__(seed)
        self.draws = 0

    def uniform(self, a, b):
        self.draws += 1
        return super().uniform(a, b)

    def randint(self, a, b):
        self.draws += 1
        return super().randint(a, b)


def _five_masks_digest(generator: BeitMaskingGenerator, **call) -> str:
    digest = hashlib.sha256()
    for _ in range(5):
        digest.update(generator(**call).tobytes())
    return digest.hexdigest()[:16]


class TestAnExplicitRngOnOneCall:
    """``__call__(rng=)``: the draw comes from the rng given, no shared state moves."""

    def test_the_default_draw_is_the_one_it_always_was(self):
        """Digests of five masks, taken at the commit before ``rng=`` existed."""
        assert _five_masks_digest(_make(seed=0)) == "d3c9152fc9ab133d"
        assert _five_masks_digest(_make(seed=1234)) == "26746a0a5fec30c8"

    def test_the_seeded_global_state_gives_the_old_mask(self):
        generator = BeitMaskingGenerator(GRID, BUDGET, min_num_patches=MIN_BLOCK)
        random.seed(99)
        assert _five_masks_digest(generator) == "6307f85e103fcefe"
        random.seed(99)
        assert _five_masks_digest(generator, rng=None) == "6307f85e103fcefe"

    def test_an_explicit_rng_is_drawn_from(self):
        generator = _make(seed=1)
        mine = _CountingRandom(5)
        mask = generator(rng=mine)
        assert mine.draws > 0
        np.testing.assert_array_equal(mask, _make(seed=5)())

    def test_an_explicit_rng_touches_no_shared_state(self):
        own = random.Random(1)
        generator = BeitMaskingGenerator(
            GRID, BUDGET, min_num_patches=MIN_BLOCK, rng=own
        )
        own_before = own.getstate()
        global_before = random.getstate()
        generator(rng=random.Random(2))
        assert own.getstate() == own_before
        assert random.getstate() == global_before
        assert generator._rng is own

    def test_the_global_module_is_untouched_by_an_explicit_rng_on_a_default_generator(self):
        generator = BeitMaskingGenerator(GRID, BUDGET, min_num_patches=MIN_BLOCK)
        before = random.getstate()
        generator(rng=random.Random(2))
        assert random.getstate() == before

    def test_the_mask_depends_on_the_rng_given_and_not_on_the_generator_s_own(self):
        a = _make(seed=1)
        b = _make(seed=2)
        np.testing.assert_array_equal(a(rng=random.Random(7)), b(rng=random.Random(7)))


# ---------------------------------------------------------------------
# The stateless map function
# ---------------------------------------------------------------------


def _reference_mask(grid, budget, minimum, seed, salt, index) -> np.ndarray:
    """The documented key, drawn by a fresh generator: no shared state at all."""
    generator = BeitMaskingGenerator(
        grid, budget, min_num_patches=minimum,
        rng=random.Random(f"{seed}:{salt}:{index}"),
    )
    return generator().reshape(-1).astype(np.bool_)


def _zero_ids(count: int):
    return lambda image: tf.zeros((count,), tf.int32)


def _stream(fn, image, count: int, parallel=tf.data.AUTOTUNE):
    """One in-order pass of ``count`` rows of one image through ``fn``; masks, ids, weights."""
    ds = (
        tf.data.Dataset.from_tensors(image)
        .repeat(count)
        .enumerate()
        .map(fn, num_parallel_calls=parallel, deterministic=True)
    )
    masks, ids, weights = [], [], []
    for (_image, mask), target, weight in ds:
        masks.append(mask.numpy())
        ids.append(target.numpy())
        weights.append(weight.numpy())
    return np.stack(masks), np.stack(ids), np.stack(weights)


_SMALL_IMAGE = np.zeros((32, 32, 3), "float32")  # 8x8 grid at patch 4 (cifar10)

_DIGEST_SOURCE = """
import hashlib
import tensorflow as tf
from dl_techniques.datasets.vision.beit_masking import make_stateless_beit_mim_map_fn

def digest():
    fn = make_stateless_beit_mim_map_fn(
        lambda image: tf.zeros((64,), tf.int32), 8, 24, 5, seed=42, salt="train")
    image = tf.zeros((32, 32, 3))
    h = hashlib.sha256()
    for i in range(60):
        (_, mask), _, weight = fn(tf.constant(i, tf.int64), image)
        h.update(mask.numpy().tobytes())
        h.update(weight.numpy().tobytes())
    return h.hexdigest()
"""


class TestTheStatelessBeitMapFn:
    def test_the_mask_is_the_one_the_documented_key_draws(self):
        fn = make_stateless_beit_mim_map_fn(
            _zero_ids(64), 8, 24, 5, seed=42, salt="train"
        )
        masks, _, _ = _stream(fn, _SMALL_IMAGE, 40)
        for index in range(40):
            np.testing.assert_array_equal(
                masks[index], _reference_mask(8, 24, 5, 42, "train", index)
            )

    def test_two_passes_are_equal_under_autotune(self):
        fn = make_stateless_beit_mim_map_fn(
            _zero_ids(64), 8, 24, 5, seed=42, salt="train"
        )
        first = _stream(fn, _SMALL_IMAGE, 200)
        second = _stream(fn, _SMALL_IMAGE, 200)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_a_fresh_interpreter_gives_the_same_masks(self, tmp_path):
        namespace = {}
        exec(_DIGEST_SOURCE, namespace)
        here = namespace["digest"]()
        src = str(Path(dl_techniques.__file__).resolve().parents[1])
        env = dict(os.environ, PYTHONPATH=src, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3")
        env.pop("PYTHONHASHSEED", None)
        out = subprocess.run(
            [sys.executable, "-c", _DIGEST_SOURCE + "\nprint('DIGEST', digest())"],
            capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=300,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        assert [l for l in out.stdout.splitlines() if l.startswith("DIGEST ")] == [
            f"DIGEST {here}"
        ]

    def test_a_different_index_salt_or_seed_gives_a_different_mask(self):
        def masks(seed, salt):
            fn = make_stateless_beit_mim_map_fn(_zero_ids(64), 8, 24, 5, seed=seed, salt=salt)
            return _stream(fn, _SMALL_IMAGE, 24)[0]

        base = masks(42, "train")
        assert len({row.tobytes() for row in base}) > 20  # one mask per index
        assert not np.any(np.all(base == masks(42, "select"), axis=1))
        assert not np.any(np.all(base == masks(43, "train"), axis=1))

    def test_seed_zero_is_a_seed(self):
        fn_zero = make_stateless_beit_mim_map_fn(_zero_ids(64), 8, 24, 5, seed=0, salt="train")
        masks, _, _ = _stream(fn_zero, _SMALL_IMAGE, 8)
        for index in range(8):
            np.testing.assert_array_equal(
                masks[index], _reference_mask(8, 24, 5, 0, "train", index)
            )

    def test_the_element_has_the_legacy_structure_and_a_fully_static_spec(self, images):
        fn = make_stateless_beit_mim_map_fn(
            _fake_tokenizer, GRID, BUDGET, MIN_BLOCK, seed=1, salt="train"
        )
        ds = tf.data.Dataset.from_tensor_slices(images).enumerate().map(fn)
        (img_spec, mask_spec), ids_spec, w_spec = ds.element_spec
        assert img_spec.shape.as_list() == [IMAGE_SIZE, IMAGE_SIZE, CHANNELS]
        assert mask_spec.shape.as_list() == [NUM_PATCHES] and mask_spec.dtype == tf.bool
        assert ids_spec.shape.as_list() == [NUM_PATCHES] and ids_spec.dtype == tf.int32
        assert w_spec.shape.as_list() == [NUM_PATCHES] and w_spec.dtype == tf.float32

    def test_the_targets_come_from_the_supplied_tokenizer(self, images):
        fn = make_stateless_beit_mim_map_fn(
            _fake_tokenizer, GRID, BUDGET, MIN_BLOCK, seed=1, salt="train"
        )
        ds = tf.data.Dataset.from_tensor_slices(images).enumerate().map(fn)
        for (image, _mask), ids, _weight in ds.take(3):
            np.testing.assert_array_equal(
                ids.numpy(), _fake_tokenizer(image).numpy().reshape(-1)
            )

    def test_the_weight_is_the_mask_times_N_over_the_masked_count(self, images):
        fn = make_stateless_beit_mim_map_fn(
            _fake_tokenizer, GRID, BUDGET, MIN_BLOCK, seed=1, salt="train"
        )
        ds = tf.data.Dataset.from_tensor_slices(images).enumerate().map(fn)
        for (_image, mask), _ids, weight in ds:
            mask, weight = mask.numpy(), weight.numpy()
            expected = mask.astype(np.float32) * np.float32(NUM_PATCHES / max(int(mask.sum()), 1))
            np.testing.assert_allclose(weight, expected, rtol=1e-6, atol=0)
            assert weight.dtype == np.float32
            assert weight[~mask].max() == 0.0
            np.testing.assert_allclose(weight.sum(), NUM_PATCHES, rtol=1e-5)

    def test_an_under_filled_mask_is_re_weighted_by_what_it_masked(self):
        """4x4 grid, budget 8, blocks of at least 6: some draws stop short."""
        fn = make_stateless_beit_mim_map_fn(
            _zero_ids(16), 4, 8, 6, seed=17, salt="train"
        )
        image = np.zeros((8, 8, 3), "float32")  # patch 2
        masks, _, weights = _stream(fn, image, 200)
        counts = masks.sum(axis=1)
        short = np.flatnonzero((counts > 0) & (counts < 8))
        assert short.size > 0, "expected at least one under-filled draw"
        for row in short[:20]:
            np.testing.assert_allclose(
                weights[row], masks[row] * np.float32(16 / counts[row]), rtol=1e-6
            )
        # the same weight at a full budget is a different number: the mask count matters
        full = np.flatnonzero(counts == 8)
        assert full.size > 0
        assert not np.allclose(weights[short[0]].max(), weights[full[0]].max())

    def test_an_all_false_mask_weighs_zero_and_divides_by_nothing(self):
        """3x3 grid, one block would be the whole grid: every candidate is rejected."""
        fn = make_stateless_beit_mim_map_fn(_zero_ids(9), 3, 9, 9, seed=5, salt="train")
        image = np.zeros((6, 6, 3), "float32")
        masks, _, weights = _stream(fn, image, 6)
        assert not masks.any()
        assert np.all(weights == 0.0)
        assert np.isfinite(weights).all()

    def test_concurrent_calls_share_no_state(self):
        """600 elements under 16 parallel calls and a 1 microsecond thread switch.

        At the default switch interval a generator whose ``_rng`` is swapped per call
        corrupts none of 2000 rows (measured), so this MUST run with the interval
        tightened or the planted defect passes.
        """
        count = 600
        reference = np.stack(
            [_reference_mask(8, 24, 5, 42, "train", i) for i in range(count)]
        )
        interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            fn = make_stateless_beit_mim_map_fn(
                _zero_ids(64), 8, 24, 5, seed=42, salt="train"
            )
            for _pass in range(3):
                masks, _, _ = _stream(fn, _SMALL_IMAGE, count, parallel=16)
                wrong = int((masks != reference).any(axis=1).sum())
                assert wrong == 0, f"{wrong} of {count} masks differ from their own key"
        finally:
            sys.setswitchinterval(interval)

    def test_the_element_feeds_a_stock_cross_entropy_as_the_mean_masked_token_loss(self, images):
        vocab = 32
        fn = make_stateless_beit_mim_map_fn(
            lambda image: tf.math.floormod(_fake_tokenizer(image), vocab),
            GRID, BUDGET, MIN_BLOCK, seed=3, salt="train",
        )
        ds = tf.data.Dataset.from_tensor_slices(images).enumerate().map(fn).batch(4)
        (_, mask), ids, weight = next(iter(ds))
        logits = tf.constant(
            np.random.default_rng(0).normal(size=(4, NUM_PATCHES, vocab)).astype("float32")
        )
        got = float(
            keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
                ids, logits, sample_weight=weight
            )
        )
        per_token = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )(ids, logits).numpy()
        m = mask.numpy()
        per_example = (per_token * m).sum(axis=1) / m.sum(axis=1)  # mean over MASKED tokens
        assert got == pytest.approx(float(per_example.mean()), rel=1e-5)

    def test_a_bad_budget_is_refused_when_the_function_is_built(self):
        with pytest.raises(ValueError, match="exceeds the grid size"):
            make_stateless_beit_mim_map_fn(
                _zero_ids(64), 8, 65, 5, seed=0, salt="train"
            )

    def test_the_key_is_validated(self):
        for bad in (None, 1.5, "42", True):
            with pytest.raises(TypeError, match="seed"):
                make_stateless_beit_mim_map_fn(_zero_ids(64), 8, 24, 5, seed=bad, salt="train")
        for bad in (None, "", 3):
            with pytest.raises(TypeError, match="salt"):
                make_stateless_beit_mim_map_fn(_zero_ids(64), 8, 24, 5, seed=1, salt=bad)
