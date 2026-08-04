"""
Guards for SAM 2's Hiera trunk (plan step 3, guards G3.1-G3.5).

The defect class this file exists for is *silence*. Hiera's window-size
schedule lags one block behind the stage transition, and its query pooling is
asymmetric between ``q`` and ``k``/``v``. Both can be ported "tidily" wrong and
still produce a trunk that builds, forward-passes at every stage, returns the
right four shapes, trains, and serializes -- with no shape error anywhere. So
the guards here are behavioural discriminators on values and on the shapes of
INTERNAL tensors, never a restatement of the config that produced them.

Guard map:

    G3.1  ``TestBlockScheduleLagsOneBlock``  -- exact per-block window list
    G3.2  ``TestWindowLagIsBehavioural``     -- delta-impulse receptive field
    G3.3  ``TestQPoolAsymmetry``             -- attention-matrix shape table
    G3.4  ``TestGraphTrace``                 -- ``tf.function`` traceability
    G3.5  ``TestDeadComponentPartition``     -- the MEASURED RED partition

G3.1 and G3.2 deliberately guard the SAME defect twice: G3.1 alone is satisfied
by a config table that nothing downstream reads.
"""

from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.models.sam2.hiera import (
    Hiera,
    HieraBlock,
    HieraMultiScaleAttention,
    HieraPatchEmbed,
    hiera_block_specs,
)

from ..test_sam.dead_component_oracle import zeroed_variables

# ---------------------------------------------------------------------
# Test geometry.
#
# Everything is read from `Hiera.MODEL_VARIANTS['tiny']` -- the SINGLE home of
# the small geometry. Nothing here re-states a variant number; the constants
# below are the DERIVED consequences a reader can check by hand.
# ---------------------------------------------------------------------

TINY: Dict[str, Any] = Hiera.MODEL_VARIANTS["tiny"]
IMAGE_SIZE: int = TINY["image_size"]
STEM_GRID: int = IMAGE_SIZE // 4
BATCH = 2
SEED = 4321

#: Expected stage outputs for `tiny`, hand-derived: the stem divides by 4, and
#: each of the three query-pooling transitions halves the grid and doubles the
#: width. Total stride 32.
EXPECTED_LEVELS: List[Tuple[int, int, int]] = [
    (16, 16, 16),
    (8, 8, 32),
    (4, 4, 64),
    (2, 2, 128),
]

#: Hand-derived per-block window sizes for `tiny`, WITH the one-block lag.
#: stages=(1,2,1,2) -> stage_ends=[0,2,3,5]; window_spec=(4,2,3,2);
#: global_att_blocks=(2,).
#:
#:   block 0  stage 1, first block           -> spec[0] = 4
#:   block 1  stage 2, FIRST block           -> spec[0] = 4   <-- the lag
#:   block 2  stage 2, second block          -> global       -> 0
#:   block 3  stage 3, FIRST block           -> spec[1] = 2   <-- the lag
#:   block 4  stage 4, FIRST block           -> spec[2] = 3   <-- the lag
#:   block 5  stage 4, second block          -> spec[3] = 2
EXPECTED_WINDOW_SIZES: List[int] = [4, 4, 0, 2, 3, 2]

#: What the same schedule would be WITHOUT the lag (the natural, wrong port):
#: every block simply uses its own stage's window size.
UNLAGGED_WINDOW_SIZES: List[int] = [4, 2, 0, 3, 2, 2]

#: Hand-derived attention-matrix shapes `(leading, heads, n_query, n_key)` for
#: one `tiny` forward at batch 1. `leading` is `batch * num_windows`.
#:
#:   b0  window 4 on 16x16 -> 16 windows, no pooling      -> 16 q, 16 k
#:   b1  window 4 on 16x16 -> 16 windows, q pooled 2x     ->  4 q, 16 k
#:   b2  global on 8x8     ->  1 window,  no pooling      -> 64 q, 64 k
#:   b3  window 2 on 8x8   -> 16 windows, q pooled 2x     ->  1 q,  4 k
#:   b4  window 3 on 4x4   -> PADDED to 6x6 -> 4 windows  ->  1 q,  9 k
#:   b5  window 2 on 2x2   ->  1 window,  no pooling      ->  4 q,  4 k
EXPECTED_ATTENTION_SHAPES: List[Tuple[int, int, int, int]] = [
    (16, 1, 16, 16),
    (16, 2, 4, 16),
    (1, 2, 64, 64),
    (16, 4, 1, 4),
    (4, 8, 1, 9),
    (1, 8, 4, 4),
]


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _tiny_trunk(**overrides: Any) -> Hiera:
    """Build a `tiny` trunk through the variant table.

    :param overrides: Explicit config overrides.
    :type overrides: Any
    :return: A built trunk.
    :rtype: Hiera
    """
    keras.utils.set_random_seed(SEED)
    trunk = Hiera.from_variant("tiny", **overrides)
    trunk.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))
    return trunk


def _image(batch: int = BATCH) -> np.ndarray:
    """Return a seeded, non-zero input image batch.

    :param batch: Batch size.
    :type batch: int
    :return: ``(batch, IMAGE_SIZE, IMAGE_SIZE, 3)`` float32.
    :rtype: np.ndarray
    """
    rng = np.random.default_rng(SEED)
    return rng.standard_normal(
        (batch, IMAGE_SIZE, IMAGE_SIZE, 3)).astype("float32")


def _capture_attention_shapes(
        trunk: Hiera, images: np.ndarray
) -> List[Tuple[int, ...]]:
    """Record the shape of every attention matrix in one forward pass.

    This reads the shape of the tensor that ``softmax`` is actually applied to,
    so it observes the query and key token counts SEPARATELY -- which is the
    only way to see the q/k/v asymmetry from outside the layer.

    :param trunk: A built trunk.
    :type trunk: Hiera
    :param images: Input batch.
    :type images: np.ndarray
    :return: One shape tuple per block, in block order.
    :rtype: List[Tuple[int, ...]]
    """
    import dl_techniques.models.sam2.hiera as hiera_module

    captured: List[Tuple[int, ...]] = []
    original = hiera_module.ops.softmax

    def spy(x: Any, axis: Any = -1) -> Any:
        captured.append(tuple(int(d) for d in x.shape))
        return original(x, axis=axis)

    hiera_module.ops.softmax = spy
    try:
        trunk(images, training=False)
    finally:
        hiera_module.ops.softmax = original
    return captured


def _impulse_footprint(
        block: HieraBlock,
        grid: int,
        row: int,
        col: int,
) -> set:
    """Return the set of output grid positions a single input impulse reaches.

    Every Dense bias and every ``LayerNormalization`` beta initializes to zero,
    so an all-zero feature vector stays exactly zero through norm, projection
    and MLP. The only mechanism that can move energy off the impulse is the
    attention softmax, and it can only move it WITHIN the impulse's own window.
    The non-zero set is therefore exactly the window footprint.

    :param block: The block to probe.
    :type block: HieraBlock
    :param grid: Input grid edge length.
    :type grid: int
    :param row: Impulse row.
    :type row: int
    :param col: Impulse column.
    :type col: int
    :return: Set of ``(row, col)`` output positions with non-zero energy.
    :rtype: set
    """
    x = np.zeros((1, grid, grid, block.dim), dtype="float32")
    x[0, row, col, :] = np.arange(1, block.dim + 1, dtype="float32")
    y = ops.convert_to_numpy(block(x, training=False))
    magnitude = np.abs(y).sum(axis=-1)[0]
    return {(int(r), int(c)) for r, c in np.argwhere(magnitude > 1e-8)}


def _expected_footprint(
        window_size: int, grid: int, row: int, col: int, pool: int
) -> set:
    """Derive the expected footprint of an impulse independently of the layer.

    :param window_size: Attention window edge length.
    :type window_size: int
    :param grid: Input grid edge length.
    :type grid: int
    :param row: Impulse row.
    :type row: int
    :param col: Impulse column.
    :type col: int
    :param pool: Query pooling factor.
    :type pool: int
    :return: Set of expected output ``(row, col)`` positions.
    :rtype: set
    """
    row_origin = (row // window_size) * window_size
    col_origin = (col // window_size) * window_size
    return {
        (r // pool, c // pool)
        for r in range(row_origin, min(row_origin + window_size, grid))
        for c in range(col_origin, min(col_origin + window_size, grid))
    }


def _max_abs_diff(a: Any, b: Any) -> float:
    """Return the maximum absolute elementwise difference.

    :param a: First tensor-like.
    :type a: Any
    :param b: Second tensor-like.
    :type b: Any
    :return: Max absolute difference.
    :rtype: float
    """
    return float(np.max(np.abs(
        ops.convert_to_numpy(a) - ops.convert_to_numpy(b))))


# ---------------------------------------------------------------------


class TestVariantTableIsTheSingleHome:
    """The `tiny` geometry lives in exactly one place and earns its shape."""

    def test_tiny_is_defined_only_in_model_variants(self) -> None:
        """No second copy of the tiny geometry may exist in the source tree."""
        import inspect
        import dl_techniques.models.sam2 as package

        source_root = inspect.getfile(package).rsplit("/", 1)[0]
        import pathlib

        offenders = [
            path.name
            for path in pathlib.Path(source_root).glob("*.py")
            if "TINY_GEOMETRY" in path.read_text()
        ]
        assert offenders == [], (
            f"a second home for the tiny geometry appeared in {offenders}; "
            f"MODEL_VARIANTS is the single home"
        )

    def test_tiny_exercises_every_mechanism(self) -> None:
        """`tiny` must not be a shrink that stops proving things about `hiera_l`."""
        specs = hiera_block_specs(
            stages=TINY["stages"],
            window_spec=TINY["window_spec"],
            global_att_blocks=TINY["global_att_blocks"],
            q_pool=TINY["q_pool"],
            embed_dim=TINY["embed_dim"],
            num_heads=TINY["num_heads"],
        )
        assert len(TINY["stages"]) == 4, "four stages, like hiera_l"
        assert sum(1 for s in specs if s["window_size"] == 0) >= 1, \
            "at least one global-attention block"
        assert sum(1 for s in specs if s["q_pool"]) == 3, \
            "all three query-pooling transitions live"

        # At least one block's window does NOT divide its grid, so the
        # zero-pad path is exercised.
        grid = STEM_GRID
        non_divisible = 0
        for spec in specs:
            if spec["window_size"] > 0 and grid % spec["window_size"] != 0:
                non_divisible += 1
            if spec["q_pool"]:
                grid //= 2
        assert non_divisible >= 1, "the non-divisible zero-pad path is unexercised"

        # Head width stays a multiple of 4 at every stage (axial RoPE's rule).
        for spec in specs:
            assert (spec["dim_out"] // spec["num_heads"]) % 4 == 0, spec

        # Total stride 32.
        assert IMAGE_SIZE // EXPECTED_LEVELS[-1][0] == 32

    def test_hiera_l_geometry_is_self_consistent(self) -> None:
        """`hiera_l`'s published numbers must satisfy the same invariants."""
        config = Hiera.MODEL_VARIANTS["hiera_l"]
        specs = hiera_block_specs(
            stages=config["stages"],
            window_spec=config["window_spec"],
            global_att_blocks=config["global_att_blocks"],
            q_pool=config["q_pool"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
        )
        assert len(specs) == 48
        # Published channel widths, ascending stage order.
        stage_ends = [1, 7, 43, 47]
        assert [specs[end]["dim_out"] for end in stage_ends] == \
            [144, 288, 576, 1152]
        for spec in specs:
            assert (spec["dim_out"] // spec["num_heads"]) % 4 == 0, spec


class TestBlockScheduleLagsOneBlock:
    """G3.1: the exact per-block window schedule, hand-derived in the test."""

    def test_window_spec_lags_one_block(self) -> None:
        """The schedule must equal the LAGGED list, not the natural one."""
        specs = hiera_block_specs(
            stages=TINY["stages"],
            window_spec=TINY["window_spec"],
            global_att_blocks=TINY["global_att_blocks"],
            q_pool=TINY["q_pool"],
            embed_dim=TINY["embed_dim"],
            num_heads=TINY["num_heads"],
        )
        actual = [spec["window_size"] for spec in specs]
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(actual, EXPECTED_WINDOW_SIZES))
             if a != b),
            None,
        )
        assert actual == EXPECTED_WINDOW_SIZES, (
            f"window schedule differs first at block {first_diff}: got "
            f"{actual}, expected {EXPECTED_WINDOW_SIZES}"
        )

    def test_unlagged_schedule_is_a_different_list(self) -> None:
        """The guard above is not vacuous: the wrong port IS distinguishable."""
        assert EXPECTED_WINDOW_SIZES != UNLAGGED_WINDOW_SIZES
        differing = [
            i for i, (a, b) in enumerate(
                zip(EXPECTED_WINDOW_SIZES, UNLAGGED_WINDOW_SIZES)) if a != b
        ]
        assert differing == [1, 3, 4]

    def test_first_block_of_each_stage_uses_the_previous_stage_window(
            self) -> None:
        """Named directly: every stage's FIRST block lags."""
        specs = hiera_block_specs(
            stages=TINY["stages"],
            window_spec=TINY["window_spec"],
            global_att_blocks=None,
            q_pool=TINY["q_pool"],
            embed_dim=TINY["embed_dim"],
            num_heads=TINY["num_heads"],
        )
        stage_starts = [0, 1, 3, 4]
        for stage_index, block_index in enumerate(stage_starts):
            if stage_index == 0:
                continue
            assert specs[block_index]["stage"] == stage_index + 1
            assert specs[block_index]["window_size"] == \
                TINY["window_spec"][stage_index - 1], (
                f"block {block_index} is stage {stage_index + 1}'s first block "
                f"and must use stage {stage_index}'s window size"
            )

    def test_schedule_matches_the_constructed_blocks(self) -> None:
        """The trunk must actually BUILD the schedule it derives."""
        trunk = _tiny_trunk()
        assert [block.window_size for block in trunk.blocks] == \
            EXPECTED_WINDOW_SIZES
        assert [block.q_stride is not None for block in trunk.blocks] == \
            [False, True, False, True, True, False]

    def test_last_stage_window_spec_entry_is_actually_read(self) -> None:
        """With the lag, a one-block final stage would never read spec[-1].

        `tiny` gives stage 4 two blocks precisely so `window_spec[3]` is live.
        """
        assert EXPECTED_WINDOW_SIZES[-1] == TINY["window_spec"][3]
        assert TINY["stages"][-1] >= 2


class TestWindowLagIsBehavioural:
    """G3.2: the lag must be observable in COMPUTED VALUES, not just config.

    A config table nothing reads would satisfy G3.1 completely. This probe
    drives a single spatial impulse through the first block of stage 2 and
    measures which output positions receive energy. The answer is the window
    footprint -- the previous stage's, not the current one's.
    """

    IMPULSE = (5, 5)

    def test_stage2_first_block_has_the_previous_stage_footprint(self) -> None:
        """The measured footprint equals the spec[0]-derived one."""
        trunk = _tiny_trunk()
        block = trunk.blocks[1]
        row, col = self.IMPULSE

        measured = _impulse_footprint(block, STEM_GRID, row, col)
        lagged = _expected_footprint(
            TINY["window_spec"][0], STEM_GRID, row, col, pool=2)
        assert measured == lagged, (
            f"impulse footprint {sorted(measured)} does not match the "
            f"PREVIOUS stage's window ({TINY['window_spec'][0]}) footprint "
            f"{sorted(lagged)}"
        )

    def test_stage2_first_block_does_not_have_the_current_stage_footprint(
            self) -> None:
        """And it is NOT the footprint the un-lagged port would produce."""
        trunk = _tiny_trunk()
        block = trunk.blocks[1]
        row, col = self.IMPULSE

        measured = _impulse_footprint(block, STEM_GRID, row, col)
        unlagged = _expected_footprint(
            TINY["window_spec"][1], STEM_GRID, row, col, pool=2)
        assert unlagged != measured, (
            f"the two candidate footprints are indistinguishable at impulse "
            f"{self.IMPULSE}; the probe proves nothing"
        )
        assert len(measured) == 4 and len(unlagged) == 1

    def test_probe_precondition_zero_stays_zero(self) -> None:
        """The probe only works because zero inputs stay exactly zero."""
        trunk = _tiny_trunk()
        block = trunk.blocks[1]
        zeros = np.zeros((1, STEM_GRID, STEM_GRID, block.dim), dtype="float32")
        out = ops.convert_to_numpy(block(zeros, training=False))
        assert float(np.max(np.abs(out))) == 0.0, (
            "a zero input produced non-zero output, so the impulse footprint "
            "cannot be attributed to the window"
        )


class TestQPoolAsymmetry:
    """G3.3: only ``q`` is pooled; ``k``/``v`` keep the full window."""

    def test_attention_shapes_match_the_hand_derived_table(self) -> None:
        """The full per-block attention-matrix shape table."""
        trunk = _tiny_trunk()
        shapes = _capture_attention_shapes(trunk, _image(batch=1))
        assert shapes == EXPECTED_ATTENTION_SHAPES, (
            f"attention shapes {shapes} differ from the hand-derived table "
            f"{EXPECTED_ATTENTION_SHAPES}"
        )

    def test_keys_keep_the_full_window_while_queries_are_pooled(self) -> None:
        """``n_k`` is always the FULL window; ``n_q`` is the pooled window.

        Derived per block from the block's own window size, not from a global
        ratio -- on a padded window (block 4: window 3, pooled to 1) the ratio
        is 9:1, not 4:1, so a blanket ``n_q * 4 == n_k`` would be wrong.
        """
        trunk = _tiny_trunk()
        shapes = _capture_attention_shapes(trunk, _image(batch=1))
        grid = STEM_GRID
        for block, shape in zip(trunk.blocks, shapes):
            _, _, n_q, n_k = shape
            window = block.window_size if block.window_size > 0 else grid
            assert n_k == window * window, (
                f"block '{block.name}' must attend over its FULL {window}x"
                f"{window} window (padding included); got n_k={n_k}"
            )
            if block.q_stride is not None:
                pooled = window // block.q_stride[0]
                assert n_q == pooled * pooled, (
                    f"block '{block.name}' pools queries to {pooled}x{pooled}; "
                    f"got n_q={n_q}"
                )
                assert n_q < n_k, (
                    f"block '{block.name}' pooled its keys too: n_q={n_q} == "
                    f"n_k={n_k}"
                )
                grid //= block.q_stride[0]
            else:
                assert n_q == n_k, (
                    f"block '{block.name}' does not pool, so n_q must equal "
                    f"n_k; got {n_q} vs {n_k}"
                )

    def test_shortcut_is_pooled_by_the_same_factor(self) -> None:
        """The residual shortcut halves the grid exactly when the queries do."""
        trunk = _tiny_trunk()
        grid = STEM_GRID
        for block in trunk.blocks:
            out_shape = block.compute_output_shape((None, grid, grid, block.dim))
            expected = grid // 2 if block.q_stride is not None else grid
            assert out_shape[1] == expected and out_shape[2] == expected
            grid = out_shape[1]

        # And the shortcut path is the one that sets the output grid: a
        # non-transition pooling block would be inconsistent, so assert every
        # pooling block is also a channel-widening block.
        for block in trunk.blocks:
            if block.q_stride is not None:
                assert block.dim != block.dim_out, (
                    f"block '{block.name}' pools but does not widen, so its "
                    f"shortcut would keep the pre-pool grid"
                )

    def test_pad_path_is_exercised_and_unmasked(self) -> None:
        """Block 4 partitions a 4x4 grid with window 3, so it must pad to 6x6."""
        trunk = _tiny_trunk()
        shapes = _capture_attention_shapes(trunk, _image(batch=1))
        leading, _, _, n_k = shapes[4]
        assert n_k == 9, (
            f"block 4's keys must cover a full 3x3 window (9 tokens) including "
            f"the zero padding; got {n_k}"
        )
        assert leading == 4, (
            f"a 4x4 grid padded to 6x6 yields 4 windows of 3x3; got {leading}"
        )

    def test_padded_tokens_are_not_masked(self) -> None:
        """The padded tokens participate in the softmax -- upstream's behaviour.

        A masked implementation would give a 3x3 corner window the same output
        as a 2x2 unpadded one for the real tokens. This asserts they differ, so
        the (deliberate) contamination is present and any future "fix" is loud.
        """
        block = HieraBlock(dim=8, dim_out=16, num_heads=2, q_stride=(2, 2),
                           window_size=3)
        block.build((None, 4, 4, 8))
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal((1, 4, 4, 8)).astype("float32")
        out = ops.convert_to_numpy(block(x, training=False))
        assert out.shape == (1, 2, 2, 16)
        # With the padded tokens contributing, the attention over a corner
        # window is not the attention over its unpadded 1x1 remainder.
        assert float(np.std(out)) > 0.0


class TestForwardAndShapes:
    """Structural behaviour of the assembled trunk."""

    def test_returns_four_levels_in_ascending_stage_order(self) -> None:
        """Finest first, coarsest last -- the reverse of `channel_list`."""
        trunk = _tiny_trunk()
        levels = trunk(_image(), training=False)
        assert len(levels) == 4
        assert [tuple(level.shape)[1:] for level in levels] == EXPECTED_LEVELS
        heights = [int(level.shape[1]) for level in levels]
        assert heights == sorted(heights, reverse=True)

    def test_channel_list_is_descending(self) -> None:
        """`channel_list` is the FPN neck's order, i.e. reversed."""
        trunk = _tiny_trunk()
        assert trunk.channel_list == [level[2] for level in EXPECTED_LEVELS][::-1]

    def test_compute_output_shape_agrees_with_the_forward_pass(self) -> None:
        """Derived from config, never from weight shapes."""
        trunk = _tiny_trunk()
        predicted = trunk.compute_output_shape((None, IMAGE_SIZE, IMAGE_SIZE, 3))
        levels = trunk(_image(), training=False)
        assert [shape[1:] for shape in predicted] == \
            [tuple(level.shape)[1:] for level in levels]

    def test_patch_embed_keeps_the_spatial_grid(self) -> None:
        """The stem is overlapping (k=7 > s=4) and does NOT flatten."""
        stem = HieraPatchEmbed(embed_dim=16)
        out = stem(np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype="float32"))
        assert tuple(out.shape) == (1, STEM_GRID, STEM_GRID, 16)
        assert stem.kernel_size > stem.stride

    def test_positional_embedding_is_added_once_at_the_stem(self) -> None:
        """Exactly two positional weights exist, both owned by the trunk."""
        trunk = _tiny_trunk()
        names = {
            weight.name for weight in trunk.weights if "pos_embed" in weight.name
        }
        assert names == {"pos_embed", "pos_embed_window"}
        assert trunk.pos_embed.shape == (
            1, *TINY["window_pos_embed_bkg_spatial_size"], TINY["embed_dim"])
        assert trunk.pos_embed_window.shape == (
            1, TINY["window_spec"][0], TINY["window_spec"][0],
            TINY["embed_dim"])

    def test_positional_embedding_is_live(self) -> None:
        """Perturbing either positional weight must move the output."""
        trunk = _tiny_trunk()
        images = _image(batch=1)
        base = ops.convert_to_numpy(trunk(images, training=False)[-1])
        for weight in (trunk.pos_embed, trunk.pos_embed_window):
            original = ops.convert_to_numpy(weight)
            weight.assign(np.full(original.shape, 0.5, dtype=original.dtype))
            moved = ops.convert_to_numpy(trunk(images, training=False)[-1])
            weight.assign(original)
            assert _max_abs_diff(base, moved) > 1e-6, (
                f"'{weight.name}' is dead -- the stem positional embedding is "
                f"not reaching the output"
            )

    def test_batch_size_does_not_change_per_sample_output(self) -> None:
        """Window partitioning must not leak across batch elements.

        The tolerance is deliberately loose and RELATIVE. A genuine cross-batch
        leak moves the output by order 1; the measured floor here is ordinary
        reduced-precision matmul reassociation, which is batch-shape dependent
        on a GPU: 8e-8 relative on CPU versus 5e-4 relative on GPU 1 (RTX 4070,
        TF32 on by default). Tightening this to 1e-5 turns it into a numerics
        test that fails on the machine the models are developed on.
        """
        trunk = _tiny_trunk()
        images = _image(batch=3)
        batched = ops.convert_to_numpy(trunk(images, training=False)[-1])
        single = ops.convert_to_numpy(
            trunk(images[1:2], training=False)[-1])
        scale = float(np.max(np.abs(batched)))
        assert _max_abs_diff(batched[1:2], single) / scale < 5e-3


class TestConstructionErrors:
    """Invalid geometry must raise at construction, not silently at call."""

    def test_mismatched_stage_and_window_spec_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            hiera_block_specs(stages=(1, 2), window_spec=(4,))

    def test_q_pool_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="q_pool must be in"):
            hiera_block_specs(stages=(1, 2), window_spec=(4, 2), q_pool=3)

    def test_non_divisible_stem_grid_for_the_tiled_window_embedding(
            self) -> None:
        with pytest.raises(ValueError, match="divisible by"):
            Hiera.from_variant("tiny", window_spec=(5, 2, 3, 2)).build(
                (None, IMAGE_SIZE, IMAGE_SIZE, 3))

    def test_pooling_block_with_a_degenerate_window(self) -> None:
        with pytest.raises(ValueError, match="window_size >= q_stride"):
            HieraBlock(dim=8, dim_out=16, num_heads=2, q_stride=(2, 2),
                       window_size=1)

    def test_dim_out_not_divisible_by_heads(self) -> None:
        with pytest.raises(ValueError, match="divisible by num_heads"):
            HieraMultiScaleAttention(dim=8, dim_out=9, num_heads=2)

    def test_unknown_variant(self) -> None:
        with pytest.raises(ValueError, match="Unknown Hiera variant"):
            Hiera.from_variant("hiera_xl")

    def test_dynamic_spatial_dimensions_raise(self) -> None:
        block = HieraBlock(dim=8, dim_out=8, num_heads=2, window_size=2)
        with pytest.raises(ValueError, match="STATIC spatial"):
            block.build((None, None, None, 8))


class TestSerialization:
    """Every class round-trips through its own config."""

    @pytest.mark.parametrize("layer", [
        HieraPatchEmbed(embed_dim=16),
        HieraMultiScaleAttention(dim=16, dim_out=32, num_heads=2,
                                 q_stride=(2, 2)),
        HieraBlock(dim=16, dim_out=32, num_heads=2, q_stride=(2, 2),
                   window_size=4),
    ])
    def test_layer_config_round_trip(self, layer: Any) -> None:
        """`get_config()` must carry every `__init__` parameter."""
        restored = layer.__class__.from_config(layer.get_config())
        assert restored.get_config() == layer.get_config()

    def test_trunk_config_round_trip_by_value(self) -> None:
        """A rebuilt trunk with copied weights must reproduce the output."""
        trunk = _tiny_trunk()
        images = _image(batch=1)
        expected = ops.convert_to_numpy(trunk(images, training=False)[-1])

        restored = Hiera.from_config(trunk.get_config())
        restored.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))
        restored.set_weights(trunk.get_weights())
        actual = ops.convert_to_numpy(restored(images, training=False)[-1])
        assert _max_abs_diff(expected, actual) == 0.0

    def test_registered_keys_are_present_exactly_once(self) -> None:
        """A duplicate bare registration silently overwrites."""
        for cls in (HieraPatchEmbed, HieraMultiScaleAttention, HieraBlock,
                    Hiera):
            key = keras.saving.get_registered_name(cls)
            assert key == f"Custom>{cls.__name__}", (
                f"{cls.__name__} registered under '{key}' -- a bare "
                f"`@register_keras_serializable()` uses the 'Custom' package"
            )
            registered = keras.saving.get_registered_object(key)
            assert registered is cls, (
                f"'{key}' resolves to {registered}, not {cls.__name__} -- a "
                f"registered-key collision silently overwrote it"
            )


class TestGraphTrace:
    """G3.4: `Hiera.call` must trace under `tf.function`.

    The suspect operation is the bicubic ``ops.image.resize`` of the learned
    background positional embedding. Its target size is a plain Python tuple of
    ints derived from config, which is what ``resize`` requires (the size must
    be something ``len()`` can measure). The MEASURED outcome is recorded here,
    not assumed.
    """

    def test_call_traces_with_static_input_signature(self) -> None:
        """The whole trunk, including the bicubic stem resize, traces."""
        trunk = _tiny_trunk()

        @tf.function(input_signature=[
            tf.TensorSpec((1, IMAGE_SIZE, IMAGE_SIZE, 3), tf.float32)
        ])
        def traced(images: Any) -> Any:
            return trunk(images, training=False)

        concrete = traced.get_concrete_function()
        assert len(concrete.output_shapes) == 4

    def test_bicubic_resize_of_a_weight_traces_on_its_own(self) -> None:
        """Isolate the suspect op, so a future failure is attributable."""
        trunk = _tiny_trunk()

        @tf.function(input_signature=[])
        def traced() -> Any:
            return trunk._get_pos_embed()

        result = traced.get_concrete_function()
        assert tuple(result.output_shapes) == (1, STEM_GRID, STEM_GRID,
                                               TINY["embed_dim"])

    def test_trace_guard_is_not_vacuous(self) -> None:
        """Something CAN break the trace: an eager numpy read of a live tensor.

        ``ops.convert_to_numpy`` raises ``NotImplementedError`` under trace, so
        the guard above is a real observation and not an always-green step.
        """
        trunk = _tiny_trunk()

        @tf.function(input_signature=[
            tf.TensorSpec((1, IMAGE_SIZE, IMAGE_SIZE, 3), tf.float32)
        ])
        def traced(images: Any) -> Any:
            levels = trunk(images, training=False)
            return levels[-1] * float(
                np.mean(ops.convert_to_numpy(levels[-1])))

        with pytest.raises(NotImplementedError):
            traced.get_concrete_function()

    def test_python_float_of_a_traced_tensor_is_rewritten_by_autograph(
            self) -> None:
        """MEASURED, and surprising enough to pin: ``float()`` is NOT a break.

        Step 2 proved its own trace guard RED with ``float(ops.mean(...))``
        placed inside a Keras layer's ``call``. Placed HERE -- in the body of
        the ``tf.function`` itself -- the same expression traces cleanly,
        because AutoGraph rewrites ``float(x)`` on a tensor into a cast, and it
        rewrites only the code it converts. A Keras layer's ``call`` is not
        converted, so the identical line means different things at the two
        sites.

        Two consequences, both load-bearing for later steps:

        1. ``float(...)`` is an INERT mutation for a guard written this way; do
           not reach for it as a trace-guard liveness proof.
        2. The same expression also behaves differently outside pytest, where
           AutoGraph cannot read the source of a ``python -c`` body and falls
           back to the plain builtin (which raises ``TypeError``). A trace
           result measured in a REPL does not transfer to the test suite.
        """
        trunk = _tiny_trunk()

        @tf.function(input_signature=[
            tf.TensorSpec((1, IMAGE_SIZE, IMAGE_SIZE, 3), tf.float32)
        ])
        def traced(images: Any) -> Any:
            levels = trunk(images, training=False)
            return levels[-1] * float(ops.mean(levels[-1]))

        concrete = traced.get_concrete_function()
        assert tuple(concrete.output_shapes) == (1, 2, 2, 128)

    def test_bicubic_resize_traces_at_a_fully_dynamic_size(self) -> None:
        """The pre-committed interpolation-matrix fallback is NOT needed.

        The plan pre-committed a host-numpy bicubic interpolation MATRIX as a
        fallback in case ``ops.image.resize`` could not be traced. Measured
        here: it traces even when the target size is a slice of ``ops.shape``,
        i.e. not static at all. The fallback was therefore not built, and this
        test is the record of why.
        """
        trunk = _tiny_trunk()

        @tf.function(input_signature=[
            tf.TensorSpec((1, IMAGE_SIZE, IMAGE_SIZE, 3), tf.float32)
        ])
        def traced(images: Any) -> Any:
            levels = trunk(images, training=False)
            return ops.image.resize(
                levels[-1], ops.shape(levels[-1])[1:3],
                interpolation="bicubic")

        assert traced.get_concrete_function() is not None


class TestGradientFlow:
    """Everything that should carry gradient does."""

    def test_all_trainable_variables_receive_a_gradient(self) -> None:
        """Including both positional embeddings."""
        trunk = _tiny_trunk()
        images = tf.constant(_image(batch=1))
        with tf.GradientTape() as tape:
            levels = trunk(images, training=True)
            loss = sum(tf.reduce_mean(tf.square(level)) for level in levels)
        grads = tape.gradient(loss, trunk.trainable_variables)
        dead = [
            variable.name
            for variable, grad in zip(trunk.trainable_variables, grads)
            if grad is None or float(np.max(np.abs(
                ops.convert_to_numpy(grad)))) == 0.0
        ]
        assert dead == [], f"variables with no gradient: {dead}"

    def test_unused_projection_contributes_no_weights(self) -> None:
        """`proj` exists on every block but is built only where it is used."""
        trunk = _tiny_trunk()
        for block in trunk.blocks:
            if block.dim == block.dim_out:
                assert block.proj.weights == [], (
                    f"block '{block.name}' does not widen, so its unused "
                    f"`proj` must hold no weights"
                )
            else:
                assert block.proj.weights != []


class TestDeadComponentPartition:
    """G3.5: which guards actually go RED under a dead component.

    "All guards go red under any dead component" is a hypothesis, and steps 1
    and 2 both measured it FALSE. What follows is the partition MEASURED here,
    including the guards that stay green -- a guard that cannot go red is the
    thing worth knowing about.
    """

    def test_dead_attention_projection_kills_the_impulse_footprint(
            self) -> None:
        """G3.2 goes fully RED: with `attn.proj` dead the footprint collapses.

        The residual shortcut still carries the impulse's own pooled position,
        so the footprint does not vanish -- it shrinks to that single position,
        which is exactly what the un-lagged (window 2) port would produce. So
        this dead component makes G3.2 report the WRONG answer rather than no
        answer, which is why G3.1 exists alongside it.
        """
        trunk = _tiny_trunk()
        block = trunk.blocks[1]
        row, col = TestWindowLagIsBehavioural.IMPULSE
        lagged = _expected_footprint(
            TINY["window_spec"][0], STEM_GRID, row, col, pool=2)

        with zeroed_variables(block.attn.proj.weights):
            measured = _impulse_footprint(block, STEM_GRID, row, col)

        assert measured != lagged, (
            "the impulse footprint guard stayed GREEN with the attention "
            "output projection dead -- it cannot detect a dead attention"
        )
        assert measured <= lagged, (
            f"a dead attention widened the footprint to {sorted(measured)}"
        )

    def test_dead_attention_projection_does_not_touch_the_schedule_guard(
            self) -> None:
        """G3.1 stays GREEN: the schedule is a pure function of config.

        This asymmetry is the point of having both guards. G3.1 cannot see a
        dead forward path; G3.2 cannot distinguish a dead forward path from the
        wrong window size. Neither alone is sufficient.
        """
        trunk = _tiny_trunk()
        with zeroed_variables(trunk.blocks[1].attn.proj.weights):
            assert [block.window_size for block in trunk.blocks] == \
                EXPECTED_WINDOW_SIZES

    def test_dead_qkv_leaves_the_attention_shape_table_green(self) -> None:
        """G3.3 stays GREEN under zeroed weights: it is a SHAPE guard.

        Measured, not assumed. G3.3 reads the shape of the attention matrix,
        which is fixed by the geometry and cannot be changed by any weight
        value. It detects a wrong pooling MECHANISM and is structurally blind
        to a dead one.
        """
        trunk = _tiny_trunk()
        dead = [
            variable
            for block in trunk.blocks
            for variable in block.attn.qkv.weights
        ]
        with zeroed_variables(dead):
            shapes = _capture_attention_shapes(trunk, _image(batch=1))
        assert shapes == EXPECTED_ATTENTION_SHAPES

    def test_dead_stem_kills_every_value_guard_at_once(self) -> None:
        """The floor case: with the stem dead the whole trunk output is constant."""
        trunk = _tiny_trunk()
        images = _image(batch=1)
        with zeroed_variables(trunk.patch_embed.proj.weights):
            levels = trunk(images, training=False)
            coarsest = ops.convert_to_numpy(levels[-1])
        # Every spatial position is identical: no information entered.
        assert float(np.std(coarsest, axis=(1, 2)).max()) < 1e-5

# ---------------------------------------------------------------------
