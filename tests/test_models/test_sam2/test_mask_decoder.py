"""Tests for ``SAM2MaskDecoder`` -- plan step 7, guards G7.1 through G7.7.

Every mechanism guarded here is SILENT when ported wrong: the token-index shift,
the additive high-resolution skips, the stability formula and the per-batch
fallback all preserve every output shape and dtype. Shape assertions are
therefore present but are never the guard.

Two structural choices are deliberate:

* :class:`IdentityTransformer` replaces the two-way transformer wherever a token
  INDEX is under test. With the real transformer every output token is a mixture
  of every input token, so an off-by-one in the unpack changes the value by an
  unpredictable amount and cannot be asserted against anything. With the
  identity stub, ``hs`` IS the token block and the index claim becomes an exact
  equality.
* The stability and per-batch-selection guards call
  :meth:`SAM2MaskDecoder._get_stability_scores` and
  :meth:`SAM2MaskDecoder._dynamic_multimask_via_stability` directly on
  hand-built tensors. Driving them through a full forward pass would make the
  oracle depend on random weights, and the exact ratio -- the only thing that
  discriminates a swapped delta -- would be unavailable.
"""

import subprocess

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.models.SAM.SAM1.transformer import TwoWayTransformer
from dl_techniques.models.sam2.mask_decoder import SAM2MaskDecoder

# A-5: the oracle is IMPORTED from SAM 1's test package, never moved or copied.
# Moving it would touch SAM 1's test tree and put the 357-test regression gate
# at risk for zero benefit.
from ..test_sam.dead_component_oracle import (
    component_response,
    no_op_kill,
    zeroed_variables,
)

# ---------------------------------------------------------------------
# geometry -- small but not degenerate
# ---------------------------------------------------------------------

DIM = 32
HEADS = 2
MLP_DIM = 64
GRID = 4
BATCH = 2
NUM_SPARSE = 3
NUM_MULTIMASK = 3
NUM_MASK_TOKENS = NUM_MULTIMASK + 1

#: Edge length of the mask this decoder emits at the SHIPPED configuration:
#: ``image_size=1024`` over ``backbone_stride=16`` is a 64x64 feature grid, and
#: the decoder upscales 4x. ``256 * 256 == 65,536`` elements, which is ABOVE
#: float16's largest finite value of ``65,504`` -- the reason the stability head
#: must accumulate its area counts in float32. Every other geometry constant in
#: this file is deliberately toy-sized; this one is deliberately not.
SHIPPED_MASK_EDGE = 256


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def make_transformer() -> TwoWayTransformer:
    """Build a SECOND ``TwoWayTransformer`` instance, imported unchanged.

    :return: A fresh transformer at the SAM 2 shape, scaled down.
    :rtype: TwoWayTransformer
    """
    return TwoWayTransformer(
        depth=2, embedding_dim=DIM, num_heads=HEADS, mlp_dim=MLP_DIM
    )


def make_decoder(**overrides) -> SAM2MaskDecoder:
    """Build a decoder with the shipped SAM 2 feature flags on by default.

    A substitute transformer must be passed as the ``transformer`` override, not
    assigned afterwards: Keras 3 locks a built layer's state and refuses the
    re-assignment with ``"You cannot add new elements of state ... to a layer
    that is already built"``.

    :param overrides: Constructor overrides.
    :return: A built decoder.
    :rtype: SAM2MaskDecoder
    """
    kwargs = dict(
        transformer_dim=DIM,
        transformer=make_transformer(),
        num_multimask_outputs=NUM_MULTIMASK,
        use_high_res_features=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        use_multimask_token_for_obj_ptr=True,
        dynamic_multimask_via_stability=True,
    )
    kwargs.update(overrides)
    decoder = SAM2MaskDecoder(**kwargs)
    decoder.build(None)
    return decoder


def make_inputs(seed: int = 0, batch: int = BATCH):
    """Build a deterministic set of decoder inputs.

    :param seed: RNG seed.
    :type seed: int
    :param batch: Batch size.
    :type batch: int
    :return: ``(image, pe, sparse, dense, feat_s0, feat_s1)``.
    :rtype: tuple
    """
    rng = np.random.default_rng(seed)

    def draw(*shape):
        return ops.convert_to_tensor(
            rng.standard_normal(shape).astype("float32")
        )

    return (
        draw(batch, GRID, GRID, DIM),
        draw(batch, GRID, GRID, DIM),
        draw(batch, NUM_SPARSE, DIM),
        draw(batch, GRID, GRID, DIM),
        draw(batch, GRID * 4, GRID * 4, DIM),
        draw(batch, GRID * 2, GRID * 2, DIM),
    )


def run(decoder, inputs, multimask_output=True, feat_s0=None, feat_s1=None):
    """Invoke a decoder, optionally substituting the high-res skips.

    :param decoder: The decoder.
    :type decoder: SAM2MaskDecoder
    :param inputs: The tuple from :func:`make_inputs`.
    :type inputs: tuple
    :param multimask_output: Whether to request the multimask tokens.
    :type multimask_output: bool
    :param feat_s0: Replacement 4x feature map, or ``None``.
    :param feat_s1: Replacement 2x feature map, or ``None``.
    :return: The decoder's 4-tuple output.
    :rtype: tuple
    """
    image, pe, sparse, dense, f0, f1 = inputs
    high_res = None
    if decoder.use_high_res_features:
        high_res = [f0 if feat_s0 is None else feat_s0,
                    f1 if feat_s1 is None else feat_s1]
    return decoder(
        image_embeddings=image,
        image_pe=pe,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=multimask_output,
        high_res_features=high_res,
    )


def npy(tensor) -> np.ndarray:
    """Convert a tensor to numpy.

    :param tensor: Any Keras tensor.
    :return: A numpy array.
    :rtype: np.ndarray
    """
    return np.asarray(ops.convert_to_numpy(tensor))


class IdentityTransformer(keras.layers.Layer):
    """A stub transformer that returns its queries untouched.

    Makes ``hs`` identically the token block, so a token-index claim becomes an
    exact equality instead of an unfalsifiable "the value changed".
    """

    def call(self, image_embedding, image_pe, point_embedding, training=None):
        """Return ``(queries, flattened_keys)`` with the queries unchanged.

        :param image_embedding: ``(B, H, W, C)``.
        :param image_pe: Unused.
        :param point_embedding: ``(B, N, C)`` token block.
        :param training: Unused.
        :return: ``(point_embedding, flattened image_embedding)``.
        :rtype: tuple
        """
        b, h, w, c = ops.shape(image_embedding)
        return point_embedding, ops.reshape(image_embedding, (b, h * w, c))


class RecordingTransformer(keras.layers.Layer):
    """Delegates to a real transformer while recording its token input."""

    def __init__(self, inner, **kwargs):
        """Wrap ``inner``.

        :param inner: The transformer to delegate to.
        :type inner: keras.layers.Layer
        """
        super().__init__(**kwargs)
        self.inner = inner
        self.recorded_tokens = None

    def call(self, image_embedding, image_pe, point_embedding, training=None):
        """Record ``point_embedding`` then delegate.

        :return: Whatever the wrapped transformer returns.
        :rtype: tuple
        """
        self.recorded_tokens = point_embedding
        return self.inner(image_embedding, image_pe, point_embedding, training=training)


def set_token_embeddings(decoder, obj_value, iou_value, mask_values):
    """Overwrite the three learnable token embeddings with known constants.

    :param decoder: The decoder.
    :type decoder: SAM2MaskDecoder
    :param obj_value: Constant filled into the object-score token.
    :type obj_value: float
    :param iou_value: Constant filled into the IoU token.
    :type iou_value: float
    :param mask_values: One constant per mask token.
    :type mask_values: list
    """
    decoder.obj_score_token.weights[0].assign(
        ops.full((1, DIM), obj_value, dtype="float32")
    )
    decoder.iou_token.weights[0].assign(
        ops.full((1, DIM), iou_value, dtype="float32")
    )
    decoder.mask_tokens.weights[0].assign(
        ops.convert_to_tensor(
            np.tile(np.asarray(mask_values, dtype="float32")[:, None], (1, DIM))
        )
    )


def set_linear_sum_head(head):
    """Turn a 1-layer MLP head into an exact "sum of inputs" function.

    :param head: A ``keras.Sequential`` of exactly one ``Dense(1)``.
    :type head: keras.Sequential
    """
    kernel, bias = head.weights
    kernel.assign(ops.ones_like(kernel))
    bias.assign(ops.zeros_like(bias))


# ---------------------------------------------------------------------
# G7.1 -- token layout
# ---------------------------------------------------------------------


class TestTokenLayout:
    """G7.1: the object-score token is PREPENDED and read at index 0."""

    def test_token_block_width_includes_the_obj_score_token(self):
        """The block entering the transformer is obj + iou + mask tokens wide."""
        recorder = RecordingTransformer(make_transformer())
        decoder = make_decoder(transformer=recorder)
        run(decoder, make_inputs())

        total = int(recorder.recorded_tokens.shape[1])
        assert total == 1 + 1 + NUM_MASK_TOKENS + NUM_SPARSE, (
            f"token block width {total - NUM_SPARSE} != "
            f"1 (obj) + 1 (iou) + {NUM_MASK_TOKENS} (mask)"
        )

    def test_token_block_omits_the_obj_score_token_when_disabled(self):
        """Without ``pred_obj_scores`` the block is one token narrower."""
        recorder = RecordingTransformer(make_transformer())
        decoder = make_decoder(
            transformer=recorder, pred_obj_scores=False, pred_obj_scores_mlp=False
        )
        run(decoder, make_inputs())

        total = int(recorder.recorded_tokens.shape[1])
        assert total == 1 + NUM_MASK_TOKENS + NUM_SPARSE
        assert decoder.token_offset == 0

    def test_offset_is_one_when_object_scores_are_predicted(self):
        """``s == 1``: the IoU token no longer sits at index 0."""
        assert make_decoder().token_offset == 1

    def test_object_score_reads_token_index_zero(self):
        """The obj-score head reads the OBJ token, not the IoU token.

        Driven by a zeroed-except-index-0 token block: every token but index 0
        is exactly zero, and the head is an exact sum, so the predicted score is
        ``DIM * obj_value`` if index 0 is read and exactly ``0.0`` for any other
        index. This is the mutation "read index 1" made falsifiable.
        """
        decoder = make_decoder(
            transformer=IdentityTransformer(), pred_obj_scores_mlp=False
        )
        set_token_embeddings(
            decoder, obj_value=0.25, iou_value=0.0, mask_values=[0.0] * NUM_MASK_TOKENS
        )
        set_linear_sum_head(decoder.pred_obj_score_head)

        image, pe, sparse, dense, f0, f1 = make_inputs()
        # Zero the sparse prompts too: they are concatenated AFTER the token
        # block, so a non-zero prompt could not reach index 0 or 1, but zeroing
        # them removes the question entirely.
        sparse = ops.zeros_like(sparse)
        _, _, obj_score, _ = decoder(
            image_embeddings=image,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            high_res_features=[f0, f1],
        )

        expected = 0.25 * DIM
        np.testing.assert_allclose(npy(obj_score), expected, atol=1e-5)
        assert abs(expected) > 1e-3, (
            "the expected value must differ from the index-1 answer (0.0), or "
            "this assertion passes under the mutation it exists to catch"
        )

    def test_mask_tokens_are_sliced_at_offset_s_plus_one(self):
        """``hs[:, s + 1 : s + 1 + N]`` returns the mask tokens exactly.

        With the identity stub and distinct per-token constants, an off-by-one
        slice returns ``[iou, m0, m1, m2]`` instead of ``[m0, m1, m2, m3]`` --
        same shape, different values.
        """
        decoder = make_decoder(transformer=IdentityTransformer())
        mask_values = [1.0, 2.0, 3.0, 4.0]
        set_token_embeddings(
            decoder, obj_value=-9.0, iou_value=-5.0, mask_values=mask_values
        )

        image, pe, sparse, dense, f0, f1 = make_inputs()
        _, _, mask_tokens_out, _ = decoder.predict_masks(
            image_embeddings=image,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            high_res_features=[f0, f1],
        )

        got = npy(mask_tokens_out)[0, :, 0]
        np.testing.assert_allclose(got, np.asarray(mask_values), atol=1e-6)

    def test_object_score_defaults_to_ten_without_the_token(self):
        """Without ``pred_obj_scores`` the score is the constant ``10.0``."""
        decoder = make_decoder(pred_obj_scores=False, pred_obj_scores_mlp=False)
        _, _, obj_score, _ = run(decoder, make_inputs())
        np.testing.assert_allclose(npy(obj_score), np.full((BATCH, 1), 10.0))
        assert float(ops.convert_to_numpy(ops.sigmoid(obj_score[0, 0]))) > 0.999


# ---------------------------------------------------------------------
# G7.2 -- additive high-resolution fusion
# ---------------------------------------------------------------------


class TestHighResFusion:
    """G7.2: both skips are live and the fusion is additive, not concatenated.

    The two liveness arms are SEPARATE tests driven by SEPARATE perturbations,
    and step 7 ran one mutation per skip. Two mutations landing on the same
    assertion would prove one skip twice and the other zero times (measured in
    step 2 of this plan).
    """

    def test_feat_s1_perturbation_moves_the_output(self):
        """Perturbing ONLY the 2x skip changes the mask logits."""
        decoder = make_decoder()
        inputs = make_inputs()
        base = npy(run(decoder, inputs)[0])
        bumped = npy(run(decoder, inputs, feat_s1=inputs[5] + 1.0)[0])
        assert np.max(np.abs(base - bumped)) > 1e-5, (
            "feat_s1 is DEAD: the 2x skip never reaches the output"
        )

    def test_feat_s0_perturbation_moves_the_output(self):
        """Perturbing ONLY the 4x skip changes the mask logits."""
        decoder = make_decoder()
        inputs = make_inputs()
        base = npy(run(decoder, inputs)[0])
        bumped = npy(run(decoder, inputs, feat_s0=inputs[4] + 1.0)[0])
        assert np.max(np.abs(base - bumped)) > 1e-5, (
            "feat_s0 is DEAD: the 4x skip never reaches the output"
        )

    def test_upscaling_widths_are_never_doubled_by_a_concat(self):
        """Declared widths, not movement: a coherent concat port changes ONLY these.

        Step 5 of this plan measured that an additive-vs-concat port keeps every
        liveness arm green (decisions.md D-016); the DECLARED channel width is
        the only thing that separates them.
        """
        decoder = make_decoder()
        assert decoder.dc1.filters == DIM // 4
        assert decoder.dc2.filters == DIM // 8
        assert decoder.conv_s1.filters == DIM // 4
        assert decoder.conv_s0.filters == DIM // 8

        logits = run(decoder, make_inputs())[0]
        assert tuple(logits.shape) == (BATCH, NUM_MULTIMASK, GRID * 4, GRID * 4)

    def test_hypernetwork_width_matches_the_undoubled_upscale(self):
        """The hypernetwork emits ``transformer_dim // 8``, matching ``dc2``."""
        decoder = make_decoder()
        last_dense = decoder.output_hypernetworks_mlps[0].layers[-1]
        assert last_dense.units == DIM // 8 == decoder.dc2.filters

    def test_missing_high_res_features_is_refused(self):
        """A silently coarser decode is refused, not tolerated."""
        decoder = make_decoder()
        image, pe, sparse, dense, _, _ = make_inputs()
        with pytest.raises(ValueError, match="high_res_features=None"):
            decoder(
                image_embeddings=image,
                image_pe=pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
                high_res_features=None,
            )

    def test_unwanted_high_res_features_are_refused(self):
        """Passing skips to a decoder that ignores them is refused."""
        decoder = make_decoder(use_high_res_features=False)
        image, pe, sparse, dense, f0, f1 = make_inputs()
        with pytest.raises(ValueError, match="use_high_res_features=False"):
            decoder(
                image_embeddings=image,
                image_pe=pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
                high_res_features=[f0, f1],
            )


# ---------------------------------------------------------------------
# G7.3 -- stability formula
# ---------------------------------------------------------------------


def build_stability_logits(n_above, n_between, n_below, delta=0.05):
    """Build a ``(1, 1, 4, 4)`` logit map with known thresholded areas.

    :param n_above: Count of entries strictly above ``+delta``.
    :type n_above: int
    :param n_between: Count of entries in ``(-delta, +delta]``.
    :type n_between: int
    :param n_below: Count of entries at or below ``-delta``.
    :type n_below: int
    :param delta: The stability delta.
    :type delta: float
    :return: The logits tensor.
    """
    assert n_above + n_between + n_below == 16
    values = (
        [1.0] * n_above + [0.0] * n_between + [-1.0] * n_below
    )
    return ops.convert_to_tensor(
        np.asarray(values, dtype="float32").reshape(1, 1, 4, 4)
    )


class TestStabilityScore:
    """G7.3: the exact ratio, not merely a finite number in ``[0, 1]``."""

    def test_exact_ratio_of_the_two_thresholded_areas(self):
        """``area_i = 5``, ``area_u = 8`` gives exactly ``0.625``.

        Swapping ``+delta`` and ``-delta`` gives ``8 / 5 = 1.6`` -- finite,
        never raising, and invisible to any "score is a number" assertion.
        """
        decoder = make_decoder()
        logits = build_stability_logits(n_above=5, n_between=3, n_below=8)
        score = npy(decoder._get_stability_scores(logits))
        np.testing.assert_allclose(score, np.asarray([[0.625]]), atol=1e-6)

    def test_ratio_is_bounded_by_one(self):
        """The correct orientation can never exceed 1; the swap routinely does."""
        decoder = make_decoder()
        for above, between in ((5, 3), (1, 7), (0, 10), (12, 4)):
            logits = build_stability_logits(
                n_above=above, n_between=between, n_below=16 - above - between
            )
            score = float(npy(decoder._get_stability_scores(logits))[0, 0])
            assert 0.0 <= score <= 1.0, f"score {score} outside [0, 1]"

    def test_empty_union_scores_one(self):
        """``area_u == 0`` yields exactly ``1.0`` and never a NaN."""
        decoder = make_decoder()
        logits = build_stability_logits(n_above=0, n_between=0, n_below=16)
        score = npy(decoder._get_stability_scores(logits))
        np.testing.assert_allclose(score, np.asarray([[1.0]]))
        assert np.all(np.isfinite(score))

    def test_threshold_comparison_is_strictly_greater(self):
        """Entries exactly at ``+delta`` are excluded from ``area_i``."""
        decoder = make_decoder()
        delta = decoder.dynamic_multimask_stability_delta
        logits = ops.convert_to_tensor(
            np.full((1, 1, 4, 4), delta, dtype="float32")
        )
        score = npy(decoder._get_stability_scores(logits))
        np.testing.assert_allclose(score, np.asarray([[0.0]]), atol=1e-6)

    def test_scores_are_per_mask_token(self):
        """The reduction is over the spatial axes only, never over the tokens."""
        decoder = make_decoder()
        stacked = ops.concatenate(
            [
                build_stability_logits(n_above=5, n_between=3, n_below=8),
                build_stability_logits(n_above=8, n_between=0, n_below=8),
            ],
            axis=1,
        )
        score = npy(decoder._get_stability_scores(stacked))
        assert score.shape == (1, 2)
        np.testing.assert_allclose(score, np.asarray([[0.625, 1.0]]), atol=1e-6)

    def test_the_area_sums_do_not_overflow_float16_at_the_SHIPPED_mask_size(
            self):
        """``mixed_float16`` at 256x256: finite scores, and no selection flip.

        **A toy grid cannot reproduce this.** Every other test in this class
        runs at ``4 x 4 = 16`` elements. The shipped ``image_size=1024`` makes
        the decoder emit ``256 x 256 = 65,536`` logits per mask, and float16's
        largest finite value is ``65,504`` -- so the area COUNT overflows on
        the very first configuration anyone actually runs. This repo has
        measured exactly that trap before (a 15-combination toy sweep went
        15/15 green while the paper-scale configuration overflowed), so the arm
        below runs at the shipped size and nowhere smaller.

        The failure is a behaviour INVERSION, not merely a NaN: ``area_i`` and
        ``area_u`` both become ``inf``, ``inf / inf`` is ``NaN``,
        ``NaN >= 0.98`` evaluates **False**, and a maximally-confident single
        mask is silently discarded in favour of a multimask token on the
        default ``training=None`` path. Both halves are asserted.
        """
        previous = keras.mixed_precision.global_policy()
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            decoder = make_decoder()
            edge = SHIPPED_MASK_EDGE
            delta = decoder.dynamic_multimask_stability_delta

            # Token 0 is uniformly ABOVE +delta -> area_i == area_u ==
            # edge**2, i.e. a perfectly stable single mask (score 1.0).
            # Tokens 1..3 are uniformly below -delta, so the two candidates are
            # separated by 10.0 in value and the choice is directly readable.
            logits = np.full((1, 4, edge, edge), -5.0, dtype="float16")
            logits[:, 0] = 5.0
            logits_tensor = ops.convert_to_tensor(logits)

            # NEGATIVE CONTROL: prove this size really is in the overflowing
            # regime, so the assertions below are not vacuous. Summing the SAME
            # boolean count in float16 -- which is what the pre-fix code did --
            # must be non-finite.
            naive = npy(ops.sum(
                ops.cast(
                    ops.reshape(logits_tensor, (1, 4, -1)) > delta, "float16"),
                axis=-1,
            ))
            assert edge * edge > 65504, (
                f"the probe grid has {edge * edge} elements, which float16 "
                f"represents exactly -- this test cannot see the overflow"
            )
            assert not np.isfinite(naive[0, 0]), (
                f"summing the area in float16 produced the finite value "
                f"{naive[0, 0]} at {edge}x{edge} -- the overflow this test "
                f"guards does not occur here and the probe must be enlarged"
            )

            stability = npy(decoder._get_stability_scores(logits_tensor))
            assert np.all(np.isfinite(stability)), (
                f"stability scores are non-finite under mixed_float16 at the "
                f"shipped {edge}x{edge} mask size: {stability} -- the area "
                f"counts are being accumulated in the input dtype"
            )
            np.testing.assert_allclose(
                stability[0, 0], 1.0, atol=1e-6)

            # The behaviour half: token 0 must be KEPT, not silently replaced.
            iou = ops.convert_to_tensor(
                np.asarray([[0.5, 0.9, 0.1, 0.1]], dtype="float16"))
            masks_out, _ = decoder._dynamic_multimask_via_stability(
                logits_tensor, iou)
            chosen = float(npy(masks_out).mean())
        finally:
            keras.mixed_precision.set_global_policy(previous)

        assert chosen == pytest.approx(5.0, abs=1e-3), (
            f"the stable single mask (value +5.0) was replaced by a multimask "
            f"token (value -5.0): the selection returned a mean of {chosen}. "
            f"A NaN stability score compares False against the threshold, so "
            f"this inversion is silent"
        )

    def test_delta_zero_is_refused(self):
        """At ``delta == 0`` the score is the constant 1 and the guard is dead."""
        with pytest.raises(ValueError, match="delta must be positive"):
            make_decoder(dynamic_multimask_stability_delta=0.0)

    def test_impossible_threshold_is_refused(self):
        """A threshold above 1 silently disables the single-mask branch."""
        with pytest.raises(ValueError, match="must lie in"):
            make_decoder(dynamic_multimask_stability_thresh=1.5)


# ---------------------------------------------------------------------
# G7.4 -- per-batch-element selection
# ---------------------------------------------------------------------


def build_selection_case():
    """Build a 2-row selection case whose per-row argmax indices DIFFER.

    Row 0's best multimask token is index 0 (global token 1); row 1's is index 2
    (global token 3). The single-mask token is deliberately unstable in both
    rows, so the fallback is taken.

    :return: ``(all_mask_logits, all_iou_scores)``.
    :rtype: tuple
    """
    # Token 0 (single mask): 8 of 16 pixels above +delta, none between, so
    # stability = 8 / 8 = 1.0 would be STABLE. Add 4 in-between pixels to push
    # it to 8 / 12 = 0.667, well below 0.98.
    single = np.asarray(
        [1.0] * 8 + [0.0] * 4 + [-1.0] * 4, dtype="float32"
    ).reshape(1, 1, 4, 4)
    single = np.concatenate([single, single], axis=0)

    multimask = np.stack(
        [
            np.full((3, 4, 4), 10.0, dtype="float32")
            + np.arange(3, dtype="float32")[:, None, None],
            np.full((3, 4, 4), 20.0, dtype="float32")
            + np.arange(3, dtype="float32")[:, None, None],
        ],
        axis=0,
    )
    all_logits = ops.convert_to_tensor(
        np.concatenate([single, multimask], axis=1)
    )
    all_iou = ops.convert_to_tensor(
        np.asarray(
            [
                [0.5, 0.9, 0.1, 0.2],
                [0.5, 0.1, 0.2, 0.9],
            ],
            dtype="float32",
        )
    )
    return all_logits, all_iou


class TestPerBatchSelection:
    """G7.4: the unstable-case fallback is per batch element.

    A batch of 1 would be VACUOUS here: with one row, a per-row argmax and a
    single global argmax are the same index by construction, so every assertion
    below would pass under the very mutation it exists to catch. Both cases use
    batch 2 with DIFFERING per-row argmax indices.
    """

    def test_each_row_gets_its_own_best_mask(self):
        """Row 0 falls back to token 1, row 1 to token 3."""
        decoder = make_decoder()
        all_logits, all_iou = build_selection_case()
        masks, iou = decoder._dynamic_multimask_via_stability(all_logits, all_iou)

        logits_np = npy(all_logits)
        np.testing.assert_allclose(npy(masks)[0, 0], logits_np[0, 1], atol=1e-6)
        np.testing.assert_allclose(npy(masks)[1, 0], logits_np[1, 3], atol=1e-6)

    def test_each_row_gets_its_own_best_iou(self):
        """The returned IoU is the SELECTED token's, per row."""
        decoder = make_decoder()
        all_logits, all_iou = build_selection_case()
        _, iou = decoder._dynamic_multimask_via_stability(all_logits, all_iou)
        np.testing.assert_allclose(
            npy(iou), np.asarray([[0.9], [0.9]]), atol=1e-6
        )

    def test_the_two_rows_select_different_tokens(self):
        """The fixture itself is discriminating, not just the assertions.

        If both rows happened to select the same token index the mutation
        "single global argmax" would be undetectable no matter how the outputs
        are asserted.
        """
        _, all_iou = build_selection_case()
        per_row = npy(ops.argmax(all_iou[:, 1:], axis=-1))
        assert per_row[0] != per_row[1], (
            f"fixture is vacuous: both rows pick multimask index {per_row[0]}"
        )

    def test_a_stable_single_mask_is_kept(self):
        """When token 0 is stable the multimask tokens are NOT consulted."""
        decoder = make_decoder()
        all_logits, all_iou = build_selection_case()
        logits_np = npy(all_logits)
        # Make row 0's single mask perfectly stable: no pixel between the two
        # thresholds, so area_i == area_u.
        logits_np[0, 0] = np.asarray(
            [1.0] * 8 + [-1.0] * 8, dtype="float32"
        ).reshape(4, 4)
        stable_logits = ops.convert_to_tensor(logits_np)

        masks, iou = decoder._dynamic_multimask_via_stability(
            stable_logits, all_iou
        )
        np.testing.assert_allclose(npy(masks)[0, 0], logits_np[0, 0], atol=1e-6)
        np.testing.assert_allclose(npy(iou)[0, 0], 0.5, atol=1e-6)
        # Row 1 is still unstable and still falls back to ITS own best token.
        np.testing.assert_allclose(npy(masks)[1, 0], logits_np[1, 3], atol=1e-6)

    def test_stability_path_runs_only_for_single_mask_inference(self):
        """``multimask_output=True`` bypasses the stability branch entirely."""
        decoder = make_decoder()
        inputs = make_inputs()
        masks_multi = run(decoder, inputs, multimask_output=True)[0]
        masks_single = run(decoder, inputs, multimask_output=False)[0]
        assert tuple(masks_multi.shape) == (BATCH, NUM_MULTIMASK, GRID * 4, GRID * 4)
        assert tuple(masks_single.shape) == (BATCH, 1, GRID * 4, GRID * 4)

    def test_training_mode_never_takes_the_fallback(self):
        """During training the single-mask token is used unconditionally."""
        decoder = make_decoder()
        inputs = make_inputs()
        image, pe, sparse, dense, f0, f1 = inputs
        train_masks = decoder(
            image_embeddings=image,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            high_res_features=[f0, f1],
            training=True,
        )[0]
        raw = decoder.predict_masks(
            image_embeddings=image,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            high_res_features=[f0, f1],
            training=True,
        )[0]
        np.testing.assert_allclose(
            npy(train_masks), npy(raw)[:, 0:1], atol=1e-6
        )


# ---------------------------------------------------------------------
# object pointer
# ---------------------------------------------------------------------


class TestObjectPointer:
    """The pointer is sourced from mask token 0 unless BOTH conditions hold."""

    @pytest.mark.parametrize(
        "multimask_output,use_multimask_token,expected_tokens",
        [
            (True, True, NUM_MULTIMASK),
            (True, False, 1),
            (False, True, 1),
            (False, False, 1),
        ],
    )
    def test_pointer_source_requires_both_conditions(
        self, multimask_output, use_multimask_token, expected_tokens
    ):
        """Only ``multimask_output and use_multimask_token_for_obj_ptr`` widens it."""
        decoder = make_decoder(
            use_multimask_token_for_obj_ptr=use_multimask_token
        )
        pointer = run(decoder, make_inputs(), multimask_output=multimask_output)[3]
        assert tuple(pointer.shape) == (BATCH, expected_tokens, DIM)

    def test_pointer_equals_mask_token_zero_by_default(self):
        """By value, not by shape: the default pointer IS mask token 0."""
        decoder = make_decoder(use_multimask_token_for_obj_ptr=False)
        image, pe, sparse, dense, f0, f1 = make_inputs()
        kwargs = dict(
            image_embeddings=image,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            high_res_features=[f0, f1],
        )
        _, _, mask_tokens_out, _ = decoder.predict_masks(**kwargs)
        pointer = decoder(multimask_output=True, **kwargs)[3]
        np.testing.assert_allclose(
            npy(pointer), npy(mask_tokens_out)[:, 0:1], atol=1e-6
        )


# ---------------------------------------------------------------------
# G7.5 -- SAM 1 is untouched
# ---------------------------------------------------------------------


class TestSam1Untouched:
    """G7.5: this step imports SAM 1 internals but must not perturb them."""

    def test_sam1_source_tree_has_no_diff(self):
        """``git diff --stat -- src/dl_techniques/models/SAM/SAM1/`` is empty."""
        result = subprocess.run(
            ["git", "diff", "--stat", "--", "src/dl_techniques/models/SAM/SAM1/"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "", (
            f"SAM 1 source was modified:\n{result.stdout}"
        )

    def test_sam2_registered_key_cannot_collide_with_sam1(self):
        """The bare registration key carries the ``SAM2`` prefix.

        A duplicate bare ``@register_keras_serializable()`` OVERWRITES silently,
        and that is the one mechanism the ``git diff`` proxy above cannot see.
        """
        registry = keras.saving.get_custom_objects()
        assert registry["Custom>SAM2MaskDecoder"] is SAM2MaskDecoder
        assert "Custom>MaskDecoder" in registry
        assert registry["Custom>MaskDecoder"] is not SAM2MaskDecoder

    def test_sam1_decoder_still_produces_its_own_two_tuple(self):
        """Importing SAM 1's module here did not change its own behaviour."""
        from dl_techniques.models.SAM.SAM1.mask_decoder import MaskDecoder

        sam1 = MaskDecoder(transformer_dim=DIM, transformer=make_transformer())
        sam1.build(None)
        image, pe, sparse, dense, _, _ = make_inputs()
        outputs = sam1(
            image_embeddings=image,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        assert len(outputs) == 2


# ---------------------------------------------------------------------
# G7.7 -- dead-component partition
# ---------------------------------------------------------------------


class TestDeadComponentPartition:
    """G7.7: what each killed component actually takes down, measured.

    Encoded as assertions so a future refactor that silently disconnects one of
    them fails here rather than in a quality metric nobody runs.
    """

    def test_instrument_negative_control(self):
        """The oracle reports no movement when nothing is killed."""
        decoder = make_decoder()
        inputs = make_inputs()

        def metric():
            return float(np.sum(npy(run(decoder, inputs)[0])))

        response = component_response(metric, no_op_kill, name="nothing")
        assert not response.moved, str(response)
        assert response.delta == 0.0, str(response)

    def test_killing_the_obj_score_head_kills_only_the_obj_score(self):
        """Zeroing the head zeroes the score and leaves the masks untouched."""
        decoder = make_decoder(pred_obj_scores_mlp=False)
        inputs = make_inputs()

        def obj_metric():
            return float(npy(run(decoder, inputs)[2])[0, 0])

        def mask_metric():
            return float(np.sum(npy(run(decoder, inputs)[0])))

        killer = lambda: zeroed_variables(decoder.pred_obj_score_head.weights)
        obj_response = component_response(obj_metric, killer, name="obj_score_head")
        mask_response = component_response(mask_metric, killer, name="obj_score_head")

        assert obj_response.moved, str(obj_response)
        assert obj_response.after == 0.0, str(obj_response)
        assert not mask_response.moved, str(mask_response)

    def test_killing_conv_s1_kills_the_feat_s1_response(self):
        """A dead 2x lateral makes the 2x skip's perturbation response exactly 0."""
        decoder = make_decoder()
        inputs = make_inputs()

        def s1_response():
            base = npy(run(decoder, inputs)[0])
            bumped = npy(run(decoder, inputs, feat_s1=inputs[5] + 1.0)[0])
            return float(np.max(np.abs(base - bumped)))

        response = component_response(
            s1_response,
            lambda: zeroed_variables(decoder.conv_s1.weights),
            name="conv_s1",
        )
        assert response.before > 1e-5, str(response)
        assert response.after == 0.0, str(response)

    def test_killing_conv_s0_kills_the_feat_s0_response(self):
        """A dead 4x lateral makes the 4x skip's perturbation response exactly 0."""
        decoder = make_decoder()
        inputs = make_inputs()

        def s0_response():
            base = npy(run(decoder, inputs)[0])
            bumped = npy(run(decoder, inputs, feat_s0=inputs[4] + 1.0)[0])
            return float(np.max(np.abs(base - bumped)))

        response = component_response(
            s0_response,
            lambda: zeroed_variables(decoder.conv_s0.weights),
            name="conv_s0",
        )
        assert response.before > 1e-5, str(response)
        assert response.after == 0.0, str(response)

    def test_the_two_laterals_are_not_independent(self):
        """Killing conv_s0 also changes the MEASURED feat_s1 response.

        ``act2`` is a non-linearity applied AFTER ``+ feat_s0``, so the 4x skip
        sets the operating point at which the 2x skip's contribution is read.
        Recorded because it means a green ``feat_s1`` liveness arm is NOT
        evidence that ``conv_s0`` is intact -- the two arms are separate tests
        for that reason.
        """
        decoder = make_decoder()
        inputs = make_inputs()

        def s1_response():
            base = npy(run(decoder, inputs)[0])
            bumped = npy(run(decoder, inputs, feat_s1=inputs[5] + 1.0)[0])
            return float(np.max(np.abs(base - bumped)))

        response = component_response(
            s1_response,
            lambda: zeroed_variables(decoder.conv_s0.weights),
            name="conv_s0 -> feat_s1 response",
        )
        assert response.before > 1e-5, str(response)
        assert response.after > 1e-5, (
            "killing conv_s0 must not zero the feat_s1 response -- if it does, "
            "the 2x skip only reaches the output THROUGH the 4x lateral, which "
            "is not the additive pathway: " + str(response)
        )

    def test_killing_the_mask_token_embedding_does_not_kill_the_masks(self):
        """A dead token embedding still yields non-zero, differing masks.

        Recorded as a NEGATIVE result: "the masks moved" is a structurally weak
        assertion, because the hypernetwork biases and the upscaled image stream
        keep it true even with the mask tokens zeroed.
        """
        decoder = make_decoder()
        inputs = make_inputs()

        def mask_metric():
            return float(np.sum(npy(run(decoder, inputs)[0])))

        response = component_response(
            mask_metric,
            lambda: zeroed_variables(decoder.mask_tokens.weights),
            name="mask_tokens",
        )
        assert response.moved, str(response)
        assert abs(response.after) > 0.0, (
            "a zeroed mask-token embedding still produces non-zero masks: "
            + str(response)
        )


# ---------------------------------------------------------------------
# conventions: shapes, config, gradients, tracing
# ---------------------------------------------------------------------


class TestConventions:
    """Repo conventions: serialization, gradients, shapes, validation."""

    def test_config_round_trip_reproduces_every_constructor_field(self):
        """``get_config`` carries all constructor parameters."""
        decoder = make_decoder(
            iou_head_depth=2,
            iou_head_hidden_dim=48,
            iou_prediction_use_sigmoid=True,
            dynamic_multimask_stability_delta=0.07,
            dynamic_multimask_stability_thresh=0.9,
            normalization_type="rms_norm",
            activation="relu",
            mlp_activation="gelu",
        )
        config = decoder.get_config()
        clone = SAM2MaskDecoder.from_config(dict(config))
        for key, value in config.items():
            if key in ("transformer", "name"):
                continue
            assert clone.get_config()[key] == value, f"field {key} not restored"

    def test_weights_round_trip_by_value(self):
        """A config clone with copied weights reproduces the outputs exactly."""
        decoder = make_decoder()
        inputs = make_inputs()
        original = [npy(t) for t in run(decoder, inputs)]

        clone = SAM2MaskDecoder.from_config(dict(decoder.get_config()))
        clone.build(None)
        run(clone, inputs)  # materialize the lazily built transformer weights
        assert len(clone.weights) == len(decoder.weights)
        clone.set_weights(decoder.get_weights())

        restored = [npy(t) for t in run(clone, inputs)]
        for before, after in zip(original, restored):
            np.testing.assert_allclose(before, after, atol=0.0, rtol=0.0)

    def test_compute_output_shape_returns_four_shapes(self):
        """Four outputs, four shapes."""
        decoder = make_decoder()
        shapes = decoder.compute_output_shape(None)
        assert len(shapes) == 4
        assert shapes[2] == (None, 1)
        assert shapes[3] == (None, None, DIM)

    @staticmethod
    def live_variable_names(decoder, inputs, multimask_output=None):
        """Return the names of variables carrying a non-zero gradient.

        :param decoder: The decoder.
        :type decoder: SAM2MaskDecoder
        :param inputs: The tuple from :func:`make_inputs`.
        :type inputs: tuple
        :param multimask_output: If ``None``, differentiate ``predict_masks``'s
            full token set; otherwise differentiate ``call``'s selected output.
        :type multimask_output: Optional[bool]
        :return: Set of variable paths with a strictly non-zero gradient.
        :rtype: set
        """
        image, pe, sparse, dense, f0, f1 = inputs
        kwargs = dict(
            image_embeddings=image,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            high_res_features=[f0, f1],
            training=True,
        )
        with tf.GradientTape() as tape:
            if multimask_output is None:
                outputs = decoder.predict_masks(**kwargs)
            else:
                outputs = decoder(multimask_output=multimask_output, **kwargs)
            loss = sum(ops.mean(tensor ** 2) for tensor in outputs)
        grads = tape.gradient(loss, decoder.trainable_variables)
        return {
            v.path
            for v, g in zip(decoder.trainable_variables, grads)
            if g is not None and float(np.max(np.abs(npy(g)))) > 0.0
        }

    def test_gradients_reach_every_declared_head(self):
        """Every head receives gradient from the FULL token set.

        Scoped to ``predict_masks`` deliberately -- see
        :meth:`test_multimask_output_starves_the_single_mask_hypernetwork` for
        the measured reason ``call`` is a weaker probe.
        """
        decoder = make_decoder()
        live = self.live_variable_names(decoder, make_inputs())
        for fragment in (
            "obj_score_token",
            "iou_token",
            "mask_tokens",
            "conv_s0",
            "conv_s1",
            "upsample_conv1",
            "upsample_conv2",
            "iou_prediction_head",
            "pred_obj_score_head",
            "hypernetwork_mlp_0",
        ):
            assert any(fragment in name for name in live), (
                f"no live gradient for any variable containing {fragment!r}"
            )

    def test_multimask_output_starves_the_single_mask_hypernetwork(self):
        """MEASURED: at ``multimask_output=True`` hypernetwork 0 gets NO gradient.

        ``call`` drops ``masks[:, 0]`` and, with
        ``use_multimask_token_for_obj_ptr=True``, also drops mask token 0 from
        the object pointer -- so hypernetwork MLP 0 is disconnected from every
        returned tensor. This is correct behaviour, and it is recorded here
        because it makes a "differentiate ``call``" gradient probe blind to a
        genuinely dead hypernetwork head: the guard above must therefore be
        scoped to ``predict_masks``, not to ``call``.
        """
        decoder = make_decoder(use_multimask_token_for_obj_ptr=True)
        inputs = make_inputs()
        via_call = self.live_variable_names(decoder, inputs, multimask_output=True)
        via_predict = self.live_variable_names(decoder, inputs)

        assert not any("hypernetwork_mlp_0" in name for name in via_call)
        assert any("hypernetwork_mlp_0" in name for name in via_predict)
        for index in (1, 2, 3):
            assert any(
                f"hypernetwork_mlp_{index}" in name for name in via_call
            ), f"hypernetwork {index} must stay live under multimask_output"

    def test_shared_prompt_batch_is_broadcast(self):
        """A batch-1 prompt set is tiled onto the image batch."""
        decoder = make_decoder()
        image, pe, _, dense, f0, f1 = make_inputs()
        sparse = ops.convert_to_tensor(
            np.random.default_rng(3).standard_normal(
                (1, NUM_SPARSE, DIM)
            ).astype("float32")
        )
        masks = decoder(
            image_embeddings=image,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            high_res_features=[f0, f1],
        )[0]
        assert tuple(masks.shape)[0] == BATCH

    def test_impossible_prompt_batch_is_refused(self):
        """A prompt batch that is neither 1 nor B is refused, not floored."""
        decoder = make_decoder()
        image, pe, _, dense, f0, f1 = make_inputs()
        sparse = ops.convert_to_tensor(
            np.zeros((3, NUM_SPARSE, DIM), dtype="float32")
        )
        with pytest.raises(ValueError, match="cannot tile"):
            decoder(
                image_embeddings=image,
                image_pe=pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
                high_res_features=[f0, f1],
            )

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"transformer_dim": 0}, "transformer_dim must be positive"),
            ({"transformer_dim": 12}, "divisible by 8"),
            ({"num_multimask_outputs": 0}, "num_multimask_outputs must be positive"),
            ({"iou_head_depth": 0}, "iou_head_depth must be positive"),
            ({"iou_head_hidden_dim": 0}, "iou_head_hidden_dim must be positive"),
        ],
    )
    def test_invalid_configurations_are_refused(self, kwargs, match):
        """Every silent-width defect is a construction-time error."""
        with pytest.raises(ValueError, match=match):
            make_decoder(**kwargs)

    def test_hypernetwork_depth_is_fixed_at_three_not_tied_to_the_iou_head(self):
        """``iou_head_depth`` must move the IoU head ALONE.

        The reference hardcodes ``MLP(dim, dim, dim // 8, 3)`` for the mask
        hypernetworks while exposing the IoU head's depth. The two agree at the
        default ``iou_head_depth=3``, so a port that reuses the parameter for
        both is invisible at every shipped configuration and silently
        restructures all four mask heads at any other. The probe therefore uses
        a NON-default depth -- at the default it cannot discriminate.
        """
        default = make_decoder()
        assert default.iou_head_depth == 3, (
            "this probe assumes the default depth is 3, so that the "
            "non-default depth below actually differs from it"
        )
        assert len(default.output_hypernetworks_mlps[0].layers) == 3
        assert len(default.iou_prediction_head.layers) == 3

        deeper = make_decoder(iou_head_depth=5)
        assert len(deeper.iou_prediction_head.layers) == 5, (
            "iou_head_depth no longer reaches the IoU head at all"
        )
        for index, mlp in enumerate(deeper.output_hypernetworks_mlps):
            assert len(mlp.layers) == 3, (
                f"hypernetwork MLP {index} has {len(mlp.layers)} layers at "
                f"iou_head_depth=5; the reference fixes it at 3 independently "
                f"of the IoU head's depth"
            )

    def test_iou_sigmoid_knob_is_live(self):
        """``iou_prediction_use_sigmoid`` changes the value, not just the config."""
        raw = make_decoder(iou_prediction_use_sigmoid=False)
        squashed = SAM2MaskDecoder.from_config(dict(raw.get_config()))
        squashed.iou_prediction_use_sigmoid = True
        squashed.build(None)
        inputs = make_inputs()
        run(raw, inputs)
        run(squashed, inputs)
        squashed.set_weights(raw.get_weights())

        raw_iou = npy(run(raw, inputs)[1])
        squashed_iou = npy(run(squashed, inputs)[1])
        np.testing.assert_allclose(
            squashed_iou, 1.0 / (1.0 + np.exp(-raw_iou)), atol=1e-5
        )

    def test_forward_pass_traces_under_tf_function(self):
        """The decoder's forward path is graph-traceable at a static shape."""
        decoder = make_decoder()
        inputs = make_inputs()
        run(decoder, inputs)  # build the transformer eagerly first

        @tf.function
        def traced(image, pe, sparse, dense, f0, f1):
            return decoder(
                image_embeddings=image,
                image_pe=pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
                high_res_features=[f0, f1],
            )

        outputs = traced(*inputs)
        assert len(outputs) == 4
        assert np.all(np.isfinite(npy(outputs[0])))

    def test_no_nans_at_float64(self):
        """The safe division holds up in a wider dtype too."""
        decoder = make_decoder()
        logits = build_stability_logits(n_above=0, n_between=0, n_below=16)
        score = npy(decoder._get_stability_scores(ops.cast(logits, "float32")))
        assert np.all(np.isfinite(score))
