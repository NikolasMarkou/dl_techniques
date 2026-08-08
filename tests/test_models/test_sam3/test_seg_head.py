"""Tests for `models/SAM/SAM3/maskformer_segmentation.py` -- `Sam3SegmentationHead`.

The guards here are organized around the four ways this head can be silently
wrong: the upsampling MODE, the fusion OPERATOR, the merge ORDER, and the
presence mechanism that must not exist at all.
"""

import inspect
import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.SAM.SAM3.maskformer_segmentation import (
    Sam3SegmentationHead,
)

# ---------------------------------------------------------------------
# fixtures / constants
# ---------------------------------------------------------------------

TINY = dict(d_model=8, upsampling_stages=3, num_heads=2, num_groups=2)
BATCH, QUERIES, TOKENS, LAYERS = 2, 5, 6, 3
GRIDS = (16, 8, 4, 2)          # finest -> coarsest, the neck's own order
SHIPPED = dict(d_model=256, upsampling_stages=3, num_heads=8, num_groups=8)

# MEASURED in BOTH regimes, not derived from first principles (D-105). The
# IDENTICAL float64 comparison of this head's forward pass measures
# `1.25e-06` on CPU and `2.22e-03` on GPU 1 under TF32 -- a 1,775x swing that
# has nothing to do with correctness. The tolerance is therefore set from the
# slower-precision regime, and the wrong-candidate margin is pinned in-suite
# (`test_the_bilinear_candidate_fails_the_nearest_oracle`, measured 1.23, i.e.
# 246x this tolerance) so the looser number cannot swallow a real defect.
ORACLE_ATOL = 5e-3


def _np(tensor):
    return np.asarray(tensor)


def _feats(seed=0, grids=GRIDS, width=8):
    rng = np.random.RandomState(seed)
    return [rng.randn(BATCH, g, g, width).astype("float32") for g in grids]


def _payload(seed=0):
    rng = np.random.RandomState(seed + 100)
    coarse = GRIDS[-1] * GRIDS[-1]
    return dict(
        backbone_feats=_feats(seed),
        obj_queries=rng.randn(LAYERS, BATCH, QUERIES, 8).astype("float32"),
        # deliberately LONGER than the coarse grid: the reference's fused
        # sequence carries non-spatial tokens after the image tokens.
        encoder_hidden_states=rng.randn(BATCH, coarse + 7, 8).astype("float32"),
        prompt=rng.randn(BATCH, TOKENS, 8).astype("float32"),
    )


def _build(head, payload):
    head.build(
        [f.shape for f in payload["backbone_feats"]],
        payload["obj_queries"].shape,
        payload["encoder_hidden_states"].shape,
        payload["prompt"].shape,
    )
    return head


@pytest.fixture
def head():
    payload = _payload()
    return _build(Sam3SegmentationHead(**TINY), payload)


# ---------------------------------------------------------------------
# the float64 reference forward
# ---------------------------------------------------------------------


def _conv(x, kernel, bias, pad):
    kh, kw, _, co = kernel.shape
    b, h, w, _ = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    out = np.zeros((b, h, w, co), dtype="float64")
    for i in range(kh):
        for j in range(kw):
            out += np.einsum("bhwc,co->bhwo", xp[:, i:i + h, j:j + w, :],
                             kernel[i, j])
    return out + bias


def _group_norm(x, gamma, beta, groups, eps):
    b, h, w, c = x.shape
    xr = x.reshape(b, h, w, groups, c // groups)
    mean = xr.mean(axis=(1, 2, 4), keepdims=True)
    var = xr.var(axis=(1, 2, 4), keepdims=True)
    normed = ((xr - mean) / np.sqrt(var + eps)).reshape(b, h, w, c)
    return normed * gamma + beta


def _resize(x, size, mode):
    """Reference upsample. `nearest` is the reference mode; `bilinear` is the
    wrong candidate M8.1 names, kept here so the oracle can separate them."""
    h, w = size
    if mode == "nearest":
        rows = np.floor(np.arange(h) * x.shape[1] / h).astype(int)
        cols = np.floor(np.arange(w) * x.shape[2] / w).astype(int)
        return x[:, rows][:, :, cols]
    src_r = (np.arange(h) + 0.5) * x.shape[1] / h - 0.5
    src_c = (np.arange(w) + 0.5) * x.shape[2] / w - 0.5
    src_r = np.clip(src_r, 0, x.shape[1] - 1)
    src_c = np.clip(src_c, 0, x.shape[2] - 1)
    r0, c0 = np.floor(src_r).astype(int), np.floor(src_c).astype(int)
    r1 = np.minimum(r0 + 1, x.shape[1] - 1)
    c1 = np.minimum(c0 + 1, x.shape[2] - 1)
    wr = (src_r - r0)[None, :, None, None]
    wc = (src_c - c0)[None, None, :, None]
    top = x[:, r0][:, :, c0] * (1 - wc) + x[:, r0][:, :, c1] * wc
    bot = x[:, r1][:, :, c0] * (1 - wc) + x[:, r1][:, :, c1] * wc
    return top * (1 - wr) + bot * wr


def _reference_forward(head, payload, mode=None, start_at_finest=False):
    """A float64 reimplementation of the whole head, cross-attend disabled."""
    mode = mode or head.interpolation_mode
    feats = [f.astype("float64") for f in payload["backbone_feats"]]
    coarse = feats[-1]
    spatial = coarse.shape[1] * coarse.shape[2]
    flat = payload["encoder_hidden_states"].astype("float64")[:, :spatial, :]
    feats[-1] = flat.reshape(coarse.shape)

    if start_at_finest:
        running, laterals = feats[0], list(feats[1:])
    else:
        running, laterals = feats[-1], list(reversed(feats[:-1]))
    for index, lateral in enumerate(laterals):
        upsampled = _resize(running, lateral.shape[1:3], mode)
        running = lateral + upsampled
        running = _conv(running, _np(head.pixel_convs[index].kernel),
                        _np(head.pixel_convs[index].bias), pad=1)
        running = _group_norm(running, _np(head.pixel_norms[index].gamma),
                              _np(head.pixel_norms[index].beta),
                              head.num_groups, head.norm_epsilon)
        running = np.maximum(running, 0.0)

    pixel_embed = _conv(running, _np(head.instance_seg_head.kernel),
                        _np(head.instance_seg_head.bias), pad=0)
    semantic = _conv(running, _np(head.semantic_seg_head.kernel),
                     _np(head.semantic_seg_head.bias), pad=0)
    queries = payload["obj_queries"].astype("float64")[-1]
    for position, dense in enumerate(head.mask_embed):
        queries = queries @ _np(dense.kernel) + _np(dense.bias)
        if position < len(head.mask_embed) - 1:
            queries = np.maximum(queries, 0.0)
    masks = np.einsum("bqc,bhwc->bqhw", queries, pixel_embed)
    return {"pred_masks": masks, "semantic_seg": semantic}


@pytest.fixture
def plain():
    """A head with the prompt cross-attend switched OFF, for the oracle."""
    payload = _payload()
    config = dict(TINY, use_cross_attend_prompt=False)
    head = Sam3SegmentationHead(**config)
    head.build([f.shape for f in payload["backbone_feats"]],
               payload["obj_queries"].shape,
               payload["encoder_hidden_states"].shape)
    return head


# =====================================================================


class TestReferenceForward:

    def test_matches_the_float64_reference_forward(self, plain):
        """VACUITY NOTE: the oracle takes its upsampling mode FROM the
        implementation, so a mode swap agrees on both sides and this test
        alone cannot see M8.1 (MEASURED: it stayed green under it). The arm
        that does see it is
        `test_the_bilinear_candidate_fails_the_nearest_oracle`, which pins the
        mode literally.
        """
        payload = _payload()
        got = plain(**payload, training=False)
        want = _reference_forward(plain, payload)
        for key in want:
            np.testing.assert_allclose(_np(got[key]), want[key],
                                       atol=ORACLE_ATOL)

    def test_the_probe_is_not_constant_so_the_upsample_mode_is_visible(self):
        """Non-vacuity: at a CONSTANT feature map nearest and bilinear agree
        exactly, so a constant probe cannot see M8.1. This asserts the probe
        this suite actually uses is not that probe."""
        feats = _feats()
        coarse = feats[-1]
        spread = float(np.abs(coarse - coarse.mean()).max())
        assert spread > 0.1, "the coarse probe is (near) constant"

    def test_nearest_and_bilinear_coincide_at_a_constant_input(self):
        """The named COINCIDENCE POINT, executed rather than asserted in prose.

        Measured at ONE stage: with more stages the ``same``-padded 3x3
        convolution makes an initially constant map non-constant at its
        border, so the two modes separate again from stage two onward. The
        coincidence is a property of the RESIZE, and this is the configuration
        that isolates it.
        """
        config = dict(TINY, upsampling_stages=1, use_cross_attend_prompt=False,
                      interpolation_mode="nearest")
        payload = _payload()
        payload["backbone_feats"] = [
            np.full((BATCH, 4, 4, 8), 0.75, "float32"),
            np.full((BATCH, 2, 2, 8), 0.75, "float32"),
        ]
        payload["encoder_hidden_states"] = np.full(
            (BATCH, 4 + 7, 8), 0.75, "float32")
        heads = []
        for mode in ("nearest", "bilinear"):
            built = Sam3SegmentationHead(**dict(config,
                                                interpolation_mode=mode))
            built.build([f.shape for f in payload["backbone_feats"]],
                        payload["obj_queries"].shape,
                        payload["encoder_hidden_states"].shape)
            heads.append(built)
        heads[1].set_weights(heads[0].get_weights())
        delta = np.abs(_np(heads[0](**payload)["pred_masks"])
                       - _np(heads[1](**payload)["pred_masks"])).max()
        assert delta == 0.0, (
            f"expected the two modes to coincide on a constant map, got "
            f"{delta}")

    def test_bilinear_is_measurably_different_at_an_interior_pixel(self, plain):
        """M8.1's discriminating probe: a NON-constant coarse map.

        SEED-PINNED, and it builds its own head rather than taking the
        `plain` fixture. That is not garnish: this is a MAGNITUDE threshold on
        randomly initialized weights, and those initializers draw from the
        KERAS GLOBAL RNG -- whose state depends on every layer built by every
        test that ran earlier in the same process. MEASURED: adding tests to
        `test_query_selection.py` (which sorts before this file) moved this
        delta to 0.818 and turned the whole directory gate RED without
        touching anything this test covers, while the file ALONE stayed green.

        The pin is local and immediately precedes construction, so the probe no
        longer depends on what ran before it. The SHIPPED initializers are kept
        -- only their stream is fixed. The seed is not a lucky one: the same
        measurement at seeds 0/1/2/3/7 reads 1.06 / 1.71 / 1.38 / 2.10 / 3.35,
        i.e. every one clears the 1.0 bar, and 3 is quoted here with a 2.1x
        margin. What that spread also says is that this bar sits at the low
        edge of the distribution; widening the margin is a separate question
        from removing the ambient dependence, and is deliberately not done
        here.
        """
        del plain
        keras.utils.set_random_seed(3)
        payload = _payload()
        built = []
        for mode in ("nearest", "bilinear"):
            head = Sam3SegmentationHead(
                **dict(TINY, use_cross_attend_prompt=False,
                       interpolation_mode=mode))
            head.build([f.shape for f in payload["backbone_feats"]],
                       payload["obj_queries"].shape,
                       payload["encoder_hidden_states"].shape)
            built.append(head)
        built[1].set_weights(built[0].get_weights())
        delta = np.abs(_np(built[0](**payload)["pred_masks"])
                       - _np(built[1](**payload)["pred_masks"])).max()
        assert delta > 1.0, f"nearest vs bilinear separated by only {delta}"

    def test_the_bilinear_candidate_fails_the_nearest_oracle(self, plain):
        """The wrong-candidate margin, pinned so ORACLE_ATOL cannot hide it."""
        payload = _payload()
        wrong = _reference_forward(plain, payload, mode="bilinear")
        got = _np(plain(**payload)["pred_masks"])
        margin = np.abs(got - wrong["pred_masks"]).max()
        assert margin > 100 * ORACLE_ATOL, f"margin only {margin}"

    def test_the_finest_first_candidate_fails_the_oracle_by_shape(self, plain):
        payload = _payload()
        wrong = _reference_forward(plain, payload, start_at_finest=True)
        got = _np(plain(**payload)["pred_masks"])
        assert got.shape != wrong["pred_masks"].shape


class TestTopDownOrder:

    def test_the_merge_starts_at_the_coarsest_feature_and_ends_at_the_finest(
            self, head):
        """M8.3's guard. Starting from the finest emits masks at 2x2*... the
        WRONG resolution, with no exception anywhere."""
        out = head(**_payload())
        assert out["pred_masks"].shape[-2:] == (GRIDS[0], GRIDS[0])
        assert out["semantic_seg"].shape[1:3] == (GRIDS[0], GRIDS[0])

    def test_every_stage_conv_is_built_on_its_own_lateral_grid(self, head):
        """Stage k consumes the k-th COARSEST remaining level, not the k-th."""
        expected = list(reversed(GRIDS[:-1]))
        for index, grid in enumerate(expected):
            built = head.pixel_convs[index]._build_shapes_dict
            shape = list(built.values())[0]
            assert tuple(shape)[1:3] == (grid, grid), (
                f"stage {index} built on {shape}, expected grid {grid}")

    def test_the_stage_count_must_match_the_pyramid(self):
        payload = _payload()
        head = Sam3SegmentationHead(**dict(TINY, upsampling_stages=2))
        with pytest.raises(ValueError, match="upsampling_stages"):
            _build(head, payload)


class TestSkipFusion:

    def test_the_skip_fusion_is_additive_and_preserves_width(self, head):
        """M8.2's guard is the WIDTH, not the output movement: iteration 1
        MEASURED that a coherent concat port left 35 of 37 tests green and only
        the width assertions fired."""
        for conv in head.pixel_convs:
            shape = list(conv._build_shapes_dict.values())[0]
            assert tuple(shape)[-1] == TINY["d_model"]
        assert list(head.instance_seg_head._build_shapes_dict.values())[0][-1] \
               == TINY["d_model"]

    def test_a_width_widening_fusion_is_refused(self, head):
        """The width check RED-proven: a concatenating fusion reaches it."""
        fine = ops.convert_to_tensor(np.zeros((BATCH, 4, 4, 16), "float32"))
        coarse = ops.convert_to_tensor(np.zeros((BATCH, 2, 2, 16), "float32"))
        with pytest.raises(ValueError, match="concatenating fusion"):
            head._merge(coarse, fine)

    def test_the_declared_output_widths_come_from_config(self, head):
        payload = _payload()
        declared = head.compute_output_shape(
            [f.shape for f in payload["backbone_feats"]],
            payload["obj_queries"].shape)
        got = head(**payload)
        for key in declared:
            assert tuple(declared[key]) == tuple(got[key].shape)


class TestPromptCrossAttend:

    def test_a_different_prompt_changes_the_masks(self, head):
        payload = _payload()
        first = _np(head(**payload)["pred_masks"])
        payload["prompt"] = payload["prompt"] + 3.0
        second = _np(head(**payload)["pred_masks"])
        assert np.abs(first - second).max() > 1e-4

    def test_the_prompt_padding_mask_changes_the_masks(self, head):
        payload = _payload()
        first = _np(head(**payload)["pred_masks"])
        mask = np.zeros((BATCH, TOKENS), dtype=bool)
        mask[:, TOKENS // 2:] = True
        second = _np(head(**payload, prompt_padding_mask=mask)["pred_masks"])
        assert np.abs(first - second).max() > 1e-4

    def test_gradients_reach_the_cross_attend_weights(self, head):
        import tensorflow as tf
        payload = {k: ops.convert_to_tensor(v) for k, v in _payload().items()
                   if k != "backbone_feats"}
        feats = [ops.convert_to_tensor(f) for f in _feats()]
        variables = head.cross_attend_prompt.trainable_weights
        assert variables, "the cross-attend carries no trainable weights"
        with tf.GradientTape() as tape:
            loss = ops.sum(head(feats, **payload)["pred_masks"] ** 2)
        grads = tape.gradient(loss, variables)
        moved = [float(ops.max(ops.abs(g))) for g in grads if g is not None]
        assert len(moved) == len(variables)
        assert max(moved) > 1e-8

    def test_disabling_the_cross_attend_changes_the_output(self, head, plain):
        """The shared sub-layers are transplanted one by one.

        Comparing two INDEPENDENTLY initialized heads would make this test
        vacuous -- it would pass with the cross-attend deleted, because the
        two heads' pixel decoders already disagree. MEASURED: the first draft
        of this test did exactly that and survived the dead-component probe.
        """
        for mine, theirs in zip(
                head.pixel_convs + head.pixel_norms + head.mask_embed
                + [head.semantic_seg_head, head.instance_seg_head],
                plain.pixel_convs + plain.pixel_norms + plain.mask_embed
                + [plain.semantic_seg_head, plain.instance_seg_head]):
            theirs.set_weights(mine.get_weights())
        payload = _payload()
        with_prompt = _np(head(**payload)["pred_masks"])
        without = _np(plain(**payload)["pred_masks"])
        assert np.abs(with_prompt - without).max() > 1e-4

    def test_the_cross_attend_is_a_pre_norm_residual_not_a_replacement(
            self, head):
        """The normalized tensor feeds the attention; the UN-normalized one
        carries the skip. A replacement would discard the encoder states, so
        scaling them would leave the masks untouched.

        VACUITY NOTE: this arm guards the RESIDUAL, not the attention, so it
        survives a zeroed cross-attend by construction (MEASURED). It is not a
        liveness arm for the cross-attend -- the four in this class that are
        all died under that probe.
        """
        payload = _payload()
        first = _np(head(**payload)["pred_masks"])
        payload["encoder_hidden_states"] = \
            payload["encoder_hidden_states"] * 4.0
        second = _np(head(**payload)["pred_masks"])
        assert np.abs(first - second).max() > 1e-4

    def test_a_missing_prompt_is_refused(self, head):
        payload = _payload()
        payload.pop("prompt")
        with pytest.raises(ValueError, match="prompt is required"):
            head(**payload)


class TestNoPresenceMechanism:

    def test_the_head_exposes_no_presence_mechanism_anywhere(self, head):
        """M8.4's guard. I-7: only the DECODER's presence signal is live.

        This is an ABSENCE assertion, so a dead component satisfies it by
        construction -- it is paired with, not a substitute for, the liveness
        arms in `TestPromptCrossAttend`.
        """
        names = set(inspect.signature(
            Sam3SegmentationHead.__init__).parameters)
        assert not [n for n in names if "presence" in n.lower()]
        assert not [n for n in dir(head) if "presence" in n.lower()]
        assert not [k for k in head.get_config() if "presence" in k.lower()]
        assert set(head(**_payload())) == {"pred_masks", "semantic_seg"}


class TestEncoderStateFolding:

    def test_the_leading_spatial_tokens_replace_the_coarsest_level(self, plain):
        payload = _payload()
        first = _np(plain(**payload)["pred_masks"])
        payload["backbone_feats"] = list(payload["backbone_feats"])
        payload["backbone_feats"][-1] = \
            payload["backbone_feats"][-1] + 10.0
        second = _np(plain(**payload)["pred_masks"])
        assert np.abs(first - second).max() == 0.0

    def test_the_encoder_states_do_reach_the_masks(self, plain):
        payload = _payload()
        first = _np(plain(**payload)["pred_masks"])
        payload["encoder_hidden_states"] = \
            payload["encoder_hidden_states"] + 1.0
        second = _np(plain(**payload)["pred_masks"])
        assert np.abs(first - second).max() > 1e-4

    def test_trailing_non_spatial_tokens_are_ignored(self, plain):
        payload = _payload()
        spatial = GRIDS[-1] * GRIDS[-1]
        first = _np(plain(**payload)["pred_masks"])
        payload["encoder_hidden_states"] = \
            payload["encoder_hidden_states"].copy()
        payload["encoder_hidden_states"][:, spatial:, :] += 25.0
        second = _np(plain(**payload)["pred_masks"])
        assert np.abs(first - second).max() == 0.0


class TestMaskBranchAgainstTheReuseCandidate:
    """D-117: `EomtMask` was the reuse candidate and is REJECTED by contract.

    The rejection is measured, not argued: the two mask branches agree by
    VALUE once weights are transplanted, and the only difference is the class
    head `EomtMask` cannot be built without.
    """

    def test_the_reuse_candidate_carries_a_class_head_that_cannot_be_disabled(
            self):
        from dl_techniques.layers.eomt_mask import EomtMask
        names = set(inspect.signature(EomtMask.__init__).parameters)
        assert "num_classes" in names
        with pytest.raises(ValueError):
            EomtMask(num_classes=0, mask_dim=8)
        candidate = EomtMask(num_classes=1, mask_dim=8, hidden_dims=[8, 8])
        candidate.build(((BATCH, QUERIES, 8), (BATCH, 4, 4, 8)))
        assert candidate.class_head.count_params() == 9

    def test_the_mask_branch_matches_the_reuse_candidate_by_value(self, plain):
        from dl_techniques.layers.eomt_mask import EomtMask
        candidate = EomtMask(num_classes=1, mask_dim=8, hidden_dims=[8, 8])
        candidate.build(((BATCH, QUERIES, 8), (BATCH, 4, 4, 8)))
        assert (candidate.mask_mlp.count_params()
                + candidate.mask_projection.count_params()
                == sum(d.count_params() for d in plain.mask_embed))
        for index, dense in enumerate(plain.mask_embed[:2]):
            candidate.mask_mlp.layers[index].set_weights(dense.get_weights())
        candidate.mask_projection.set_weights(plain.mask_embed[2].get_weights())
        queries = _payload()["obj_queries"][-1]
        pixels = _feats()[2]
        _, masks = candidate((ops.convert_to_tensor(queries),
                              ops.convert_to_tensor(pixels)))
        mine = queries
        for position, dense in enumerate(plain.mask_embed):
            mine = _np(dense(ops.convert_to_tensor(mine)))
        mine = np.einsum("bqc,bhwc->bqhw", mine, pixels)
        np.testing.assert_allclose(_np(masks), mine, atol=1e-5)


class TestConstruction:

    @pytest.mark.parametrize("bad", [
        dict(d_model=0), dict(num_heads=0), dict(num_groups=0),
        dict(upsampling_stages=0), dict(d_model=9, num_heads=2),
        dict(d_model=9, num_groups=2), dict(interpolation_mode="bicubic"),
        dict(attention_dropout_rate=1.0),
    ])
    def test_invalid_configuration_is_refused(self, bad):
        with pytest.raises(ValueError):
            Sam3SegmentationHead(**dict(TINY, **bad))

    def test_a_wrong_pyramid_width_is_refused_at_build(self):
        payload = _payload()
        payload["backbone_feats"] = _feats(width=4)
        with pytest.raises(ValueError, match="must already equal d_model"):
            _build(Sam3SegmentationHead(**TINY), payload)

    def test_a_wrong_pyramid_rank_is_refused_at_build(self):
        head = Sam3SegmentationHead(**TINY)
        with pytest.raises(ValueError, match="rank-4 channels-last"):
            head.build([(BATCH, 16, 8)] * 4, (BATCH, QUERIES, 8),
                       (BATCH, 4, 8), (BATCH, TOKENS, 8))

    def test_a_wrong_query_width_is_refused_at_build(self):
        head = Sam3SegmentationHead(**TINY)
        with pytest.raises(ValueError, match="obj_queries width"):
            head.build([(BATCH, g, g, 8) for g in GRIDS],
                       (BATCH, QUERIES, 16), (BATCH, 4, 8), (BATCH, TOKENS, 8))

    def test_a_missing_prompt_shape_is_refused_at_build(self):
        head = Sam3SegmentationHead(**TINY)
        with pytest.raises(ValueError, match="prompt_shape is required"):
            head.build([(BATCH, g, g, 8) for g in GRIDS],
                       (BATCH, QUERIES, 8), (BATCH, 4, 8))

    def test_the_group_normalization_epsilon_is_the_reference_value(self, head):
        """D-118: Keras `GroupNormalization` DEFAULTS to 1e-3 where the
        reference's `nn.GroupNorm` defaults to 1e-5 -- a silent 100x."""
        for norm in head.pixel_norms:
            assert norm.epsilon == 1e-5
            assert norm.epsilon != keras.layers.GroupNormalization(
                groups=2).epsilon

    def test_the_tiny_parameter_count_matches_the_closed_form(self, head):
        d, g, s = TINY["d_model"], TINY["num_groups"], TINY["upsampling_stages"]
        expected = (
            2 * d                                 # cross-attention layer norm
            + 4 * (d * d + d)                     # q / k / v / out projections
            + s * (9 * d * d + d)                 # 3x3 stage convolutions
            + s * 2 * d                           # stage group norms
            + (d + 1)                             # semantic 1x1
            + (d * d + d)                         # instance 1x1
            + 3 * (d * d + d)                     # 3-layer mask embedding
        )
        assert head.count_params() == expected

    def test_the_shipped_parameter_count_matches_the_closed_form(self):
        head = Sam3SegmentationHead(**SHIPPED)
        head.build([(None, g, g, 256) for g in (128, 64, 32, 16)],
                   (None, 200, 256), (None, 256, 256), (None, 32, 256))
        d, s = 256, 3
        expected = (2 * d + 4 * (d * d + d) + s * (9 * d * d + d)
                    + s * 2 * d + (d + 1) + (d * d + d) + 3 * (d * d + d))
        assert head.count_params() == expected == 2_298_881

    def test_the_parameter_audit_is_not_vacuous(self):
        """Executed, not asserted: a head missing one stage must FAIL the
        closed form the test above passes."""
        head = Sam3SegmentationHead(**dict(SHIPPED, upsampling_stages=2))
        head.build([(None, g, g, 256) for g in (64, 32, 16)],
                   (None, 200, 256), (None, 256, 256), (None, 32, 256))
        assert head.count_params() != 2_298_881


class TestSerialization:

    def test_config_keys_equal_init_signature(self, head):
        expected = {name for name in inspect.signature(
            Sam3SegmentationHead.__init__).parameters
            if name not in ("self", "kwargs")}
        missing = expected - set(head.get_config())
        assert missing == set(), f"get_config() is missing {sorted(missing)}"

    def test_round_trip_by_value(self, head):
        payload = _payload()
        before = head(**payload, training=False)
        clone = _build(Sam3SegmentationHead.from_config(head.get_config()),
                       payload)
        clone.set_weights(head.get_weights())
        after = clone(**payload, training=False)
        for key in before:
            np.testing.assert_allclose(_np(before[key]), _np(after[key]),
                                       atol=1e-6)

    def test_keras_model_round_trip_by_value(self, head, tmp_path):
        """D-098: checked by VALUE. A nested sub-layer store restores FRESHLY
        INITIALIZED kernels while every count and path still matches; this
        layer stores its convolutions, its group norms and its mask embedding
        FLAT for exactly that reason."""
        payload = _payload()
        # Rank-3 queries here: the functional API reads a leading axis as the
        # batch, so the decoder's per-layer stack cannot be a model input.
        payload["obj_queries"] = payload["obj_queries"][-1]
        feat_inputs = [keras.Input(shape=f.shape[1:])
                       for f in payload["backbone_feats"]]
        query_in = keras.Input(shape=payload["obj_queries"].shape[1:])
        enc_in = keras.Input(shape=payload["encoder_hidden_states"].shape[1:])
        prompt_in = keras.Input(shape=payload["prompt"].shape[1:])
        out = head(feat_inputs, query_in, enc_in, prompt=prompt_in)
        model = keras.Model(
            feat_inputs + [query_in, enc_in, prompt_in],
            [out["pred_masks"], out["semantic_seg"]])
        probe = list(payload["backbone_feats"]) + [
            payload["obj_queries"], payload["encoder_hidden_states"],
            payload["prompt"]]
        before = [np.asarray(t) for t in model.predict(probe, verbose=0)]
        path = str(tmp_path / "seg_head.keras")
        model.save(path)
        restored = keras.models.load_model(path)
        after = [np.asarray(t) for t in restored.predict(probe, verbose=0)]
        for lhs, rhs in zip(before, after):
            assert np.max(np.abs(lhs - rhs)) == 0.0

    def test_the_round_trip_guard_can_see_a_difference(self, head):
        """The comparator is RED-proven before the exact-zero PASS is trusted."""
        payload = _payload()
        before = _np(head(**payload)["pred_masks"])
        head.pixel_norms[0].gamma.assign(head.pixel_norms[0].gamma + 0.5)
        after = _np(head(**payload)["pred_masks"])
        assert np.max(np.abs(before - after)) > 1e-3
