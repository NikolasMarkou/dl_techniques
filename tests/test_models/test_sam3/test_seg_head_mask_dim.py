"""F-16: ``Sam3SegmentationHead(mask_dim != d_model)`` must be callable.

``mask_dim`` is a documented constructor argument -- the class docstring at
``maskformer_segmentation.py:109-111`` says it is "Width of the mask embedding
and therefore of the pixel embedding the einsum contracts against". The mask
branch honoured it (``mask_embed_2`` is ``Dense(mask_dim)``); the pixel branch
did not (``instance_seg_head`` was ``Conv2D(d_model)``). ``call``'s
``ops.einsum("bqc,bhwc->bqhw", queries, pixel_embed)`` contracts those two axes
against each other, so any ``mask_dim != d_model`` constructed, VALIDATED and
BUILT fine -- the two layers are built against different shapes and never
compared -- and then raised ``InvalidArgumentError`` on the first forward.

It was latent at every shipped configuration because ``mask_dim=None`` resolves
to ``d_model`` and ``Sam3Image.from_variant`` never passes it. Nothing in
``test_seg_head.py`` touches ``mask_dim`` on this class, so nothing could catch
it. These are RED at 11f971ed1 except the two controls, which are labelled.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.SAM.SAM3.maskformer_segmentation import (
    Sam3SegmentationHead,
)

BATCH, QUERIES = 2, 5
D_MODEL = 16
GRIDS = (8, 4)  # finest -> coarsest


def _payload(width=D_MODEL, seed=0):
    rng = np.random.RandomState(seed)
    coarse = GRIDS[-1] * GRIDS[-1]
    return dict(
        backbone_feats=[
            rng.randn(BATCH, g, g, width).astype("float32") for g in GRIDS
        ],
        obj_queries=rng.randn(BATCH, QUERIES, width).astype("float32"),
        encoder_hidden_states=rng.randn(BATCH, coarse + 3, width).astype("float32"),
    )


def _head(**overrides):
    cfg = dict(d_model=D_MODEL, upsampling_stages=1, num_heads=2, num_groups=2,
               use_cross_attend_prompt=False)
    cfg.update(overrides)
    return Sam3SegmentationHead(**cfg)


def _call(head, payload):
    return head(payload["backbone_feats"], payload["obj_queries"],
                payload["encoder_hidden_states"])


class TestPixelEmbeddingFollowsMaskDim:
    """The einsum's shared ``c`` index is ``mask_dim`` on BOTH sides."""

    @pytest.mark.parametrize("mask_dim", [D_MODEL // 2, D_MODEL * 2])
    def test_the_two_einsum_operands_are_sized_the_same(self, mask_dim):
        head = _head(mask_dim=mask_dim)
        assert head.instance_seg_head.filters == mask_dim
        assert head.mask_embed[-1].units == mask_dim
        assert head.instance_seg_head.filters == head.mask_embed[-1].units, (
            "the pixel-embedding conv and the mask-embedding MLP feed the two "
            "sides of ops.einsum('bqc,bhwc->bqhw'), so their widths must agree"
        )

    @pytest.mark.parametrize("mask_dim", [D_MODEL // 2, D_MODEL * 2])
    def test_a_forward_pass_runs_and_has_the_documented_shape(self, mask_dim):
        """THE defect: this raised InvalidArgumentError before the fix.

        Pre-fix at ``d_model=32, mask_dim=16`` the message was "Expected
        dimension 16 at axis 3 of the input shaped [1,8,8,32]".
        """
        head = _head(mask_dim=mask_dim)
        payload = _payload()
        out = _call(head, payload)
        # `pred_masks` is at the FINEST level's resolution and does NOT carry
        # `mask_dim` -- the einsum contracts it away. That is the point: the
        # output shape is mask_dim-INDEPENDENT, which is exactly why a shape
        # assertion on the output could never have found this.
        assert tuple(out["pred_masks"].shape) == (BATCH, QUERIES, GRIDS[0], GRIDS[0])
        assert tuple(out["semantic_seg"].shape) == (BATCH, GRIDS[0], GRIDS[0], 1)
        assert np.all(np.isfinite(np.asarray(out["pred_masks"])))

    def test_mask_dim_none_still_resolves_to_d_model(self):
        """CONTROL (green both ways): the shipped path is untouched."""
        head = _head(mask_dim=None)
        assert head.mask_dim == D_MODEL
        assert head.instance_seg_head.filters == D_MODEL
        out = _call(head, _payload())
        assert tuple(out["pred_masks"].shape) == (BATCH, QUERIES, GRIDS[0], GRIDS[0])

    def test_mask_dim_equal_to_d_model_is_the_same_graph_as_none(self):
        """CONTROL: passing the resolved value explicitly changes nothing."""
        a, b = _head(mask_dim=None), _head(mask_dim=D_MODEL)
        assert a.instance_seg_head.filters == b.instance_seg_head.filters
        assert a.get_config()["mask_dim"] == b.get_config()["mask_dim"] == D_MODEL


class TestParameterCountMovesWithMaskDim:
    """A wrong-width conv is invisible to a total-parameter assertion alone."""

    def test_the_conv_kernel_shape_is_the_instrument(self):
        payload = _payload()
        narrow, wide = _head(mask_dim=4), _head(mask_dim=D_MODEL)
        _call(narrow, payload)
        _call(wide, payload)
        assert narrow.instance_seg_head.kernel.shape == (1, 1, D_MODEL, 4)
        assert wide.instance_seg_head.kernel.shape == (1, 1, D_MODEL, D_MODEL)


class TestMaskDimSurvivesAConfigRoundTrip:
    """A non-default ``mask_dim`` must reload as itself and stay callable."""

    def test_round_trip_rebuilds_a_callable_head(self):
        head = _head(mask_dim=D_MODEL // 2)
        clone = Sam3SegmentationHead.from_config(head.get_config())
        assert clone.mask_dim == D_MODEL // 2
        assert clone.instance_seg_head.filters == D_MODEL // 2
        out = _call(clone, _payload())
        assert tuple(out["pred_masks"].shape) == (BATCH, QUERIES, GRIDS[0], GRIDS[0])


class TestTheDocstringWasRightAndTheCodeWasWrong:
    """The fix direction is pinned by the class's own documentation."""

    def test_the_class_docstring_states_the_contract(self):
        doc = Sam3SegmentationHead.__doc__
        assert "Width of the mask embedding and therefore of the pixel" in doc, (
            "the :param mask_dim: entry is the oracle for the fix direction; "
            "if it is reworded, re-derive the direction before trusting it"
        )
