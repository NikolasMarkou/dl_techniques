"""Tests for ``SAM2`` -- plan step 8, guards G8.1 through G8.8.

This is the integration step, so most of these tests are about things that are
STRUCTURALLY invisible at the natural test point:

* a weight COUNT cannot see an internal-width change, and sampled after a
  forward pass it cannot see a failed restore either (G8.2 asserts that
  asymmetry explicitly rather than assuming it);
* a ``hiera_l`` parameter audit must not forward-pass the model, so it is
  driven by closed-form arithmetic derived HERE from the config and checked by
  a call spy (G8.3);
* the temporal embedding this class owns is the ONLY thing distinguishing
  memory frames from one another, because the rotary table inside memory
  attention is spatial-only and identical across frames (H-13);
* the streaming gradient boundary is proven TWO-SIDED -- with the boundary
  disabled the gradient must appear (G8.7).

Measured partitions are encoded as tests rather than assumed: ``call()`` is the
image path, so a ``fit()`` step on it legitimately starves every memory
component. That partition is asserted by name.
"""

import pathlib
import subprocess
import sys
from typing import Any, Dict, List
from unittest import mock

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

import dl_techniques.models.sam2.model as model_module
from dl_techniques.models.sam2.hiera import Hiera, hiera_block_specs
from dl_techniques.models.sam2.memory_encoder import SAM2MemoryEncoder
from dl_techniques.models.sam2.model import MEMORY_STRIDE, SAM2, create_sam2
from dl_techniques.models.sam2.neck import SAM2FpnNeck, SAM2ImageEncoder

# A-5: the oracle is IMPORTED from SAM 1's test package, never moved or copied.
from ..test_sam.dead_component_oracle import (
    NO_GRADIENTS_MESSAGE,
    fit_one_step_moved_variables,
    outputs_stop_gradient,
)

# ---------------------------------------------------------------------
# fixtures -- `tiny` is small enough to forward-pass, `hiera_l` never is
# ---------------------------------------------------------------------

TINY_IMAGE = 64
TINY_GRID = TINY_IMAGE // MEMORY_STRIDE          # 4
TINY_DIM = 32
TINY_MEM_DIM = 8
BATCH = 2

#: Published parameter count of SAM 2.1-L, used ONLY as a coarse control. The
#: measured figure lives in `test_hiera_l_total_is_recorded_with_its_band`.
PUBLISHED_HIERA_L_PARAMS = 224_000_000


def tiny_model(**overrides: Any) -> SAM2:
    """Build the small model. Never a module-level singleton -- several tests
    mutate weights and a shared instance would leak between them.

    :param overrides: Forwarded to :meth:`SAM2.from_variant`.
    :type overrides: Any
    :return: A BUILT model.
    :rtype: SAM2
    """
    model = SAM2.from_variant("tiny", **overrides)
    model.build(None)
    return model


def images(batch: int = BATCH, seed: int = 0) -> np.ndarray:
    """Seeded input images.

    :param batch: Batch size.
    :type batch: int
    :param seed: RNG seed.
    :type seed: int
    :return: ``(batch, 64, 64, 3)`` float32.
    :rtype: np.ndarray
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, TINY_IMAGE, TINY_IMAGE, 3)).astype(
        "float32")


def max_abs_diff(a: Any, b: Any) -> float:
    """Maximum absolute elementwise difference of two tensors.

    :param a: First tensor.
    :type a: Any
    :param b: Second tensor.
    :type b: Any
    :return: The maximum absolute difference.
    :rtype: float
    """
    return float(np.max(np.abs(
        ops.convert_to_numpy(a) - ops.convert_to_numpy(b))))


# ---------------------------------------------------------------------
# variants and the None-sentinel precedence rule (S-1, S-3)
# ---------------------------------------------------------------------


class TestVariants:
    """``MODEL_VARIANTS`` composes; it does not restate trunk geometry."""

    def test_only_two_variants_exist(self) -> None:
        """Inventing the other published sizes' numbers would be fabrication."""
        assert sorted(SAM2.MODEL_VARIANTS) == ["hiera_l", "tiny"]

    def test_variant_table_does_not_restate_trunk_or_neck_geometry(self) -> None:
        """A geometry restated in two homes is a latent defect.

        The trunk numbers live in ``Hiera.MODEL_VARIANTS`` and the neck/scalp
        numbers in ``SAM2ImageEncoder.MODEL_VARIANTS``. This asserts the SAM2
        table does not shadow any of them -- the failure mode is two tables
        drifting apart with no error anywhere.
        """
        forbidden = set(Hiera.MODEL_VARIANTS["hiera_l"]) | set(
            SAM2ImageEncoder.MODEL_VARIANTS["hiera_l"])
        for name, table in SAM2.MODEL_VARIANTS.items():
            clash = forbidden & set(table)
            assert clash == set(), (
                f"SAM2.MODEL_VARIANTS['{name}'] restates {sorted(clash)}, "
                f"which already live in Hiera/SAM2ImageEncoder"
            )

    def test_hiera_l_reads_its_geometry_from_the_trunk_table(self) -> None:
        """Every `hiera_l` number the plan pins is READ, not restated here."""
        trunk = Hiera.MODEL_VARIANTS["hiera_l"]
        assert trunk["embed_dim"] == 144
        assert trunk["num_heads"] == 2
        assert tuple(trunk["stages"]) == (2, 6, 36, 4)
        assert tuple(trunk["global_att_blocks"]) == (23, 33, 43)
        assert tuple(trunk["window_spec"]) == (8, 4, 16, 8)
        assert trunk["image_size"] == 1024
        encoder = SAM2ImageEncoder.MODEL_VARIANTS["hiera_l"]
        assert tuple(encoder["fpn_top_down_levels"]) == (2, 3)
        assert encoder["scalp"] == 1
        table = SAM2.MODEL_VARIANTS["hiera_l"]
        assert table["num_maskmem"] == 7
        assert table["mem_dim"] == 64
        assert table["memory_attention_layers"] == 4

    def test_unknown_variant_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown SAM2 variant"):
            SAM2.from_variant("hiera_b_plus")

    def test_image_size_cannot_be_overridden_through_the_variant(self) -> None:
        """Its single home is the trunk table; an override would desynchronize."""
        with pytest.raises(ValueError, match="single home is"):
            SAM2.from_variant("tiny", image_size=128)

    def test_tiny_geometry_is_self_consistent(self) -> None:
        model = tiny_model()
        assert model.image_size == TINY_IMAGE
        assert model.feature_grid == TINY_GRID
        assert model.hidden_dim == TINY_DIM
        assert model.mem_dim == TINY_MEM_DIM


class TestNoneSentinelPrecedence:
    """S-3: ``None`` defers to the table; an explicit value ALWAYS wins."""

    def test_none_defers_to_the_table(self) -> None:
        assert create_sam2("tiny").num_maskmem == \
            SAM2.MODEL_VARIANTS["tiny"]["num_maskmem"]

    def test_explicit_value_wins(self) -> None:
        assert create_sam2("tiny", num_maskmem=3).num_maskmem == 3

    def test_explicit_falsy_value_wins(self) -> None:
        """The discriminating case: ``False`` is not ``None``.

        A concrete default (rather than the ``None`` sentinel) would make this
        override indistinguishable from "argument omitted".
        """
        model = create_sam2("tiny", directly_add_no_mem_embed=False)
        assert model.directly_add_no_mem_embed is False
        assert SAM2.from_variant("tiny").directly_add_no_mem_embed is True

    def test_mem_dim_override_reaches_both_consumers(self) -> None:
        model = create_sam2("tiny", mem_dim=16)
        assert model.mem_dim == 16
        assert model.memory_attention.kv_in_dim == 16
        assert model.memory_encoder.out_dim == 16

    def test_image_size_none_defers_to_the_trunk(self) -> None:
        """The constructor-level half of the sentinel rule."""
        encoder = SAM2ImageEncoder.from_variant("tiny")
        assert encoder.trunk.image_size == TINY_IMAGE
        assert SAM2.from_variant("tiny").image_size == TINY_IMAGE


class TestComponentAgreement:
    """Every width or grid mismatch is refused at construction."""

    def _components(self) -> Dict[str, Any]:
        model = SAM2.from_variant("tiny")
        return {
            "image_encoder": model.image_encoder,
            "prompt_encoder": model.prompt_encoder,
            "mask_decoder": model.mask_decoder,
            "memory_attention": model.memory_attention,
            "memory_encoder": model.memory_encoder,
        }

    def test_a_mem_dim_mismatch_is_refused(self) -> None:
        parts = self._components()
        parts["memory_encoder"] = SAM2MemoryEncoder(
            in_dim=TINY_DIM, out_dim=16, mask_total_stride=MEMORY_STRIDE)
        with pytest.raises(ValueError, match="kv_in_dim"):
            SAM2(**parts)

    def test_a_feature_grid_mismatch_is_refused(self) -> None:
        """A-1 consistency: memory attention's ``feat_sizes`` must equal the
        stride-16 grid the encoder actually returns."""
        parts = self._components()
        parts["memory_attention"] = model_module.SAM2MemoryAttention(
            d_model=TINY_DIM, num_layers=1, dim_feedforward=16,
            feat_sizes=(TINY_GRID * 2, TINY_GRID * 2), kv_in_dim=TINY_MEM_DIM)
        with pytest.raises(ValueError, match="feat_sizes"):
            SAM2(**parts)

    def test_a_decoder_without_high_res_features_is_refused(self) -> None:
        parts = self._components()
        parts["mask_decoder"] = model_module.SAM2MaskDecoder(
            transformer_dim=TINY_DIM,
            transformer=model_module.TwoWayTransformer(
                depth=1, embedding_dim=TINY_DIM, num_heads=2, mlp_dim=32),
            use_high_res_features=False,
        )
        with pytest.raises(ValueError, match="use_high_res_features"):
            SAM2(**parts)


class TestA1StrideReconciliation:
    """A-1 is a READ fact (F-6 § G-1c), so this is a consistency check."""

    def test_vision_features_are_stride_16_at_the_shipped_resolution(
            self) -> None:
        """At ``image_size=1024`` the retained coarsest level is 64x64."""
        shapes = SAM2ImageEncoder.from_variant("hiera_l").compute_output_shape(
            (None, 1024, 1024, 3))
        assert shapes["vision_features"][1:3] == (64, 64)
        assert 1024 // shapes["vision_features"][1] == MEMORY_STRIDE
        assert len(shapes["backbone_fpn"]) == 3

    def test_the_grid_reconciles_with_memory_attention_feat_sizes(self) -> None:
        model = SAM2.from_variant("hiera_l")
        assert tuple(model.memory_attention.feat_sizes) == (64, 64)
        assert model.feature_grid == 64

    def test_tiny_reproduces_the_same_ladder(self) -> None:
        """Forward-passed, so this is a VALUE check, not a shape contract."""
        model = tiny_model()
        encoded = model.image_encoder(images(1))
        assert tuple(encoded["vision_features"].shape) == \
            (1, TINY_GRID, TINY_GRID, TINY_DIM)
        assert TINY_IMAGE // TINY_GRID == MEMORY_STRIDE


# ---------------------------------------------------------------------
# G8.1 -- serialization BY VALUE
# ---------------------------------------------------------------------


class TestSerializationByValue:
    """G8.1: a ``.keras`` round-trip on a BUILT model, compared elementwise."""

    def test_round_trip_is_bit_identical(self, tmp_path: Any) -> None:
        """`isinstance` proves nothing here; every weight is compared by VALUE."""
        model = tiny_model()
        x = images()
        reference = model({"image": x})["low_res_logits"]

        path = tmp_path / "sam2_tiny.keras"
        model.save(path)
        restored = keras.models.load_model(path)

        assert len(restored.weights) == len(model.weights)
        for source, target in zip(model.weights, restored.weights):
            # The restored model's variable PATHS are shorter (the sub-layers
            # are reconstructed before being attached to the outer model), so
            # the alignment is asserted on the common suffix rather than on the
            # full path -- the guard is the VALUE comparison below.
            assert source.path.endswith(target.path) \
                or target.path.endswith(source.path), (
                f"weight order diverged: {source.path} vs {target.path}"
            )
            assert max_abs_diff(source, target) == 0.0, (
                f"weight '{source.path}' did not survive the round trip"
            )
        assert max_abs_diff(
            reference, restored({"image": x})["low_res_logits"]) == 0.0

    def test_config_round_trip_preserves_every_init_parameter(self) -> None:
        model = SAM2.from_variant(
            "tiny", num_maskmem=3, directly_add_no_mem_embed=False)
        rebuilt = SAM2.from_config(model.get_config())
        assert rebuilt.num_maskmem == 3
        assert rebuilt.directly_add_no_mem_embed is False
        assert rebuilt.image_size == model.image_size
        assert rebuilt.mem_dim == model.mem_dim
        assert rebuilt.hidden_dim == model.hidden_dim

    def test_the_four_owned_weights_exist_with_their_declared_shapes(
            self) -> None:
        model = tiny_model()
        assert tuple(model.maskmem_tpos_enc.shape) == \
            (model.num_maskmem, 1, 1, TINY_MEM_DIM)
        assert tuple(model.no_mem_embed.shape) == (1, 1, TINY_DIM)
        assert tuple(model.no_mem_pos_enc.shape) == (1, 1, TINY_MEM_DIM)
        assert tuple(model.no_obj_ptr.shape) == (1, TINY_DIM)


# ---------------------------------------------------------------------
# G8.2 -- weight count vs parameter count, and their ASYMMETRY
# ---------------------------------------------------------------------


class TestWeightLayoutInvariant:
    """G8.2: ``count_params()`` is the PRIMARY invariant; the count is blind."""

    def test_weight_count_is_sampled_before_the_first_forward_call(
            self, tmp_path: Any) -> None:
        """Sampled AFTER a forward pass the count is filled by fresh weights.

        SAM 1 measured exactly this: 138 of 202 weights restored while
        ``len(model.weights)`` read 202 both ways. The sample point is the
        assertion.
        """
        model = tiny_model()
        model({"image": images(1)})
        expected_weights = len(model.weights)
        expected_params = model.count_params()

        path = tmp_path / "sam2_tiny.keras"
        model.save(path)
        restored = keras.models.load_model(path)

        # NO forward call between the load and these two reads.
        assert len(restored.weights) == expected_weights
        assert restored.count_params() == expected_params

    def test_param_count_fires_and_weight_count_does_not(self) -> None:
        """The asymmetry, asserted explicitly rather than assumed.

        Halving the memory attention's internal width changes the SHAPE of six
        projections per layer and changes NO variable count. A guard built on
        ``len(model.weights)`` alone is therefore structurally blind to a
        ``downsample_rate`` / ``key_dim`` regression.
        """
        shipped = tiny_model()
        halved = tiny_model(memory_attention_downsample_rate=2)

        assert len(halved.weights) == len(shipped.weights), (
            "the weight COUNT is expected to be blind here -- if this fires, "
            "the mutation changed the layout and the asymmetry claim is wrong"
        )
        assert halved.count_params() != shipped.count_params(), (
            f"count_params() did not see the halved internal width: "
            f"{halved.count_params()} == {shipped.count_params()}"
        )
        assert halved.memory_attention.layers[0].self_attn.internal_dim == \
            TINY_DIM // 2


# ---------------------------------------------------------------------
# G8.3 -- the `hiera_l` parameter audit, WITHOUT a forward pass
# ---------------------------------------------------------------------


def trunk_closed_form(variant: str) -> int:
    """Closed-form trunk parameter count, derived HERE from the config.

    Deliberately independent of the implementation: it walks the block schedule
    and adds up the layer shapes the architecture implies, so it disagrees with
    the shipped model if either drifts.

    :param variant: Key of ``Hiera.MODEL_VARIANTS``.
    :type variant: str
    :return: Parameter count.
    :rtype: int
    """
    cfg = Hiera.MODEL_VARIANTS[variant]
    embed = int(cfg["embed_dim"])
    bkg_h, bkg_w = cfg.get("window_pos_embed_bkg_spatial_size", (7, 7))
    window_edge = int(cfg["window_spec"][0])

    # stem conv (7x7x3 -> embed) + background PE + tiled window PE
    total = 7 * 7 * 3 * embed + embed
    total += bkg_h * bkg_w * embed
    total += window_edge * window_edge * embed

    specs = hiera_block_specs(
        stages=tuple(cfg["stages"]),
        window_spec=tuple(cfg["window_spec"]),
        global_att_blocks=tuple(cfg["global_att_blocks"]),
        q_pool=int(cfg["q_pool"]),
        embed_dim=embed,
        num_heads=int(cfg["num_heads"]),
        dim_mul=2.0,
        head_mul=2.0,
    )
    for spec in specs:
        dim, dim_out = int(spec["dim"]), int(spec["dim_out"])
        hidden = int(dim_out * 4.0)
        total += 2 * dim                                    # norm1
        if dim != dim_out:                                  # shortcut proj
            total += dim * dim_out + dim_out
        total += dim * 3 * dim_out + 3 * dim_out            # qkv
        total += dim_out * dim_out + dim_out                # attn proj
        total += 2 * dim_out                                # norm2
        total += dim_out * hidden + hidden                  # mlp fc1
        total += hidden * dim_out + dim_out                 # mlp fc2
    return total


def neck_closed_form(variant: str) -> int:
    """Closed-form neck parameter count: one 1x1 lateral conv per level.

    :param variant: Key of ``SAM2ImageEncoder.MODEL_VARIANTS``.
    :type variant: str
    :return: Parameter count.
    :rtype: int
    """
    d_model = int(SAM2ImageEncoder.MODEL_VARIANTS[variant]["d_model"])
    channels = Hiera.from_variant(variant).channel_list
    return sum(int(c) * d_model + d_model for c in channels)


def memory_attention_closed_form(d_model: int, kv_dim: int, layers: int,
                                 feedforward: int) -> int:
    """Closed-form memory-attention count at ``downsample_rate == 1``.

    :param d_model: Query width.
    :type d_model: int
    :param kv_dim: Memory key/value width.
    :type kv_dim: int
    :param layers: Number of blocks.
    :type layers: int
    :param feedforward: FFN hidden width.
    :type feedforward: int
    :return: Parameter count.
    :rtype: int
    """
    def attention(kv_in: int) -> int:
        internal = d_model
        return (
            d_model * internal + internal          # q_proj
            + 2 * (kv_in * internal + internal)    # k_proj, v_proj
            + internal * d_model + d_model         # out_proj
        )

    per_layer = (
        attention(d_model)                          # self-attention
        + attention(kv_dim)                         # cross-attention
        + d_model * feedforward + feedforward       # linear1
        + feedforward * d_model + d_model           # linear2
        + 3 * 2 * d_model                           # norm1..norm3
    )
    return layers * per_layer + 2 * d_model         # + the final norm


def memory_encoder_closed_form(model: SAM2) -> int:
    """Closed-form memory-encoder count, read from the layer's own geometry.

    The channel ladder is DERIVED by the layer from its stride (D-016), so it is
    read from ``channel_sequence`` rather than hardcoded -- hardcoding it would
    make this oracle only correct at stride 2.

    :param model: A constructed model.
    :type model: SAM2
    :return: Parameter count.
    :rtype: int
    """
    encoder = model.memory_encoder
    in_dim, out_dim = encoder.in_dim, encoder.out_dim
    kernel = encoder.mask_kernel_size

    total, previous = 0, encoder.mask_in_chans
    for channels in encoder.mask_downsampler.channel_sequence:
        total += kernel * kernel * previous * channels + channels  # conv
        total += 2 * channels                                      # layer norm
        previous = channels
    total += previous * in_dim + in_dim                            # final 1x1

    total += in_dim * in_dim + in_dim                              # pix_feat_proj
    block = (
        encoder.fuser_kernel_size ** 2 * in_dim + in_dim           # depthwise
        + 2 * in_dim                                               # layer norm
        + in_dim * 4 * in_dim + 4 * in_dim                          # expand
        + 4 * in_dim * in_dim + in_dim                              # reduce
        + in_dim                                                    # layer scale
    )
    total += encoder.num_fuser_layers * block
    total += in_dim * out_dim + out_dim                            # out_proj
    return total


class TestHieraLargeParameterAudit:
    """G8.3: construct-and-count at ``hiera_l``, with NO forward pass."""

    @pytest.fixture(scope="class")
    def large(self) -> SAM2:
        """Construct `hiera_l` ON CPU -- it allocates roughly 0.9 GB.

        :return: A built `hiera_l` model that is never forward-passed.
        :rtype: SAM2
        """
        with tf.device("/CPU:0"):
            model = SAM2.from_variant("hiera_l")
            model.build(None)
        return model

    def test_no_forward_pass_happens_during_construction(self) -> None:
        """The call spy. A forward pass at this size is minutes and gigabytes."""
        calls: List[int] = []
        original = Hiera.call

        def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return original(self, *args, **kwargs)

        with mock.patch.object(Hiera, "call", spy), tf.device("/CPU:0"):
            model = SAM2.from_variant("hiera_l")
            model.build(None)
            _ = model.count_params()
        assert calls == [], f"the trunk was forward-passed {len(calls)} times"

    def test_trunk_matches_its_closed_form(self, large: SAM2) -> None:
        assert large.image_encoder.trunk.count_params() == \
            trunk_closed_form("hiera_l")

    def test_neck_matches_its_closed_form(self, large: SAM2) -> None:
        assert large.image_encoder.neck.count_params() == \
            neck_closed_form("hiera_l")

    def test_memory_attention_matches_its_closed_form(self, large: SAM2) -> None:
        table = SAM2.MODEL_VARIANTS["hiera_l"]
        assert large.memory_attention.count_params() == \
            memory_attention_closed_form(
                d_model=large.hidden_dim,
                kv_dim=large.mem_dim,
                layers=table["memory_attention_layers"],
                feedforward=table["memory_attention_dim_feedforward"],
            )

    def test_memory_encoder_matches_its_closed_form(self, large: SAM2) -> None:
        assert large.memory_encoder.count_params() == \
            memory_encoder_closed_form(large)

    def test_owned_weights_match_their_closed_form(self, large: SAM2) -> None:
        expected = (
            large.num_maskmem * large.mem_dim   # maskmem_tpos_enc
            + large.hidden_dim                  # no_mem_embed
            + large.mem_dim                     # no_mem_pos_enc
            + large.hidden_dim                  # no_obj_ptr
        )
        owned = sum(
            int(np.prod(w.shape)) for w in (
                large.maskmem_tpos_enc, large.no_mem_embed,
                large.no_mem_pos_enc, large.no_obj_ptr)
        )
        assert owned == expected

    def test_total_equals_the_sum_of_its_components(self, large: SAM2) -> None:
        """No component is double-counted and none is missing."""
        components = (
            large.image_encoder.count_params()
            + large.prompt_encoder.count_params()
            + large.mask_decoder.count_params()
            + large.memory_attention.count_params()
            + large.memory_encoder.count_params()
        )
        owned = sum(
            int(np.prod(w.shape)) for w in (
                large.maskmem_tpos_enc, large.no_mem_embed,
                large.no_mem_pos_enc, large.no_obj_ptr)
        )
        assert large.count_params() == components + owned

    def test_hiera_l_total_is_recorded_with_its_band(self, large: SAM2) -> None:
        """The ONE home of the measured total, with its derivation beside it.

        MEASURED at this HEAD: **220,941,537** built parameters. That figure is
        not pinned as a literal equality because it is an output of the shipped
        code, not an input to it; what is asserted is the band around the
        published ~224M.

        The ~3M gap is ACCOUNTED FOR, not a fudge: SAM 1's ``TwoWayTransformer``
        builds its attention and FFN sub-layers lazily on first call, so at
        ``depth=2, dim=256, heads=8, mlp=2048`` roughly 4.2M parameters do not
        exist yet in a model that has never been forward-passed. Counting them
        would require the forward pass this test exists to avoid.
        """
        total = large.count_params()
        assert 200_000_000 <= total <= 250_000_000, (
            f"hiera_l built parameter count {total:,} is outside the coarse "
            f"band around the published ~{PUBLISHED_HIERA_L_PARAMS:,}"
        )
        assert large.mask_decoder.transformer.weights == [], (
            "the two-way transformer is built LAZILY; if it now holds "
            "variables, the gap explanation above is stale and the band "
            "should be re-derived"
        )


# ---------------------------------------------------------------------
# G8.4 -- traceability
# ---------------------------------------------------------------------


class TestGraphTrace:
    """G8.4: ``call()`` traces under a static input signature."""

    def test_call_traces(self) -> None:
        model = tiny_model()
        model({"image": images()})

        @tf.function(input_signature=[
            tf.TensorSpec((BATCH, TINY_IMAGE, TINY_IMAGE, 3), tf.float32)])
        def traced(x: Any) -> Any:
            return model({"image": x})["low_res_logits"]

        assert traced.get_concrete_function().output_shapes == \
            (BATCH, 1, TINY_GRID * 4, TINY_GRID * 4)

    def test_the_trace_guard_is_not_vacuous(self) -> None:
        """The mutation that ACTUALLY fires on this stack.

        The plan named a dynamic-size ``ops.image.resize``; step 2 measured that
        INERT (a dynamic size traces fine here -- the real constraint is that
        the ``size`` argument be ``len()``-able). Step 3's measured substitute
        is ``ops.convert_to_numpy`` inside the traced body, which raises
        ``NotImplementedError``.
        """
        model = tiny_model()
        model({"image": images()})

        @tf.function(input_signature=[
            tf.TensorSpec((BATCH, TINY_IMAGE, TINY_IMAGE, 3), tf.float32)])
        def traced(x: Any) -> Any:
            out = model({"image": x})["low_res_logits"]
            return out + float(np.sum(ops.convert_to_numpy(out)))

        with pytest.raises(NotImplementedError):
            traced.get_concrete_function()

    def test_stream_step_is_not_traced(self) -> None:
        """The video path mutates Python state, so it must stay eager.

        Asserted by the state change itself: a traced function would either
        raise or run the Python side effect exactly once.
        """
        model = tiny_model()
        model.stream_reset()
        model.stream_step(images(1), frame_idx=0, is_conditioning=True)
        assert model.memory_bank.num_frames == 1
        model.stream_step(images(1, seed=1), frame_idx=1)
        assert model.memory_bank.num_frames == 2


# ---------------------------------------------------------------------
# G8.5 -- registered-key uniqueness
# ---------------------------------------------------------------------

SAM2_REGISTERED_CLASSES = (
    "AxialRoPE2D",
    "Hiera",
    "HieraBlock",
    "HieraMultiScaleAttention",
    "HieraPatchEmbed",
    "SAM2",
    "SAM2FpnNeck",
    "SAM2Fuser",
    "SAM2ImageEncoder",
    "SAM2MaskDecoder",
    "SAM2MaskDownSampler",
    "SAM2MemoryAttention",
    "SAM2MemoryAttentionLayer",
    "SAM2MemoryEncoder",
)

_REGISTRY_PROBE = """
import json
import keras

names = {names!r}
before = {{n: [k for k in keras.saving.get_custom_objects()
              if k.endswith(">" + n)] for n in names}}
import dl_techniques.models.sam2.model  # noqa: F401
after = {{n: [k for k in keras.saving.get_custom_objects()
             if k.endswith(">" + n)] for n in names}}
print("RESULT " + json.dumps({{"before": before, "after": after}}))
"""


class TestRegisteredKeyUniqueness:
    """G8.5: absent BEFORE the import, present exactly once AFTER.

    Run in a subprocess because the module is already imported in this one, so
    the "before" half is unobservable here. A duplicate bare
    ``@register_keras_serializable()`` OVERWRITES silently -- the one
    SAM-1-perturbing mechanism the ``git diff`` proxy cannot see.
    """

    def test_every_key_is_absent_before_and_unique_after(self) -> None:
        probe = _REGISTRY_PROBE.format(names=list(SAM2_REGISTERED_CLASSES))
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, check=True,
        )
        payload = [line for line in result.stdout.splitlines()
                   if line.startswith("RESULT ")]
        assert payload, f"probe produced no result:\n{result.stdout}"
        import json
        data = json.loads(payload[-1][len("RESULT "):])
        for name in SAM2_REGISTERED_CLASSES:
            assert data["before"][name] == [], (
                f"'{name}' was ALREADY registered before the SAM 2 import: "
                f"{data['before'][name]}"
            )
            assert len(data["after"][name]) == 1, (
                f"'{name}' is registered {len(data['after'][name])} times: "
                f"{data['after'][name]}"
            )

    def test_sam2_does_not_shadow_a_sam1_key(self) -> None:
        registry = keras.saving.get_custom_objects()
        assert registry["Custom>SAM2"] is SAM2
        assert "Custom>SAM" in registry
        assert registry["Custom>SAM"] is not SAM2

    def test_sam1_source_is_untouched(self) -> None:
        result = subprocess.run(
            ["git", "diff", "--stat", "--", "src/dl_techniques/models/sam/"],
            capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == "", (
            f"SAM 1 source was modified:\n{result.stdout}"
        )


# ---------------------------------------------------------------------
# G8.6 -- one fit() step, and the MEASURED moved/unmoved partition
# ---------------------------------------------------------------------


def compiled_tiny() -> SAM2:
    """A built, compiled model ready for one ``fit()`` step.

    ``jit_compile=False`` is REQUIRED, not stylistic: the trunk's stem
    positional embedding uses a bicubic ``ops.image.resize``, which XLA refuses
    to convert (``tf2xla conversion failed``). Measured at this HEAD.

    :return: The compiled model.
    :rtype: SAM2
    """
    model = tiny_model()
    model({"image": images()})
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=1.0),
        loss={"low_res_logits": keras.losses.MeanSquaredError()},
        jit_compile=False,
    )
    return model


def fit_payload() -> Any:
    """Inputs and targets for one training step.

    :return: ``(x, y)``.
    :rtype: Any
    """
    x = {"image": images()}
    y = {"low_res_logits": np.zeros(
        (BATCH, 1, TINY_GRID * 4, TINY_GRID * 4), dtype="float32")}
    return x, y


class TestTrainingStep:
    """G8.6: the moved/unmoved partition, asserted BY NAME."""

    @pytest.fixture(scope="class")
    def report(self) -> Any:
        """Run exactly one ``fit()`` step and report the movement.

        :return: A ``MovedVariablesReport``.
        :rtype: Any
        """
        model = compiled_tiny()
        x, y = fit_payload()
        return fit_one_step_moved_variables(model, x, y)

    def test_the_image_path_moves(self, report: Any) -> None:
        """Trunk, neck and the decoder's mask path all carry gradient."""
        for fragment in (
                "hiera", "fpn_neck", "hypernetwork_mlp_0", "upsample_conv1",
                "mask_tokens", "two_way_transformer",
        ):
            moved = [n for n in report.moved if fragment in n]
            assert moved, (
                f"nothing matching '{fragment}' moved; report={report.summary()}"
            )

    def test_the_memory_path_is_legitimately_starved(self, report: Any) -> None:
        """MEASURED, and encoded rather than assumed.

        ``call()`` is the IMAGE path: it never reads the memory bank and never
        runs memory attention or the memory encoder, so a single-frame ``fit()``
        step cannot move them. Asserting "everything moves" here would be a
        guard that is permanently red on a correct implementation -- the same
        inversion step 7 measured for the multimask-starved hypernetwork.
        """
        for fragment in (
                "memory_attention", "memory_encoder",
                "maskmem_tpos_enc", "no_mem_embed", "no_mem_pos_enc",
        ):
            assert not [n for n in report.moved if fragment in n], (
                f"'{fragment}' moved during an IMAGE-path step, which the "
                f"image path cannot reach"
            )
            assert [n for n in report.unmoved if fragment in n], (
                f"'{fragment}' is missing from the report entirely"
            )

    def test_the_iou_head_is_starved_by_the_single_key_loss(
            self, report: Any) -> None:
        """The loss supervises ``low_res_logits`` only. Named, not silent."""
        starved = [n for n in report.unmoved if "iou_prediction_head" in n]
        assert starved, "the IoU head is missing from the report"

    def test_the_deltas_are_reported_numerically(self, report: Any) -> None:
        """A partition claim without the number behind it is not a measurement."""
        moved_trunk = [n for n in report.moved if "hiera" in n]
        deltas = [report.max_abs_delta[n] for n in moved_trunk]
        assert min(deltas) > 0.0, (
            f"a 'moved' variable had a zero delta: {report.summary()}"
        )

    def test_the_instrument_itself_is_live(self) -> None:
        """The dead-component control. Without it a green partition above
        proves nothing.

        MEASURED, and it corrects the oracle's own docstring for this model:
        with every output detached, Keras 3.8 does NOT raise
        ``"%s"`` here -- it emits a
        ``UserWarning`` naming every variable and completes the step. So the
        observable is the PARTITION (nothing moved), not the raise.
        """ % NO_GRADIENTS_MESSAGE
        model = compiled_tiny()
        x, y = fit_payload()
        with outputs_stop_gradient(model):
            report = fit_one_step_moved_variables(model, x, y)
        assert report.moved == (), (
            f"variables moved with every output detached: {report.summary()}"
        )
        assert len(report.unmoved) == report.total


# ---------------------------------------------------------------------
# G8.7 -- streaming: bank growth and the gradient boundary
# ---------------------------------------------------------------------


class TestStreaming:
    """G8.7: three frames, the predicted bank growth, and the H-4 boundary."""

    def test_the_bank_grows_as_the_policy_predicts(self) -> None:
        model = tiny_model()
        model.stream_reset()

        spatial = TINY_GRID * TINY_GRID          # 16 memory tokens per frame
        pointer = TINY_DIM // TINY_MEM_DIM       # 4 tokens per object pointer

        observed = []
        for index in range(3):
            out = model.stream_step(
                images(1, seed=index), frame_idx=index,
                is_conditioning=(index == 0))
            observed.append(
                (out["num_memory_tokens"], out["num_obj_ptr_tokens"]))

        assert observed == [
            (0, 0),                                     # empty bank
            (spatial + pointer, pointer),               # frame 0 in memory
            (2 * spatial + 2 * pointer, 2 * pointer),   # frames 0 and 1
        ]
        assert model.memory_bank.num_frames == 3
        assert sorted(model.memory_bank.cond_frames) == [0]

    def test_stream_reset_clears_the_bank(self) -> None:
        model = tiny_model()
        model.stream_reset()
        model.stream_step(images(1), frame_idx=0, is_conditioning=True)
        assert not model.memory_bank.is_empty
        model.stream_reset()
        assert model.memory_bank.is_empty
        assert model._stream_frame_counter == 0

    def _gradient_through_the_stream(self, model: SAM2, source: Any,
                                     other: np.ndarray) -> Any:
        """Differentiate a later frame's output wrt an earlier frame's input.

        :param model: The model.
        :type model: SAM2
        :param source: A watched ``tf.Variable`` used as frame 0.
        :type source: Any
        :param other: Frames 1 and 2.
        :type other: np.ndarray
        :return: The gradient, or ``None``.
        :rtype: Any
        """
        model.stream_reset()
        with tf.GradientTape() as tape:
            model.stream_step(source, frame_idx=0, is_conditioning=True)
            model.stream_step(other, frame_idx=1)
            out = model.stream_step(other, frame_idx=2)
            loss = ops.sum(ops.square(out["low_res_logits"]))
        return tape.gradient(loss, source)

    def test_gradient_does_not_reach_frame_zero(self) -> None:
        """H-4: without the boundary, N frames build one N-deep graph."""
        model = tiny_model()
        source = tf.Variable(images(1, seed=7))
        assert self._gradient_through_the_stream(
            model, source, images(1, seed=8)) is None

    def test_the_boundary_is_what_stops_it(self) -> None:
        """The other side of the guard. A one-sided ``is None`` assertion is
        satisfied by a model whose streaming path is dead altogether."""
        model = tiny_model()
        source = tf.Variable(images(1, seed=7))
        with mock.patch.object(model_module.ops, "stop_gradient", lambda x: x):
            gradient = self._gradient_through_the_stream(
                model, source, images(1, seed=8))
        assert gradient is not None, (
            "with `stop_gradient` disabled the gradient MUST reach frame 0; "
            "if it does not, the frames are not actually connected and the "
            "positive arm of this guard proves nothing"
        )
        assert float(np.max(np.abs(ops.convert_to_numpy(gradient)))) > 0.0

    def test_the_single_frame_path_does_carry_gradient(self) -> None:
        """The control for the control: one frame IS differentiable."""
        model = tiny_model()
        source = tf.Variable(images(1, seed=7))
        model.stream_reset()
        with tf.GradientTape() as tape:
            out = model.stream_step(source, frame_idx=0, is_conditioning=True)
            loss = ops.sum(ops.square(out["low_res_logits"]))
        gradient = tape.gradient(loss, source)
        assert gradient is not None
        assert float(np.max(np.abs(ops.convert_to_numpy(gradient)))) > 0.0

    def test_the_first_frame_takes_the_no_memory_path(self) -> None:
        """``directly_add_no_mem_embed`` skips memory attention entirely."""
        model = tiny_model()
        model.stream_reset()
        calls: List[int] = []
        original = type(model.memory_attention).call

        def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return original(self, *args, **kwargs)

        with mock.patch.object(type(model.memory_attention), "call", spy):
            model.stream_step(images(1), frame_idx=0, is_conditioning=True)
            assert calls == []
            model.stream_step(images(1, seed=1), frame_idx=1)
            assert len(calls) == 1

    def test_no_mem_embed_is_what_the_first_frame_adds(self) -> None:
        """A VALUE assertion, not a branch-coverage one."""
        model = tiny_model()
        model.stream_reset()
        reference = ops.convert_to_numpy(
            model.stream_step(images(1), frame_idx=0,
                              is_conditioning=True)["low_res_logits"])

        model.no_mem_embed.assign(
            np.full(model.no_mem_embed.shape, 5.0, dtype="float32"))
        model.stream_reset()
        perturbed = ops.convert_to_numpy(
            model.stream_step(images(1), frame_idx=0,
                              is_conditioning=True)["low_res_logits"])
        assert float(np.max(np.abs(reference - perturbed))) > 1e-4


# ---------------------------------------------------------------------
# H-13 -- the temporal embedding is the ONLY temporal signal
# ---------------------------------------------------------------------


class TestTemporalEmbedding:
    """The ``maskmem_tpos_enc`` / RoPE split, guarded by value."""

    def test_the_bank_returns_slot_indices_and_this_class_adds_the_vectors(
            self) -> None:
        """The split itself: the bank owns no embedding table."""
        model = tiny_model()
        assert not hasattr(model.memory_bank, "maskmem_tpos_enc")
        readout = model.memory_bank.read(0)
        assert readout.tpos_slots == ()
        model.memory_bank.add_frame(
            0,
            maskmem_features=np.zeros(
                (1, TINY_GRID, TINY_GRID, TINY_MEM_DIM), dtype="float32"),
            maskmem_pos_enc=np.zeros(
                (1, TINY_GRID, TINY_GRID, TINY_MEM_DIM), dtype="float32"),
            obj_ptr=np.zeros((1, TINY_DIM), dtype="float32"),
            is_conditioning=True,
        )
        readout = model.memory_bank.read(1)
        assert readout.tpos_slots == (model.num_maskmem - 1,)

    def test_slots_are_expanded_per_token_and_pointers_get_zeros(self) -> None:
        """The gather is per TOKEN, not per frame; the tail is unencoded."""
        model = tiny_model()
        model.stream_reset()
        model.stream_step(images(1), frame_idx=0, is_conditioning=True)
        readout = model.memory_bank.read(1)
        encoding = model._temporal_embedding(readout, readout.memory)

        assert tuple(encoding.shape) == (1, int(readout.memory.shape[1]),
                                         TINY_MEM_DIM)
        tail = ops.convert_to_numpy(
            encoding[:, -readout.num_obj_ptr_tokens:, :])
        assert float(np.max(np.abs(tail))) == 0.0

    def test_changing_a_slot_row_changes_the_conditioned_output(self) -> None:
        """The discriminating observation: a DEAD table would be invisible.

        Every other temporal mechanism in the stack is off: the rotary table is
        spatial-only and identical across memory frames (``repeat_k``), so if
        this weight did not reach the output there would be nothing left to
        tell frame ``t-1`` from frame ``t-6``.
        """
        model = tiny_model()

        def two_frames() -> np.ndarray:
            model.stream_reset()
            model.stream_step(images(1), frame_idx=0, is_conditioning=True)
            out = model.stream_step(images(1, seed=1), frame_idx=1)
            return ops.convert_to_numpy(out["low_res_logits"])

        reference = two_frames()
        table = np.zeros(model.maskmem_tpos_enc.shape, dtype="float32")
        table[model.num_maskmem - 1] = 3.0
        model.maskmem_tpos_enc.assign(table)
        assert float(np.max(np.abs(reference - two_frames()))) > 1e-5

    def test_a_different_slot_row_is_a_different_signal(self) -> None:
        """Two rows must not be interchangeable, or the slot index is dead."""
        model = tiny_model()

        def two_frames() -> np.ndarray:
            model.stream_reset()
            model.stream_step(images(1), frame_idx=0, is_conditioning=True)
            out = model.stream_step(images(1, seed=1), frame_idx=1)
            return ops.convert_to_numpy(out["low_res_logits"])

        table = np.zeros(model.maskmem_tpos_enc.shape, dtype="float32")
        table[model.num_maskmem - 1] = 3.0
        model.maskmem_tpos_enc.assign(table)
        selected = two_frames()

        other = np.zeros(model.maskmem_tpos_enc.shape, dtype="float32")
        other[0] = 3.0
        model.maskmem_tpos_enc.assign(other)
        assert float(np.max(np.abs(selected - two_frames()))) > 1e-5


# ---------------------------------------------------------------------
# the object-pointer blend
# ---------------------------------------------------------------------


class TestObjectPointerBlend:
    """``no_obj_ptr`` is interpolated towards by the object score."""

    def test_a_saturated_score_returns_the_predicted_pointer(self) -> None:
        model = tiny_model()
        model.no_obj_ptr.assign(np.full((1, TINY_DIM), 9.0, dtype="float32"))
        pointer = np.ones((BATCH, TINY_DIM), dtype="float32")
        blended = ops.convert_to_numpy(model._blend_object_pointer(
            pointer, np.full((BATCH, 1), 30.0, dtype="float32")))
        assert np.max(np.abs(blended - pointer)) < 1e-5

    def test_a_saturated_negative_score_returns_no_obj_ptr(self) -> None:
        model = tiny_model()
        model.no_obj_ptr.assign(np.full((1, TINY_DIM), 9.0, dtype="float32"))
        blended = ops.convert_to_numpy(model._blend_object_pointer(
            np.ones((BATCH, TINY_DIM), dtype="float32"),
            np.full((BATCH, 1), -30.0, dtype="float32")))
        assert np.max(np.abs(blended - 9.0)) < 1e-4

    def test_the_blend_is_per_batch_element(self) -> None:
        """A batch whose rows disagree -- a uniform batch would be vacuous."""
        model = tiny_model()
        model.no_obj_ptr.assign(np.full((1, TINY_DIM), 9.0, dtype="float32"))
        blended = ops.convert_to_numpy(model._blend_object_pointer(
            np.ones((2, TINY_DIM), dtype="float32"),
            np.array([[30.0], [-30.0]], dtype="float32")))
        assert np.max(np.abs(blended[0] - 1.0)) < 1e-5
        assert np.max(np.abs(blended[1] - 9.0)) < 1e-4

    def test_the_output_pointer_width_matches_the_memory_bank(self) -> None:
        model = tiny_model()
        out = model({"image": images()})
        assert tuple(out["object_pointer"].shape) == (BATCH, TINY_DIM)
        assert model.hidden_dim % model.mem_dim == 0


# ---------------------------------------------------------------------
# output contract
# ---------------------------------------------------------------------


class TestOutputContract:
    """Shapes, keys and the multimask override."""

    def test_call_returns_the_four_keys(self) -> None:
        out = tiny_model()({"image": images()})
        assert sorted(out) == [
            "iou_predictions", "low_res_logits", "object_pointer",
            "object_score_logits",
        ]

    def test_shapes_match_compute_output_shape(self) -> None:
        model = tiny_model()
        declared = model.compute_output_shape(
            {"image": (BATCH, TINY_IMAGE, TINY_IMAGE, 3)})
        actual = model({"image": images()})
        for key, shape in declared.items():
            assert tuple(actual[key].shape) == shape, key

    def test_multimask_override_wins_over_the_configured_default(self) -> None:
        model = tiny_model()
        single = model({"image": images()})["low_res_logits"]
        multi = model({"image": images()}, multimask_output=True)[
            "low_res_logits"]
        assert single.shape[1] == 1
        assert multi.shape[1] == model.mask_decoder.num_multimask_outputs

    def test_missing_image_key_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'image' key"):
            tiny_model()({"points": None})

    def test_prompts_reach_the_output(self) -> None:
        """A box prompt must change the prediction, or the prompt path is dead."""
        model = tiny_model()
        x = images()
        plain = ops.convert_to_numpy(model({"image": x})["low_res_logits"])
        boxes = np.tile(
            np.array([[[8.0, 8.0, 40.0, 40.0]]], dtype="float32"),
            (BATCH, 1, 1))
        prompted = ops.convert_to_numpy(
            model({"image": x, "boxes": boxes})["low_res_logits"])
        assert float(np.max(np.abs(plain - prompted))) > 1e-5


# ---------------------------------------------------------------------
# keras.ops purity
# ---------------------------------------------------------------------


class TestOpsPurity:
    """H-6: no raw ``tf.`` CALLs in the new source."""

    def test_no_raw_tensorflow_calls(self) -> None:
        source = pathlib.Path(
            "src/dl_techniques/models/sam2/model.py").read_text()
        assert "import tensorflow" not in source, (
            "model.py imports tensorflow; no raw tf. CALL is possible without "
            "it, so this is the load-bearing assertion"
        )
        # Every remaining `tf.` occurrence must be prose. Classified rather
        # than counted, per the plan's verification note.
        prose = [
            line for line in source.splitlines()
            if "tf." in line and "``" not in line
        ]
        assert prose == [], f"unclassified tf. usage:\n{chr(10).join(prose)}"
