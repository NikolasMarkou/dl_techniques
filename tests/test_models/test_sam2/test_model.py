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
from typing import Any, Dict, List, Tuple
from unittest import mock

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

import dl_techniques.models.sam2.model as model_module
from dl_techniques.models.SAM.SAM1.transformer import TwoWayTransformer
from dl_techniques.models.sam2.hiera import Hiera, hiera_block_specs
from dl_techniques.models.sam2.mask_decoder import SAM2MaskDecoder
from dl_techniques.models.sam2.memory_encoder import SAM2MemoryEncoder
from dl_techniques.models.sam2.model import (
    MEMORY_STRIDE,
    NO_OBJ_SCORE,
    SAM2,
    create_sam2,
)
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

#: Published parameter count of SAM 2.1-L, SOURCED from the upstream repository
#: at the pinned clone `sam2@2b90b9f5`, `README.md:170`, row
#: `sam2.1_hiera_large | 224.4`. The measured figure and the full
#: reconciliation live in `test_hiera_l_total_reconciles_with_the_published_figure`.
#: Quoted to the published precision (~224.4M) and no further -- inventing
#: digits the source does not carry would make the tolerance below meaningless.
PUBLISHED_HIERA_L_PARAMS = 224_400_000

#: Half of the published figure's own quantum. "224.4" is four significant
#: figures, so the true upstream count lies in `224.4M +- 50,000`. This is the
#: TIGHTEST tolerance the source supports, and stating it as the source's
#: rounding rather than as a chosen percentage is what stops it from drifting:
#: the superseded `< 1e-3` relative band was `+-224,400`, which still admitted
#: the 213,888-parameter omission it was written to catch.
PUBLISHED_FIGURE_HALF_QUANTUM = 50_000

#: Built parameter count of `hiera_l` at this HEAD, MEASURED. Asserted EXACTLY,
#: because the published figure -- however tightly its rounding is respected --
#: cannot resolve an omission smaller than 50,000, and the smallest component
#: this plan has already lost once is 64.
MEASURED_HIERA_L_BUILT_PARAMS = 221_155_425

#: Built size of SAM 1's ``TwoWayTransformer`` at the SAM 2 decoder shape
#: (``depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048``), MEASURED by
#: building one and calling it once. It builds its attention and FFN sub-layers
#: LAZILY, so it contributes zero to a `hiera_l` model that is never
#: forward-passed -- and the audit deliberately never forward-passes one.
#:
#: The first pass guessed "roughly 4.2M" here without measuring, which is 0.9M
#: too large and is precisely what made four genuinely missing components look
#: accounted for.
MEASURED_LAZY_TRANSFORMER_PARAMS = 3_291_264


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


def pin_object_score(model: SAM2, value: float) -> None:
    """Force the object-score head to emit ``value`` for every input.

    The head's last ``Dense`` gets a zero kernel and a constant bias, so the
    logit no longer depends on the image, the prompt or the weights upstream
    of it.

    This exists because several mechanisms in this file are gated on the SIGN
    of the object score, and at random initialization that sign is arbitrary.
    A test that drives the real forward path and merely HOPES for a negative
    score is vacuous on roughly half of all seeds -- and passes silently when
    it is.

    :param model: A BUILT model whose decoder predicts object scores.
    :type model: SAM2
    :param value: The logit to pin. Strictly negative means "occluded".
    :type value: float
    """
    last = model.mask_decoder.pred_obj_score_head.layers[-1]
    last.kernel.assign(np.zeros(last.kernel.shape, dtype="float32"))
    last.bias.assign(np.full(last.bias.shape, value, dtype="float32"))


def model_with_default_decoder() -> SAM2:
    """Assemble a ``tiny`` model whose decoder keeps its OWN defaults.

    Every other fixture in this file goes through :meth:`SAM2.from_variant`,
    which hardcodes ``use_multimask_token_for_obj_ptr=True``. That single fact
    made a whole configuration -- the documented direct-construction route at
    the decoder's own signature default -- unreachable from the entire suite,
    which is how a hard crash on it shipped. This builder reaches it: only the
    two settings ``SAM2`` itself REQUIRES (`use_high_res_features`,
    `pred_obj_scores`) are passed, and everything else is left at
    ``SAM2MaskDecoder``'s defaults.

    :return: A BUILT model with ``use_multimask_token_for_obj_ptr=False``.
    :rtype: SAM2
    """
    donor = SAM2.from_variant("tiny")
    table = SAM2.MODEL_VARIANTS["tiny"]
    model = SAM2(
        image_encoder=donor.image_encoder,
        prompt_encoder=donor.prompt_encoder,
        mask_decoder=SAM2MaskDecoder(
            transformer_dim=donor.hidden_dim,
            transformer=TwoWayTransformer(
                depth=table["decoder_depth"],
                embedding_dim=donor.hidden_dim,
                num_heads=table["decoder_num_heads"],
                mlp_dim=table["decoder_mlp_dim"],
            ),
            use_high_res_features=True,
            pred_obj_scores=True,
        ),
        memory_attention=donor.memory_attention,
        memory_encoder=donor.memory_encoder,
        num_maskmem=donor.num_maskmem,
        image_size=donor.image_size,
    )
    model.build(None)
    return model


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
        # The forward arm below is the only BEHAVIOURAL check in this test, and
        # without this pin it is decided by an unseeded sign: at a negative
        # object score D-043 makes both sides uniformly NO_OBJ_SCORE and
        # `max_abs_diff == 0.0` holds no matter what the restored graph does.
        # Pinning POSITIVE keeps the real decoder output on both sides. The
        # per-weight comparison above is unaffected either way; this makes the
        # forward comparison mean what it says.
        pin_object_score(model, 5.0)
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

    def test_the_five_owned_weights_exist_with_their_declared_shapes(
            self) -> None:
        model = tiny_model()
        assert tuple(model.maskmem_tpos_enc.shape) == \
            (model.num_maskmem, 1, 1, TINY_MEM_DIM)
        assert tuple(model.no_mem_embed.shape) == (1, 1, TINY_DIM)
        assert tuple(model.no_mem_pos_enc.shape) == (1, 1, TINY_MEM_DIM)
        assert tuple(model.no_obj_ptr.shape) == (1, TINY_DIM)
        # The SPATIAL no-object embedding -- the second, independent
        # no-object mechanism. Absent from the first pass entirely.
        assert tuple(model.no_obj_embed_spatial.shape) == (1, TINY_MEM_DIM)


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


def owned_weights(model: SAM2) -> Tuple[Any, ...]:
    """The weights ``SAM2`` owns directly, i.e. that belong to no component.

    Interface contract: one home for this list, so a newly added owned weight
    cannot be counted by one audit test and silently skipped by another --
    which is how ``no_obj_embed_spatial`` could have gone missing a second
    time.

    :param model: A BUILT model.
    :type model: SAM2
    :return: The five owned weight tensors.
    :rtype: Tuple[Any, ...]
    """
    return (
        model.maskmem_tpos_enc,
        model.no_mem_embed,
        model.no_mem_pos_enc,
        model.no_obj_ptr,
        model.no_obj_embed_spatial,
    )


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
            + large.mem_dim                     # no_obj_embed_spatial
        )
        owned = sum(
            int(np.prod(w.shape)) for w in owned_weights(large))
        assert owned == expected

    def test_the_object_pointer_projections_match_their_closed_forms(
            self, large: SAM2) -> None:
        """``obj_ptr_proj`` and ``obj_ptr_tpos_proj``, by closed form.

        Both were ABSENT from the first pass. They are asserted by closed form
        rather than by literal so the numbers derive from the widths:

        * ``obj_ptr_proj`` is ``MLP(256, 256, 256, 3)``: three
          ``256 -> 256`` layers, ``3 * (256 * 256 + 256) = 197,376``.
        * ``obj_ptr_tpos_proj`` is ``Linear(hidden_dim, mem_dim)``:
          ``256 * 64 + 64 = 16,448``.
        """
        dim, mem = large.hidden_dim, large.mem_dim
        assert large.obj_ptr_proj.count_params() == 3 * (dim * dim + dim)
        assert large.obj_ptr_proj.count_params() == 197_376
        assert large.obj_ptr_tpos_proj.count_params() == dim * mem + mem
        assert large.obj_ptr_tpos_proj.count_params() == 16_448

    def test_total_equals_the_sum_of_its_components(self, large: SAM2) -> None:
        """No component is double-counted and none is missing."""
        components = (
            large.image_encoder.count_params()
            + large.prompt_encoder.count_params()
            + large.mask_decoder.count_params()
            + large.memory_attention.count_params()
            + large.memory_encoder.count_params()
            + large.obj_ptr_proj.count_params()
            + large.obj_ptr_tpos_proj.count_params()
        )
        owned = sum(int(np.prod(w.shape)) for w in owned_weights(large))
        assert large.count_params() == components + owned

    def test_hiera_l_total_reconciles_with_the_published_figure(
            self, large: SAM2) -> None:
        """The ONE home of the measured total, reconciled to the last parameter.

        MEASURED at this HEAD: **221,155,425** built parameters. The model is
        never forward-passed, so SAM 1's ``TwoWayTransformer`` -- which builds
        its attention and FFN sub-layers lazily on first call -- contributes
        nothing yet. Its built size at ``depth=2, dim=256, heads=8, mlp=2048``
        was MEASURED separately (by building one and calling it once) as
        **3,291,264**, giving::

            221,155,425 + 3,291,264 = 224,446,689   vs published ~224.4M

        **Two corrections are pinned here deliberately.** The first pass
        recorded 220,941,537 and explained the residual as "roughly 4.2M
        parameters do not exist yet" -- a number nobody had measured, and 0.9M
        too large. That wrong figure is what made the four MISSING object-
        pointer components (197,376 + 16,448 + 64 = 213,888) look accounted
        for. With them built and the transformer's real size measured, the
        reconciliation closes.

        **Two assertions, because neither alone is enough.**

        1. The built total EXACTLY equals ``MEASURED_HIERA_L_BUILT_PARAMS``.
           This is the arm with resolution: it catches a missing 64-parameter
           weight.
        2. The reconciliation lands inside the published figure's OWN rounding
           quantum -- ``|reconciled - 224.4M| < 50,000``, i.e. the reconciled
           number rounds to "224.4M" at the precision the source quotes. This
           is the arm that can catch a systematically wrong total that arm 1
           would happily re-baseline to.

        **Two superseded assertions are recorded here so neither comes back.**
        The first pass asserted a +-25M band, which would pass with the entire
        memory encoder (~1.4M) deleted. The completion-fix round replaced it
        with ``relative < 1e-3``, i.e. +-224,400 -- and the omission it was
        written for was 213,888, so the PRE-fix total (224,232,801) passed it
        at 7.45e-4. A percentage tolerance chosen for how it reads is not a
        tolerance; this one is derived from the source's own precision.
        """
        total = large.count_params()
        reconciled = total + MEASURED_LAZY_TRANSFORMER_PARAMS

        assert large.mask_decoder.transformer.weights == [], (
            "the two-way transformer is built LAZILY; if it now holds "
            "variables it is already inside `total` and adding "
            "MEASURED_LAZY_TRANSFORMER_PARAMS double-counts it"
        )
        assert total == MEASURED_HIERA_L_BUILT_PARAMS, (
            f"hiera_l built {total:,} parameters, not the measured "
            f"{MEASURED_HIERA_L_BUILT_PARAMS:,} (difference "
            f"{total - MEASURED_HIERA_L_BUILT_PARAMS:+,}). Do NOT re-baseline "
            f"this number to make the test pass -- reconcile the difference "
            f"against the per-component closed forms above first"
        )
        deviation = abs(reconciled - PUBLISHED_HIERA_L_PARAMS)
        assert deviation < PUBLISHED_FIGURE_HALF_QUANTUM, (
            f"hiera_l reconciliation is off by {deviation:,}: built "
            f"{total:,} + lazy {MEASURED_LAZY_TRANSFORMER_PARAMS:,} = "
            f"{reconciled:,}, which does not round to the published "
            f"{PUBLISHED_HIERA_L_PARAMS:,} at the source's own precision. A "
            f"component is missing or mis-sized -- do NOT widen this into a "
            f"band"
        )

    def test_the_reconciliation_rejects_an_omission_at_its_own_scale(
            self, large: SAM2) -> None:
        """Probed at 213,888 parameters, not at 1.4M.

        The superseded ``relative < 1e-3`` tolerance was demonstrated
        "non-vacuous" by deleting the whole memory encoder -- an omission ten
        times larger than the one that actually happened, and one the tolerance
        was never at risk from. This arm probes at the real scale: the three
        object-pointer components that WERE missing (``obj_ptr_proj`` 197,376 +
        ``obj_ptr_tpos_proj`` 16,448 + ``no_obj_embed_spatial`` 64 = 213,888).

        It also pins, by execution, that the superseded tolerance would have
        accepted that omission -- so the reason for the change cannot be
        re-litigated from prose alone.
        """
        omission = (
            large.obj_ptr_proj.count_params()
            + large.obj_ptr_tpos_proj.count_params()
            + int(np.prod(large.no_obj_embed_spatial.shape))
        )
        assert omission == 213_888, (
            f"the object-pointer components now total {omission:,}, not the "
            f"213,888 that went missing; re-derive this probe"
        )

        crippled = large.count_params() - omission
        reconciled = crippled + MEASURED_LAZY_TRANSFORMER_PARAMS

        assert abs(reconciled - PUBLISHED_HIERA_L_PARAMS) >= \
            PUBLISHED_FIGURE_HALF_QUANTUM, (
            f"dropping the {omission:,} object-pointer parameters still "
            f"reconciles to the published figure; the tolerance is blind to "
            f"the very defect it was written for"
        )
        assert crippled != MEASURED_HIERA_L_BUILT_PARAMS

        superseded_relative = abs(
            reconciled - PUBLISHED_HIERA_L_PARAMS) / PUBLISHED_HIERA_L_PARAMS
        assert superseded_relative < 1e-3, (
            f"the superseded relative-1e-3 tolerance now REJECTS this "
            f"omission ({superseded_relative:.2e}), so the stated reason for "
            f"replacing it is stale and this test should be re-derived"
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
            ["git", "diff", "--stat", "--", "src/dl_techniques/models/SAM/SAM1/"],
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

    The object score is pinned POSITIVE (D-043). ``NO_OBJ_SCORE`` suppression
    replaces the mask logits with a constant wherever the score is negative,
    and ``ops.where`` passes no gradient through the replaced branch -- so on a
    randomly initialized model the mask path would be starved for roughly half
    of all seeds and this instrument's partition would be a coin flip. That is
    upstream's behaviour too, not an artifact of this port: real training
    supervises the object score as well, which this single-key loss does not.

    :return: The compiled model.
    :rtype: SAM2
    """
    model = tiny_model()
    pin_object_score(model, 5.0)
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

        The object score is pinned POSITIVE (D-043): under ``NO_OBJ_SCORE``
        suppression a negative score makes ``low_res_logits`` a constant with
        no gradient path at all, so BOTH arms of this two-sided guard --
        including the one that asserts the gradient MUST appear -- would be
        decided by an unseeded sign rather than by the ``stop_gradient``
        boundary they exist to measure.
        """
        pin_object_score(model, 5.0)
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
        """The control for the control: one frame IS differentiable.

        Object score pinned positive for the reason recorded on
        ``_gradient_through_the_stream``.
        """
        model = tiny_model()
        pin_object_score(model, 5.0)
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
        """A VALUE assertion, not a branch-coverage one.

        Object score pinned positive (D-043): a suppressed mask is a constant
        that no upstream perturbation can move, so without the pin this
        assertion is decided by an unseeded sign.
        """
        model = tiny_model()
        pin_object_score(model, 5.0)
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

    def test_slots_are_expanded_per_token_and_the_pointer_tail_is_encoded(
            self) -> None:
        """The gather is per TOKEN, and the tail is NOT zeros.

        **This test previously asserted the opposite.** It pinned
        ``max(abs(tail)) == 0.0`` -- i.e. it encoded a MISSING mechanism as
        intended behaviour, and would have gone red on the correct code. The
        object-pointer temporal encoding (``add_tpos_enc_to_obj_ptrs: true``)
        is the only thing that distinguishes a pointer from frame ``t-1`` from
        one from frame ``t-15``: the rotary table in memory attention is
        spatial-only and broadcast identically across memory tokens, and
        ``maskmem_tpos_enc`` is indexed by spatial slot and never reaches the
        tail. With zeros there, every object pointer is temporally identical.
        """
        model = tiny_model()
        # A non-zero projection: at the zero-initialized default the tail is
        # legitimately zero and this test could not tell a live encoding from
        # the absent one it used to pin.
        model.obj_ptr_tpos_proj.kernel.assign(
            np.full(model.obj_ptr_tpos_proj.kernel.shape, 0.05,
                    dtype="float32"))

        model.stream_reset()
        model.stream_step(images(1), frame_idx=0, is_conditioning=True)
        model.stream_step(images(1, seed=1), frame_idx=1)
        readout = model.memory_bank.read(2)
        encoding = model._temporal_embedding(readout, readout.memory)

        assert tuple(encoding.shape) == (1, int(readout.memory.shape[1]),
                                         TINY_MEM_DIM)
        assert readout.num_obj_ptr_tokens > 0, (
            "no object pointers reached the readout, so the tail assertion "
            "below is vacuous"
        )
        tail = ops.convert_to_numpy(
            encoding[:, -readout.num_obj_ptr_tokens:, :])
        assert float(np.max(np.abs(tail))) > 1e-6, (
            "the object-pointer tail of the temporal encoding is all zeros -- "
            "`add_tpos_enc_to_obj_ptrs` is not wired, and every object "
            "pointer is temporally indistinguishable"
        )

    def test_pointers_from_different_frames_get_different_encodings(
            self) -> None:
        """The tail must carry the temporal DIFFERENCE, not a constant.

        A constant non-zero tail would pass the test above while leaving every
        pointer identical, which is the whole defect. Two pointers at different
        temporal distances must therefore differ, and the sub-tokens of ONE
        pointer must agree (the bank repeat-interleaves each pointer's
        difference across its ``hidden_dim // mem_dim`` sub-tokens).
        """
        model = tiny_model()
        model.obj_ptr_tpos_proj.kernel.assign(
            np.full(model.obj_ptr_tpos_proj.kernel.shape, 0.05,
                    dtype="float32"))

        model.stream_reset()
        model.stream_step(images(1), frame_idx=0, is_conditioning=True)
        model.stream_step(images(1, seed=1), frame_idx=1)
        model.stream_step(images(1, seed=2), frame_idx=2)
        readout = model.memory_bank.read(3)

        per_pointer = model.memory_bank.tokens_per_pointer
        assert len(readout.obj_ptr_frames) >= 2, (
            f"only {len(readout.obj_ptr_frames)} pointer(s) in the readout -- "
            f"this test needs at least two at DIFFERENT distances"
        )
        assert len(set(readout.obj_ptr_tpos)) >= 2, (
            f"every pointer reports the same temporal difference "
            f"{readout.obj_ptr_tpos}, so this test cannot discriminate"
        )

        tail = ops.convert_to_numpy(ops.squeeze(
            model._object_pointer_temporal_encoding(readout), axis=0))
        assert tail.shape == (readout.num_obj_ptr_tokens, TINY_MEM_DIM)

        first = tail[0]
        second = tail[per_pointer]
        assert float(np.max(np.abs(first - second))) > 1e-6, (
            "two pointers at different temporal distances received the same "
            "encoding -- the tail is a constant, not a temporal signal"
        )
        # Within ONE pointer, every sub-token shares the pointer's distance.
        for offset in range(1, per_pointer):
            np.testing.assert_allclose(tail[offset], first, atol=1e-6)

    def test_changing_a_slot_row_changes_the_conditioned_output(self) -> None:
        """The discriminating observation: a DEAD table would be invisible.

        Every other temporal mechanism in the stack is off: the rotary table is
        spatial-only and identical across memory frames (``repeat_k``), so if
        this weight did not reach the output there would be nothing left to
        tell frame ``t-1`` from frame ``t-6``.

        Object score pinned positive (D-043): suppression would otherwise make
        both arms the same constant, AND the perturbation itself can flip the
        score's sign, so the observed difference would not be attributable to
        the table.
        """
        model = tiny_model()
        pin_object_score(model, 5.0)

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
        """Two rows must not be interchangeable, or the slot index is dead.

        Object score pinned positive, for the reason on the previous test.
        """
        model = tiny_model()
        pin_object_score(model, 5.0)

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
    """``no_obj_ptr`` is interpolated towards by the object score.

    **The saturated-score tests below are BLIND to the blend formula.** They
    use object-score logits of exactly ``+-30``, i.e. ``lambda`` in ``{0, 1}``
    -- the only two points at which the reference expression

        ``ptr' = lambda * ptr (only if fixed_no_obj_ptr); ptr' += (1-lambda)*no_obj``

    and the symmetric-looking ``lambda * ptr + (1 - lambda) * no_obj`` agree.
    The whole first pass shipped the wrong one of the two with all three of
    these tests green. They are KEPT (saturation is still worth pinning) and
    ``TestObjectPointerBlendAtIntermediateLambda`` below carries the actual
    discrimination, at ``score = 0`` and other unsaturated values.
    """

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


class TestObjectPointerBlendAtIntermediateLambda:
    """The blend formula, measured where the candidate formulas DISAGREE.

    Reference expression (pinned clone, ``sam2/modeling/sam2_base.py:396-403``,
    read directly rather than quoted)::

        lambda = object_score_logits.sigmoid() if soft_no_obj_ptr
                 else is_obj_appearing.float()
        if fixed_no_obj_ptr:
            obj_ptr = lambda * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda) * no_obj_ptr

    The ``lambda * ptr`` multiply is CONDITIONAL; the ``(1 - lambda) * no_obj``
    add is not. The symmetric ``lambda * ptr + (1 - lambda) * no_obj`` is a
    different function everywhere except ``lambda in {0, 1}``.
    """

    #: A pointer and a no-object vector that make every candidate's answer a
    #: distinct, hand-checkable number.
    POINTER_VALUE = 1.0
    NO_OBJ_VALUE = 9.0

    def make(self, **overrides: Any) -> SAM2:
        """Build a tiny model with a known ``no_obj_ptr``."""
        model = tiny_model(**overrides)
        model.no_obj_ptr.assign(
            np.full((1, TINY_DIM), self.NO_OBJ_VALUE, dtype="float32"))
        return model

    def test_soft_blend_at_score_zero_matches_the_hand_derived_value(
            self) -> None:
        """At ``score = 0`` (``lambda = 0.5``) the two candidates differ by 0.5.

        Hand-derived with ``ptr = 1.0``, ``no_obj = 9.0``,
        ``soft_no_obj_ptr=True`` and ``fixed_no_obj_ptr=True``:

        * reference: ``0.5 * 1.0 + 0.5 * 9.0 = 5.0``
        * without the conditional multiply (i.e. ``fixed_no_obj_ptr`` ignored):
          ``1.0 + 0.5 * 9.0 = 5.5``

        With ``fixed_no_obj_ptr=True`` the reference and the symmetric form
        coincide, so this arm pins the VALUE and the next arm -- which turns
        the flag off -- carries the discrimination between the two formulas.
        """
        model = self.make(soft_no_obj_ptr=True, fixed_no_obj_ptr=True)
        blended = ops.convert_to_numpy(model._blend_object_pointer(
            np.full((BATCH, TINY_DIM), self.POINTER_VALUE, dtype="float32"),
            np.zeros((BATCH, 1), dtype="float32")))
        np.testing.assert_allclose(blended, 5.0, atol=1e-5)

    def test_without_fixed_no_obj_ptr_the_pointer_is_NOT_scaled(self) -> None:
        """``fixed_no_obj_ptr=False`` at ``lambda = 0.5``: ``1.0 + 4.5 = 5.5``.

        This is THE discriminating measurement of the whole class. The
        symmetric formula the first pass shipped returns ``0.5 * 1.0 + 4.5 =
        5.0`` here, and at every saturated score it agrees with the reference,
        which is why three green tests never saw it.
        """
        model = self.make(soft_no_obj_ptr=True, fixed_no_obj_ptr=False)
        blended = ops.convert_to_numpy(model._blend_object_pointer(
            np.full((BATCH, TINY_DIM), self.POINTER_VALUE, dtype="float32"),
            np.zeros((BATCH, 1), dtype="float32")))

        reference = self.POINTER_VALUE + 0.5 * self.NO_OBJ_VALUE      # 5.5
        symmetric = 0.5 * self.POINTER_VALUE + 0.5 * self.NO_OBJ_VALUE  # 5.0
        assert abs(reference - symmetric) > 1e-2, (
            "the two candidate formulas coincide at these values -- this "
            "probe cannot discriminate and must be re-chosen"
        )
        np.testing.assert_allclose(blended, reference, atol=1e-5)
        assert abs(float(blended.reshape(-1)[0]) - symmetric) > 1e-2, (
            f"the blend returned {blended.reshape(-1)[0]}, which matches the "
            f"symmetric `lambda*ptr + (1-lambda)*no_obj` ({symmetric}) rather "
            f"than the reference ({reference})"
        )

    @pytest.mark.parametrize("score,lam", [(-1.0, 0.26894142),
                                           (0.5, 0.62245933),
                                           (2.0, 0.88079708)])
    def test_the_formula_holds_across_unsaturated_scores(
            self, score: float, lam: float) -> None:
        """Three more intermediate points, none of them a coincidence point."""
        model = self.make(soft_no_obj_ptr=True, fixed_no_obj_ptr=False)
        blended = ops.convert_to_numpy(model._blend_object_pointer(
            np.full((BATCH, TINY_DIM), self.POINTER_VALUE, dtype="float32"),
            np.full((BATCH, 1), score, dtype="float32")))

        assert lam == pytest.approx(1.0 / (1.0 + np.exp(-score)), abs=1e-6)
        expected = self.POINTER_VALUE + (1.0 - lam) * self.NO_OBJ_VALUE
        symmetric = lam * self.POINTER_VALUE + (1.0 - lam) * self.NO_OBJ_VALUE
        assert abs(expected - symmetric) > 1e-2, (
            f"at score {score} the two formulas are only "
            f"{abs(expected - symmetric)} apart"
        )
        np.testing.assert_allclose(blended, expected, atol=1e-5)

    def test_the_shipped_variant_defaults_match_the_shipped_config(
            self) -> None:
        """``fixed_no_obj_ptr=True``, ``soft_no_obj_ptr=False``.

        The shipped ``sam2.1_hiera_l.yaml`` sets ``fixed_no_obj_ptr: true`` and
        leaves ``soft_no_obj_ptr`` unset (reference default ``False``). This
        port has no YAML layer, so the constructor defaults ARE the shipped
        config. The first pass shipped both inverted.
        """
        for variant in SAM2.MODEL_VARIANTS:
            model = SAM2.from_variant(variant)
            assert model.fixed_no_obj_ptr is True, (
                f"variant '{variant}' ships fixed_no_obj_ptr=False; the "
                f"shipped config sets it true"
            )
            assert model.soft_no_obj_ptr is False, (
                f"variant '{variant}' ships soft_no_obj_ptr=True; the shipped "
                f"config leaves it unset and the reference default is False"
            )

    def test_the_hard_threshold_is_used_at_the_shipped_default(self) -> None:
        """``soft_no_obj_ptr=False`` means a STEP at zero, not a sigmoid.

        Measured at ``score = +0.5``, where a sigmoid gives ``lambda = 0.622``
        and the hard threshold gives ``1.0``: the two answers are 3.4 apart.
        """
        model = self.make()
        assert model.soft_no_obj_ptr is False
        blended = ops.convert_to_numpy(model._blend_object_pointer(
            np.full((1, TINY_DIM), self.POINTER_VALUE, dtype="float32"),
            np.full((1, 1), 0.5, dtype="float32")))

        hard = self.POINTER_VALUE                       # lambda == 1.0
        soft = self.POINTER_VALUE * 0.62245933 + \
            (1.0 - 0.62245933) * self.NO_OBJ_VALUE      # 4.02
        assert abs(hard - soft) > 1e-2
        np.testing.assert_allclose(blended, hard, atol=1e-5)


class TestBestIouSelection:
    """The frame's memory and pointer come from the model's OWN best mask.

    Under ``multimask_output=True`` the decoder has already sliced away the
    single-mask token, so ``[:, 0]`` is *multimask token 1* -- one of three,
    chosen by position. The reference gathers both the mask and the output
    token by ``argmax(ious)``. At ``M == 1`` the two are identical, which is
    why the single-mask path cannot see the difference.
    """

    def test_the_helper_gathers_each_row_by_its_own_argmax(self) -> None:
        """Per-batch-element, and NOT a single global argmax.

        The two rows deliberately select DIFFERENT indices: a batch that
        agreed would be satisfied by a global argmax too.
        """
        tensor = ops.convert_to_tensor(np.asarray([
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
        ], dtype="float32"))
        iou = ops.convert_to_tensor(
            np.asarray([[0.1, 0.9, 0.2], [0.7, 0.2, 0.1]], dtype="float32"))

        picked = ops.convert_to_numpy(
            model_module._select_best_by_iou(tensor, iou))
        assert picked.shape == (2, 1, 2)
        np.testing.assert_allclose(picked[0, 0], [2.0, 2.0])
        np.testing.assert_allclose(picked[1, 0], [4.0, 4.0])

    def test_the_helper_is_a_no_op_on_the_single_mask_path(self) -> None:
        """``M == 1`` must reproduce ``[:, 0:1]`` exactly.

        This is the arm that explains why the defect was invisible: every
        single-mask test in this file would pass under either implementation.
        """
        tensor = ops.convert_to_tensor(
            np.arange(2 * 1 * 3, dtype="float32").reshape(2, 1, 3))
        iou = ops.convert_to_tensor(np.asarray([[0.4], [0.6]], dtype="float32"))
        picked = ops.convert_to_numpy(
            model_module._select_best_by_iou(tensor, iou))
        np.testing.assert_allclose(
            picked, ops.convert_to_numpy(tensor)[:, 0:1])

    def test_it_ranks_by_iou_and_not_by_position(self) -> None:
        """A rank-reversing IoU vector must move the selection.

        A helper that ignored ``iou`` entirely and returned index 0 passes the
        no-op arm above; only a reversal separates them.
        """
        tensor = ops.convert_to_tensor(np.asarray(
            [[[1.0], [2.0], [3.0]]], dtype="float32"))
        first = ops.convert_to_numpy(model_module._select_best_by_iou(
            tensor,
            ops.convert_to_tensor(np.asarray([[0.9, 0.1, 0.0]], "float32"))))
        last = ops.convert_to_numpy(model_module._select_best_by_iou(
            tensor,
            ops.convert_to_tensor(np.asarray([[0.0, 0.1, 0.9]], "float32"))))
        np.testing.assert_allclose(first[0, 0], [1.0])
        np.testing.assert_allclose(last[0, 0], [3.0])

    def test_the_multimask_pointer_follows_the_iou_head(self) -> None:
        """End-to-end: the decoded pointer changes when the IoU ranking does.

        The previous version of this test carried this docstring and asserted
        only two SHAPES. It never read ``object_pointer``, never moved the IoU
        ranking and never compared anything, so a ``_decode`` that had reverted
        to ``pointer_tokens[:, 0, :]`` passed it. This version does the
        measurement the name promises.

        The IoU head's last bias is driven so that the winning multimask token
        moves from slice position 0 to slice position 2; nothing else changes,
        and in particular the mask tokens themselves are untouched. So any
        movement in ``object_pointer`` is attributable to the selection alone.

        The object score is pinned POSITIVE first. Under the shipped
        ``fixed_no_obj_ptr=True`` a negative score multiplies the selected
        pointer by zero, which would leave ``object_pointer`` equal to
        ``no_obj_ptr`` for BOTH rankings -- a vacuum this test would otherwise
        fall into on roughly half of all seeds.
        """
        model = tiny_model()
        pin_object_score(model, 5.0)
        inputs = {"image": images(batch=1)}

        multi = model(inputs, multimask_output=True)
        assert int(multi["iou_predictions"].shape[1]) == 3, (
            "the multimask path must offer three candidates, or a selection "
            "test over it cannot discriminate"
        )

        head = model.mask_decoder.iou_prediction_head.layers[-1]
        original = ops.convert_to_numpy(head.bias)

        def pointer_when_winner_is(slice_position: int) -> np.ndarray:
            """Force the multimask slice index that wins, return the pointer."""
            bias = np.zeros(original.shape, dtype="float32")
            # `iou_prediction_head` emits one score per mask token; the
            # multimask path drops token 0, so slice position `p` is token
            # `p + 1`.
            bias[slice_position + 1] = 100.0
            head.bias.assign(bias)
            out = model(inputs, multimask_output=True)
            winner = int(np.argmax(
                ops.convert_to_numpy(out["iou_predictions"])[0]))
            assert winner == slice_position, (
                f"forcing token {slice_position + 1} did not make slice "
                f"position {slice_position} the argmax (got {winner}); the "
                f"probe no longer controls the ranking"
            )
            return ops.convert_to_numpy(out["object_pointer"])[0]

        try:
            first = pointer_when_winner_is(0)
            third = pointer_when_winner_is(2)
        finally:
            head.bias.assign(original)

        moved = float(np.max(np.abs(first - third)))
        assert moved > 1e-4, (
            f"reversing the IoU ranking moved the decoded object pointer by "
            f"{moved} -- `_decode` is not gathering the pointer token by "
            f"argmax(iou), it is taking a fixed position"
        )

    def test_the_pointer_gather_is_skipped_when_there_is_one_pointer_token(
            self) -> None:
        """The decoder's OWN default must not crash on the multimask path.

        ``SAM2MaskDecoder`` defaults ``use_multimask_token_for_obj_ptr=False``,
        at which the multimask path returns ``iou`` with M=3 and
        ``object_pointer`` with M=1 (D-023 -- the pointer deliberately comes
        from the SINGLE-mask token). Gathering the pointer with the IoU argmax
        then indexes a length-1 axis with a 1 or a 2.

        This combination is unreachable through ``from_variant`` (which
        hardcodes the flag True), which is why the defect shipped with a green
        suite. The reference guards it at ``sam2_base.py:387``
        (``if sam_output_tokens.size(1) > 1``) and so does this port.

        **Two things about this probe were MEASURED, and neither was obvious.**

        1. The IoU ranking must be FORCED. The first version let it fall where
           a random initialization put it, ran the guard-removed mutation, and
           stayed GREEN -- ``argmax`` landed on 0, which is in range even for a
           length-1 axis. At batch 1 over three candidates that is one time in
           three.
        2. The symptom is DEVICE-DEPENDENT, so this asserts a VALUE and not an
           exception. On CPU the guard-removed mutation raises
           ``InvalidArgumentError: indices[0,0] = 2 is not in [0, 1)
           [Op:GatherV2]``. On GPU it does NOT raise -- TensorFlow's gather
           clamps out-of-range indices there and returns zeros, so the model
           silently builds its object pointer from a zero token. A
           ``pytest.raises`` guard would therefore have been green on the very
           device this suite runs on.

        The assertion is the guard's own semantics: with exactly one pointer
        token there is nothing to choose, so the decoded pointer must be
        IDENTICAL under two opposite IoU rankings.

        The object score is pinned POSITIVE, and that too was measured rather
        than assumed: unpinned, a negative score plus ``fixed_no_obj_ptr=True``
        collapses ``object_pointer`` to the zero-initialized ``no_obj_ptr``,
        at which an out-of-range gather returning zeros is INDISTINGUISHABLE
        from a correct one. The vacuity assertion below is what surfaced that.
        """
        model = model_with_default_decoder()
        pin_object_score(model, 5.0)
        assert model.mask_decoder.use_multimask_token_for_obj_ptr is False, (
            "the decoder is no longer at its own default, so this test no "
            "longer reaches the mask-axis mismatch it exists for"
        )

        head = model.mask_decoder.iou_prediction_head.layers[-1]
        inputs = {"image": images(batch=1)}

        def pointer_when_winner_is(slice_position: int) -> np.ndarray:
            """Force the winning multimask slice, return the object pointer."""
            bias = np.zeros(head.bias.shape, dtype="float32")
            # Mask token `p + 1` is multimask slice position `p`.
            bias[slice_position + 1] = 100.0
            head.bias.assign(bias)
            out = model(inputs, multimask_output=True)
            winner = int(np.argmax(
                ops.convert_to_numpy(out["iou_predictions"])[0]))
            assert winner == slice_position, (
                f"the IoU argmax is at slice position {winner}, not the "
                f"forced {slice_position}; a gather at position 0 is in range "
                f"even for a length-1 pointer axis, so this probe would be "
                f"blind to the defect it exists for"
            )
            assert int(out["iou_predictions"].shape[1]) == 3
            assert tuple(out["object_pointer"].shape) == (1, model.hidden_dim)
            return ops.convert_to_numpy(out["object_pointer"])[0]

        at_zero = pointer_when_winner_is(0)
        at_two = pointer_when_winner_is(2)

        assert np.isfinite(at_zero).all() and np.isfinite(at_two).all()
        assert float(np.max(np.abs(at_zero))) > 0.0, (
            "the pointer at the in-range ranking is identically zero, so an "
            "out-of-range gather returning zeros would be indistinguishable "
            "from it and this comparison proves nothing"
        )
        np.testing.assert_allclose(at_zero, at_two, atol=1e-6, err_msg=(
            "the decoded object pointer moved when the IoU ranking moved, "
            "even though the decoder emits exactly ONE pointer token -- the "
            "gather is not being skipped"
        ))


class TestAbsentObjectMaskSuppression:
    """``NO_OBJ_SCORE``: an absent object's mask is ERASED, not merely flagged.

    This is the third member of the ``pred_obj_scores`` family, and the one
    that was missing while the other two shipped. ``no_obj_ptr`` marks the
    pointer stream and ``no_obj_embed_spatial`` marks the spatial stream; this
    mechanism replaces the mask logits themselves. Without it the port stores
    an occluded frame's REAL mask together with an occlusion flag -- a
    contradiction the reference never writes (``sam2_base.py:358-368``).

    Every arm here pins the object score rather than hoping for a sign.
    """

    def test_a_negative_object_score_erases_every_mask_logit(self) -> None:
        """The whole mask goes to the sentinel, on the real forward path."""
        model = tiny_model()
        pin_object_score(model, -5.0)

        logits = ops.convert_to_numpy(
            model({"image": images()}, multimask_output=True)["low_res_logits"])

        assert np.all(logits == np.float32(NO_OBJ_SCORE)), (
            f"a negative object score left the mask logits in "
            f"[{logits.min()}, {logits.max()}] instead of the sentinel "
            f"{NO_OBJ_SCORE}; the memory bank will store the object's real "
            f"mask on an occluded frame"
        )

    def test_a_positive_object_score_leaves_the_mask_untouched(self) -> None:
        """The negative arm alone is passed by a transform that always fires.

        Also asserts the suppression is STATELESS: pinning the score negative
        in between must not change what the positive score returns.
        """
        model = tiny_model()
        inputs = {"image": images()}

        pin_object_score(model, 5.0)
        visible = ops.convert_to_numpy(
            model(inputs, multimask_output=True)["low_res_logits"])
        pin_object_score(model, -5.0)
        _ = model(inputs, multimask_output=True)
        pin_object_score(model, 5.0)
        again = ops.convert_to_numpy(
            model(inputs, multimask_output=True)["low_res_logits"])

        assert np.all(visible != np.float32(NO_OBJ_SCORE))
        assert float(visible.std()) > 1e-6, (
            "the visible mask is constant, so 'untouched' is indistinguishable "
            "from 'suppressed to some other constant' here"
        )
        np.testing.assert_allclose(visible, again, atol=1e-6)

    def test_the_suppression_is_per_batch_element(self) -> None:
        """A whole-batch reduction would satisfy both arms above.

        Driven through the method rather than the model because the pinning
        helper makes the score CONSTANT across the batch by construction; a
        mixed batch is only reachable here.
        """
        model = tiny_model()
        logits = np.arange(2 * 3 * 2 * 2, dtype="float32").reshape(2, 3, 2, 2)
        out = ops.convert_to_numpy(model._suppress_absent_object(
            ops.convert_to_tensor(logits),
            ops.convert_to_tensor(np.asarray([[-5.0], [5.0]], "float32"))))

        assert np.all(out[0] == np.float32(NO_OBJ_SCORE))
        np.testing.assert_array_equal(out[1], logits[1])

    def test_the_suppressed_mask_is_what_the_memory_encoder_sees(self) -> None:
        """The wiring, end to end: the memory ACTUALLY STORED is the erased one.

        ``_store_memory`` reads ``outputs['low_res_logits']``, so the ordering
        claim -- suppression BEFORE the best-IoU gather and before the memory
        encoder -- is only true if the suppression happens inside ``_decode``.
        This drives ``stream_step`` (the real path) and reconstructs, from the
        SAME pixel features, the memory that a uniformly-sentinel mask would
        give. The two must agree to the parameter.
        """
        model = tiny_model()
        pin_object_score(model, -5.0)
        image = images(batch=1)

        model.stream_reset()
        outputs = model.stream_step(image, frame_idx=0, is_conditioning=True)
        stored = ops.convert_to_numpy(model.memory_bank.cond_frames[0].features)

        assert np.all(
            ops.convert_to_numpy(outputs["low_res_logits"])
            == np.float32(NO_OBJ_SCORE)), (
            "the streaming path returned an unsuppressed mask, so this "
            "reconstruction would be comparing two different things"
        )

        features = model.image_encoder(image, training=False)["vision_features"]
        sentinel = ops.convert_to_tensor(np.full(
            (1, model.image_size, model.image_size, 1), NO_OBJ_SCORE,
            dtype="float32"))
        memory, _ = model.memory_encoder([features, sentinel])
        expected = ops.convert_to_numpy(model._mark_occlusion(
            memory, np.full((1, 1), -5.0, dtype="float32")))

        # The bank stores memory FLATTENED over the spatial grid
        # (`(B, h, w, mem_dim)` -> `(B, h * w, mem_dim)`), so the
        # reconstruction is reshaped to the bank's layout rather than the
        # comparison being loosened. Asserted, so a change in that layout is a
        # loud failure here and not a silently reshaped one.
        assert stored.shape == (1, TINY_GRID * TINY_GRID, TINY_MEM_DIM)
        np.testing.assert_allclose(
            stored, expected.reshape(stored.shape), atol=1e-4)


class TestSpatialNoObjectEmbedding:
    """``no_obj_embed_spatial``: the second, independent no-object mechanism."""

    def test_it_marks_an_occluded_frame_and_leaves_a_visible_one_alone(
            self) -> None:
        """``(1 - is_obj_appearing) * embedding``, added to the memory.

        Two arms, because either alone is passable by a wrong implementation:
        a negative object score must ADD the embedding, and a positive one must
        add exactly nothing.
        """
        model = tiny_model()
        model.no_obj_embed_spatial.assign(
            np.full((1, TINY_MEM_DIM), 3.0, dtype="float32"))
        memory = np.zeros((BATCH, TINY_GRID, TINY_GRID, TINY_MEM_DIM),
                          dtype="float32")

        occluded = ops.convert_to_numpy(model._mark_occlusion(
            memory, np.full((BATCH, 1), -5.0, dtype="float32")))
        visible = ops.convert_to_numpy(model._mark_occlusion(
            memory, np.full((BATCH, 1), 5.0, dtype="float32")))

        np.testing.assert_allclose(occluded, 3.0, atol=1e-6)
        np.testing.assert_allclose(visible, 0.0, atol=1e-6)

    def test_the_mark_is_per_batch_element(self) -> None:
        """A uniform batch would be satisfied by a global reduction."""
        model = tiny_model()
        model.no_obj_embed_spatial.assign(
            np.full((1, TINY_MEM_DIM), 3.0, dtype="float32"))
        marked = ops.convert_to_numpy(model._mark_occlusion(
            np.zeros((2, TINY_GRID, TINY_GRID, TINY_MEM_DIM), dtype="float32"),
            np.asarray([[-5.0], [5.0]], dtype="float32")))
        np.testing.assert_allclose(marked[0], 3.0, atol=1e-6)
        np.testing.assert_allclose(marked[1], 0.0, atol=1e-6)

    def test_it_reaches_the_memory_actually_stored_in_the_bank(self) -> None:
        """The wiring, not just the method: ``_store_memory`` must apply it.

        ``_store_memory`` is driven DIRECTLY with a pinned negative object
        score rather than through ``stream_step``. Going through the model
        would make the observation depend on the sign an unseeded, randomly
        initialized object-score head happens to emit -- i.e. the test would
        silently become vacuous (and pass) whenever that sign came out
        positive. (``pin_object_score`` now exists for exactly that problem;
        the direct drive is kept here because the quantity measured is a
        DIFFERENCE between two weight values, which the mask content cancels
        out of entirely.)

        The hand-built mask is deliberately NOT constant. An earlier version
        passed all-zero logits, which is (up to a shift) the state
        ``NO_OBJ_SCORE`` suppression itself produces -- so this test read as
        evidence about a mechanism it never exercised. The suppression has its
        own class, ``TestAbsentObjectMaskSuppression``; this one is about
        ``no_obj_embed_spatial`` reaching the bank and nothing else.
        """
        model = tiny_model()
        features = ops.convert_to_tensor(np.zeros(
            (1, TINY_GRID, TINY_GRID, TINY_DIM), dtype="float32"))
        mask_edge = TINY_GRID * 4
        rng = np.random.default_rng(11)
        outputs = {
            "low_res_logits": ops.convert_to_tensor(rng.standard_normal(
                (1, 1, mask_edge, mask_edge)).astype("float32")),
            "iou_predictions": ops.convert_to_tensor(
                np.asarray([[0.5]], dtype="float32")),
            # Strictly negative: the frame is predicted OCCLUDED, which is the
            # only case in which this embedding contributes anything.
            "object_score_logits": ops.convert_to_tensor(
                np.asarray([[-5.0]], dtype="float32")),
            "object_pointer": ops.convert_to_tensor(
                np.zeros((1, TINY_DIM), dtype="float32")),
        }

        def stored(value: float) -> np.ndarray:
            model.stream_reset()
            model.no_obj_embed_spatial.assign(
                np.full((1, TINY_MEM_DIM), value, dtype="float32"))
            model._store_memory(0, features, outputs, is_conditioning=True)
            return ops.convert_to_numpy(
                model.memory_bank.cond_frames[0].features)

        moved = float(np.max(np.abs(stored(7.0) - stored(0.0))))
        assert moved == pytest.approx(7.0, abs=1e-5), (
            f"changing no_obj_embed_spatial by 7.0 moved the stored memory by "
            f"{moved} -- it is declared but not wired into _store_memory"
        )


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
        """A box prompt must change the prediction, or the prompt path is dead.

        Object score pinned positive (D-043): a suppressed mask is the same
        constant with and without the prompt, and the prompt can itself move
        the score across zero, so both failure modes are removed by pinning.
        """
        model = tiny_model()
        pin_object_score(model, 5.0)
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
        # Move-proof: resolve the source file from the ALREADY-IMPORTED module
        # object rather than from a repo-relative literal, which silently
        # becomes a FileNotFoundError the moment the package is relocated.
        # Same idiom as ``test_hiera.py``'s ``inspect.getfile(package)``.
        import inspect

        source = pathlib.Path(inspect.getfile(model_module)).read_text()
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
