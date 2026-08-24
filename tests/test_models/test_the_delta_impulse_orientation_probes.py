"""R-140: delta-impulse orientation probes on NON-SQUARE grids, for the three
packages the Phase-3 audit left NOT-ASSESSED at a CRITICAL rule.

`findings/audit-batch-8.md` §4 corrected the batch-1 close-out: batch 1 claimed
"none is CRITICAL" for its NOT-ASSESSED debt only by re-grading R-140 from
CRITICAL down to HIGH, which the Phase-3 header forbids. The residue is exactly
three cells -- ``R-140`` x ``SAM``, ``bias_free_denoisers``, ``sd3_mmdit`` --
and they are the plan's ONLY CRITICAL residue. Batch 1 recorded that the harder
half (non-square shapes in the test data) is already present in those dirs; what
was missing is a genuine delta-impulse probe per stride path. This module is it.

The instrument, its two assertion forms and why a square grid is refused live in
``delta_impulse_orientation_oracle`` -- read that module first.

Every probe here is proven RED by ``transposed_stride_injection``, which swaps
the path's spatial axes on the way in and out. That injection changes no shape,
no parameter count and no dtype; it changes only orientation, which is precisely
the defect class R-140 names.
"""

import keras
import numpy as np
import pytest

from .delta_impulse_orientation_oracle import (
    assert_impulse_support_box,
    assert_orientation_is_diagonal,
    impulse_energy_map,
    transposed_stride_injection,
)


def _seeded() -> None:
    keras.utils.set_random_seed(20260820)


# ---------------------------------------------------------------------------
# SAM -- batch 1 named "the ViTDet/Hiera stride path".
# ---------------------------------------------------------------------------


def _sam1_mask_downscaling():
    from dl_techniques.models.SAM.SAM1.prompt_encoder import PromptEncoder

    _seeded()
    encoder = PromptEncoder(
        embed_dim=32,
        image_embedding_size=(16, 10),
        input_image_size=(256, 160),
        mask_in_chans=16,
    )
    return lambda x: encoder.mask_downscaling(x, training=False)


def _sam1_output_upscaling():
    from dl_techniques.models.SAM.SAM1.mask_decoder import MaskDecoder
    from dl_techniques.models.SAM.SAM1.transformer import TwoWayTransformer

    _seeded()
    decoder = MaskDecoder(
        transformer_dim=32,
        transformer=TwoWayTransformer(
            depth=1, embedding_dim=32, num_heads=2, mlp_dim=64
        ),
    )
    return lambda x: decoder.output_upscaling(x, training=False)


def _sam2_mask_downsampler():
    from dl_techniques.models.SAM.SAM2.memory_encoder import SAM2MaskDownSampler

    _seeded()
    layer = SAM2MaskDownSampler(embed_dim=16, total_stride=4, mask_in_chans=1)
    return lambda x: layer(x, training=False)


def _sam2_hiera():
    from dl_techniques.models.SAM.SAM2.hiera import Hiera

    _seeded()
    trunk = Hiera(
        embed_dim=16,
        num_heads=1,
        stages=(1, 1),
        global_att_blocks=(1,),
        window_spec=(2, 2),
        q_pool=1,
        image_size=64,
        patch_stride=4,
        patch_kernel_size=7,
        patch_padding=3,
    )
    return trunk


class TestSamStridePaths:
    """SAM's stride paths, split by whether a non-square grid is even accepted."""

    def test_sam1_mask_downscaling_maps_the_impulse_to_exactly_one_cell(self):
        """Stride-4 non-overlapping stem: ``(row, col) -> (row // 4, col // 4)``, exact."""
        forward = _sam1_mask_downscaling()
        assert_impulse_support_box(
            forward,
            (64, 40, 1),
            41,
            13,
            (10, 10, 3, 3),
            label="SAM1 PromptEncoder.mask_downscaling",
        )

    def test_sam1_mask_downscaling_is_red_under_a_transposed_stride(self):
        forward = transposed_stride_injection(_sam1_mask_downscaling())
        with pytest.raises(AssertionError, match="support box"):
            assert_impulse_support_box(
                forward,
                (64, 40, 1),
                41,
                13,
                (10, 10, 3, 3),
                label="SAM1 PromptEncoder.mask_downscaling (transposed)",
            )

    def test_sam1_output_upscaling_maps_the_impulse_to_exactly_one_block(self):
        """Two stride-2 transposed convs: ``(row, col) -> the 4x4 block at (4row, 4col)``."""
        forward = _sam1_output_upscaling()
        assert_impulse_support_box(
            forward,
            (16, 10, 32),
            11,
            3,
            (44, 47, 12, 15),
            label="SAM1 MaskDecoder.output_upscaling",
        )

    def test_sam1_output_upscaling_is_red_under_a_transposed_stride(self):
        forward = transposed_stride_injection(_sam1_output_upscaling())
        with pytest.raises(AssertionError, match="support box"):
            assert_impulse_support_box(
                forward,
                (16, 10, 32),
                11,
                3,
                (44, 47, 12, 15),
                label="SAM1 MaskDecoder.output_upscaling (transposed)",
            )

    def test_sam2_mask_downsampler_orientation_is_diagonal(self):
        assert_orientation_is_diagonal(
            _sam2_mask_downsampler(),
            (64, 40, 1),
            (28, 12),
            8,
            label="SAM2MaskDownSampler",
        )

    def test_sam2_mask_downsampler_is_red_under_a_transposed_stride(self):
        forward = transposed_stride_injection(_sam2_mask_downsampler())
        with pytest.raises(AssertionError):
            assert_orientation_is_diagonal(
                forward,
                (64, 40, 1),
                (28, 12),
                8,
                label="SAM2MaskDownSampler (transposed)",
            )

    def test_sam2_hiera_accepts_a_non_square_grid_and_both_stages_are_oriented(self):
        """Hiera IS probeable: ``patch_stride`` and ``q_stride`` both take a 2-tuple."""
        trunk = _sam2_hiera()
        outputs = trunk(np.zeros((1, 64, 40, 3), dtype="float32"), training=False)
        assert [tuple(t.shape) for t in outputs] == [
            (1, 16, 10, 16),
            (1, 8, 5, 32),
        ], (
            "Hiera's non-square output shapes changed; the probe grid below is "
            "pinned to them."
        )
        for index, label in ((0, "patch_stride stage"), (1, "q_stride stage")):
            assert_orientation_is_diagonal(
                lambda x, _i=index: trunk(x, training=False)[_i],
                (64, 40, 3),
                (28, 12),
                8,
                label=f"SAM2 Hiera {label}",
            )

    def test_sam2_hiera_is_red_under_a_transposed_stride(self):
        trunk = _sam2_hiera()
        forward = transposed_stride_injection(
            lambda x: trunk(x, training=False)[0]
        )
        with pytest.raises(AssertionError):
            assert_orientation_is_diagonal(
                forward,
                (64, 40, 3),
                (28, 12),
                8,
                label="SAM2 Hiera patch_stride stage (transposed)",
            )

    def test_the_two_vit_trunks_are_square_only_by_construction(self):
        """R-140 has NO subject in the ViT trunks -- measured, not assumed.

        Batch 1 wrote "the ViTDet/Hiera stride path". Hiera is probeable (above).
        The two ViT trunks are not: both refuse a non-square grid outright, so
        there is no non-square forward to orient. The refusals differ in quality
        and both halves are pinned, because a change either way matters:

        * ``Sam3ViTDetBackbone`` refuses with an explicit, keyed ``ValueError``.
        * ``ImageEncoderViT`` refuses with a clean ``ValueError`` when the width
          is not divisible by the patch, but at a divisible non-square width it
          crashes inside the absolute-position add instead -- the position
          embedding is stored square. That is a latent guard gap, recorded here
          rather than silently absorbed into an R-140 "N/A".
        """
        from dl_techniques.models.SAM.SAM1.image_encoder import ImageEncoderViT
        from dl_techniques.models.SAM.SAM3.vitdet import Sam3ViTDetBackbone

        _seeded()
        vitdet = Sam3ViTDetBackbone(
            img_size=64,
            patch_size=16,
            in_channels=3,
            embed_dim=32,
            depth=2,
            num_heads=2,
            window_size=4,
            global_att_blocks=(1,),
            use_rope=False,
        )
        with pytest.raises(ValueError, match=r"must match img_size"):
            vitdet(np.zeros((1, 64, 48, 3), dtype="float32"), training=False)

        encoder = ImageEncoderViT(
            img_size=64,
            patch_size=16,
            embed_dim=32,
            depth=2,
            num_heads=2,
            out_chans=16,
            window_size=0,
        )
        with pytest.raises(ValueError, match=r"divisible by patch width"):
            encoder(np.zeros((1, 64, 40, 3), dtype="float32"), training=False)
        with pytest.raises(Exception) as excinfo:
            encoder(np.zeros((1, 64, 48, 3), dtype="float32"), training=False)
        _assert_positional_add_shape_error(str(excinfo.value))


# ---------------------------------------------------------------------------
# bias_free_denoisers -- three builders, one U-shaped stride path each
# (bfcnn is stride-1 throughout, which is itself the thing to pin).
# ---------------------------------------------------------------------------


def _bfd_builders():
    from dl_techniques.models.bias_free_denoisers.bfcnn import create_bfcnn_denoiser
    from dl_techniques.models.bias_free_denoisers.bfconvunext import (
        create_convunext_denoiser,
    )
    from dl_techniques.models.bias_free_denoisers.bfunet import create_bfunet_denoiser

    shape = (64, 40, 1)
    return {
        "bfunet": lambda: create_bfunet_denoiser(
            input_shape=shape, depth=2, initial_filters=8, blocks_per_level=1
        ),
        "bfconvunext": lambda: create_convunext_denoiser(
            input_shape=shape, depth=2, initial_filters=8, blocks_per_level=1
        ),
        "bfcnn": lambda: create_bfcnn_denoiser(
            input_shape=shape, num_blocks=3, filters=8
        ),
    }


class TestBiasFreeDenoiserStridePaths:
    """All three bias-free denoisers, probed at their native non-square input."""

    @pytest.mark.parametrize("name", ["bfunet", "bfconvunext", "bfcnn"])
    def test_orientation_is_diagonal(self, name):
        _seeded()
        model = _bfd_builders()[name]()
        assert_orientation_is_diagonal(
            lambda x: model(x, training=False),
            (64, 40, 1),
            (28, 12),
            8,
            label=f"bias_free_denoisers {name}",
        )

    @pytest.mark.parametrize("name", ["bfunet", "bfconvunext", "bfcnn"])
    def test_is_red_under_a_transposed_stride(self, name):
        _seeded()
        model = _bfd_builders()[name]()
        forward = transposed_stride_injection(lambda x: model(x, training=False))
        with pytest.raises(AssertionError):
            assert_orientation_is_diagonal(
                forward,
                (64, 40, 1),
                (28, 12),
                8,
                label=f"bias_free_denoisers {name} (transposed)",
            )

    def test_the_denoisers_preserve_the_non_square_shape(self):
        """A U-Net that silently squared its output would pass every mean-PSNR test."""
        for name, build in _bfd_builders().items():
            _seeded()
            model = build()
            out = model(np.zeros((1, 64, 40, 1), dtype="float32"), training=False)
            assert tuple(out.shape) == (1, 64, 40, 1), (
                f"{name} returned {tuple(out.shape)} for a 64x40 input."
            )


# ---------------------------------------------------------------------------
# sd3_mmdit -- batch 1 named "the VAE stride path".
# ---------------------------------------------------------------------------


def _sd3_vae():
    from dl_techniques.models.sd3_mmdit.vae import create_sd3_vae

    _seeded()
    return create_sd3_vae("tiny")


def _sd3_latent_channels() -> int:
    from dl_techniques.models.sd3_mmdit.config import get_sd3_config

    return int(get_sd3_config("tiny")[1].z_channels)


class TestSd3VaeStridePath:
    """The SD3 VAE's down and up paths. Both mix globally, so the diagonal form applies."""

    def test_encoder_orientation_is_diagonal(self):
        vae = _sd3_vae()
        assert_orientation_is_diagonal(
            lambda x: vae.encoder(x, training=False),
            (64, 40, 3),
            (28, 12),
            8,
            label="sd3_mmdit VAE encoder",
        )

    def test_encoder_is_red_under_a_transposed_stride(self):
        vae = _sd3_vae()
        forward = transposed_stride_injection(
            lambda x: vae.encoder(x, training=False)
        )
        with pytest.raises(AssertionError):
            assert_orientation_is_diagonal(
                forward,
                (64, 40, 3),
                (28, 12),
                8,
                label="sd3_mmdit VAE encoder (transposed)",
            )

    def test_decoder_orientation_is_diagonal(self):
        vae = _sd3_vae()
        assert_orientation_is_diagonal(
            lambda x: vae.decoder(x, training=False),
            (16, 10, _sd3_latent_channels()),
            (6, 3),
            2,
            label="sd3_mmdit VAE decoder",
        )

    def test_decoder_is_red_under_a_transposed_stride(self):
        vae = _sd3_vae()
        forward = transposed_stride_injection(
            lambda x: vae.decoder(x, training=False)
        )
        with pytest.raises(AssertionError):
            assert_orientation_is_diagonal(
                forward,
                (16, 10, _sd3_latent_channels()),
                (6, 3),
                2,
                label="sd3_mmdit VAE decoder (transposed)",
            )

    def test_the_vae_round_trip_preserves_the_non_square_shape(self):
        vae = _sd3_vae()
        latent = vae.encoder(np.zeros((1, 64, 40, 3), dtype="float32"), training=False)
        assert tuple(latent.shape)[1:3] == (32, 20), (
            f"encoder gave {tuple(latent.shape)} for 64x40; the ch_mult of the "
            "'tiny' preset is (1, 2), i.e. one 2x downsample."
        )


# ---------------------------------------------------------------------------
# The instrument's own refusals.
# ---------------------------------------------------------------------------


class TestTheProbeRefusesASquareGrid:
    """R-140's discriminating condition, asserted on the instrument itself."""

    def test_a_square_grid_is_refused(self):
        identity = lambda x: keras.ops.convert_to_tensor(x)
        with pytest.raises(AssertionError, match="SQUARE"):
            impulse_energy_map(identity, (32, 32, 1), 5, 7, label="square probe")

    def test_a_non_square_grid_is_accepted(self):
        identity = lambda x: keras.ops.convert_to_tensor(x)
        energy = impulse_energy_map(identity, (32, 20, 1), 5, 7, label="non-square")
        assert energy.shape == (32, 20)
        assert energy[5, 7] == 1.0
        assert energy.sum() == 1.0


# ---------------------------------------------------------------------------
# The `ImageEncoderViT` non-square refusal is matched on the OP, not the prose
# ---------------------------------------------------------------------------

#: TF's shape-mismatch wording is DEVICE-DEPENDENT for the same failure, so a
#: literal match on either phrase is a test that passes on one machine and
#: fails on the next. MEASURED for the identical call
#: (`ImageEncoderViT(img_size=64, patch_size=16, ...)` fed `(1, 64, 48, 3)`):
#:
#:   CPU  "Incompatible shapes: [1,4,3,32] vs. [1,4,4,32] [Op:AddV2]"
#:   GPU  "required broadcastable shapes [Op:AddV2]"
#:
#: The first form was pinned at step 19 and was RED on a GPU run. What is
#: invariant -- and what the docstring above actually claims -- is that the
#: refusal happens inside the ABSOLUTE-POSITION ADD, whose op is `AddV2`. That
#: is the substantive assertion; the prose is decoration. Both phrasings are
#: accepted so a wrong-op or non-shape failure is still rejected.
_SHAPE_ERROR_PHRASES = ("Incompatible shapes", "required broadcastable shapes")


def _assert_positional_add_shape_error(message: str) -> None:
    """Reject anything that is not a shape mismatch in an `AddV2`."""
    assert "[Op:AddV2]" in message, (
        f"expected the non-square refusal to come from the absolute-position "
        f"add (`[Op:AddV2]`); got: {message!r}"
    )
    assert any(p in message for p in _SHAPE_ERROR_PHRASES), (
        f"expected a shape-mismatch message in one of the two device-dependent "
        f"phrasings {_SHAPE_ERROR_PHRASES}; got: {message!r}"
    )


def test_the_positional_add_matcher_rejects_the_wrong_failure():
    """RED proof for the matcher itself, both halves.

    Without this the matcher could be weakened to `assert message` and nothing
    would notice.
    """
    for good in (
        "Incompatible shapes: [1,4,3,32] vs. [1,4,4,32] [Op:AddV2] name:",
        "required broadcastable shapes [Op:AddV2] name:",
    ):
        _assert_positional_add_shape_error(good)

    with pytest.raises(AssertionError, match=r"absolute-position add"):
        # right words, WRONG op -- a shape error somewhere else entirely
        _assert_positional_add_shape_error(
            "Incompatible shapes: [1,4,3,32] vs. [1,4,4,32] [Op:MatMul]"
        )
    with pytest.raises(AssertionError, match=r"shape-mismatch message"):
        # right op, but not a shape failure
        _assert_positional_add_shape_error("Could not find device [Op:AddV2]")
