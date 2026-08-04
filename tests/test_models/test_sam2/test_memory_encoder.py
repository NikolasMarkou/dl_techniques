"""
Guards for SAM 2's memory encoder (plan step 5, guards G5.1-G5.5).

Every mechanism guarded here is SILENT when ported wrong -- the model builds,
forward-passes, trains and serializes either way:

    * The mask affine ``sigmoid(20x - 10)`` degrades to a bare ``sigmoid(x)``
      with identical shapes, an identical output range and a plausible loss.
    * The downsampler's signature default ``k=4/s=4/p=0`` reaches the SAME
      total stride of 16 as the shipped ``k=3/s=2/p=1``, with two convolutions
      instead of four. An assertion on the output resolution is therefore
      VACUOUS -- ``TestDownSamplerVacuity`` proves that by executing it.
    * Concatenating instead of adding doubles the fused width, which the fuser
      would silently absorb if it inferred its width at build time.
    * The positional encoding is 64 channels HERE and 256 at the neck. On a
      square grid a 128-wide encoding BROADCASTS against a 64-wide memory
      rather than raising, so this site fails silently where the neck fails
      loudly.

Guard map:

    G5.1  ``TestAffineSigmoidValue``     -- VALUE oracle, not liveness
    G5.2  ``TestDownSamplerGeometry``    -- layer count + derived channel ladder
          ``TestDownSamplerVacuity``     -- proves the stride assertion vacuous
    G5.3  ``TestAdditiveFusion``         -- fuser width 256, not 512
    G5.4  ``TestPositionEncodingWidth``  -- 64 channels HERE (256 in step 4)
    G5.5  ``TestDeadComponentPartition`` -- the MEASURED red/green partition

    A-6   ``TestConvNextBlockMatchesCXBlock`` -- the FIRST execution of the
          plan's assumption that ``ConvNextV1Block`` reproduces the reference
          ``CXBlock``. Read structurally at EXPLORE, never run until here.
"""

from typing import Any, Dict, Iterator, List, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops
from scipy.special import erf

from dl_techniques.layers.convnext_v1_block import ConvNextV1Block
from dl_techniques.models.sam2.memory_encoder import (
    SAM2Fuser,
    SAM2MaskDownSampler,
    SAM2MemoryEncoder,
)

from ..test_sam.dead_component_oracle import zeroed_variables

# ---------------------------------------------------------------------
# Test geometry.
#
# Small enough to run on CPU, structurally identical to the shipped
# configuration: four downsampler stages at total stride 16, two fuser blocks,
# an additive fusion and an out-projection that narrows the width.
# ---------------------------------------------------------------------

BATCH = 2
SEED = 5171

#: Pixel-feature grid. The mask grid is `total_stride` times larger.
FEATURE_GRID = 4
IN_DIM = 32
OUT_DIM = 8
TOTAL_STRIDE = 16
MASK_GRID = FEATURE_GRID * TOTAL_STRIDE

#: The SHIPPED downsampler configuration (the YAML, not the class signature).
SHIPPED_DOWNSAMPLER: Dict[str, int] = {
    "kernel_size": 3,
    "stride": 2,
    "padding": 1,
    "total_stride": 16,
}
#: The reference class's SIGNATURE DEFAULT. Same total stride, half the depth.
SIGNATURE_DEFAULT_DOWNSAMPLER: Dict[str, int] = {
    "kernel_size": 4,
    "stride": 4,
    "padding": 0,
    "total_stride": 16,
}

#: The shipped affine on the mask logits.
SIGMOID_SCALE = 20.0
SIGMOID_BIAS = -10.0

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded generator so every assertion below is reproducible."""
    return np.random.default_rng(SEED)


@pytest.fixture()
def pix_feat(rng: np.random.Generator) -> np.ndarray:
    """Pixel features at the memory grid."""
    return rng.normal(
        size=(BATCH, FEATURE_GRID, FEATURE_GRID, IN_DIM)).astype("float32")


@pytest.fixture()
def mask_logits(rng: np.random.Generator) -> np.ndarray:
    """High-resolution mask LOGITS -- not probabilities."""
    return rng.normal(
        scale=2.0, size=(BATCH, MASK_GRID, MASK_GRID, 1)).astype("float32")


@pytest.fixture()
def encoder() -> SAM2MemoryEncoder:
    """A built memory encoder at the shipped structural configuration."""
    layer = SAM2MemoryEncoder(in_dim=IN_DIM, out_dim=OUT_DIM)
    layer.build([
        (None, FEATURE_GRID, FEATURE_GRID, IN_DIM),
        (None, MASK_GRID, MASK_GRID, 1),
    ])
    return layer


@pytest.fixture()
def no_tf32() -> Iterator[None]:
    """Disable TF32 for the duration of a numeric-oracle test.

    MEASURED at step 5: the same ``ConvNextV1Block``-versus-oracle comparison
    reports a max-abs difference of ``5.2e-4`` with TF32 on (an RTX 4070's
    default) and ``2.4e-7`` with it off -- a factor of ~2000. A tolerance
    chosen under one regime is meaningless in the other, and the repo already
    records a test module that disables TF32 process-globally at import, so the
    ambient state depends on which files pytest collected. This fixture pins it
    and restores the previous value.
    """
    previous = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    try:
        yield
    finally:
        tf.config.experimental.enable_tensor_float_32_execution(previous)

# ---------------------------------------------------------------------
# G5.1 -- the mask affine, by VALUE
# ---------------------------------------------------------------------


class TestAffineSigmoidValue:
    """``sigmoid(20x - 10)``, asserted as a VALUE, never as liveness.

    A liveness probe ("the mask branch moves the output") goes green on code
    computing a bare ``sigmoid(x)``: both are monotone maps into ``(0, 1)`` and
    both move the output. Only the transformed VALUE distinguishes them.
    """

    def test_transforms_a_known_logit_to_a_known_probability(
            self, encoder: SAM2MemoryEncoder) -> None:
        """At logit 0.6 the affine gives ``sigmoid(2.0)``, not ``sigmoid(0.6)``.

        ``20 * 0.6 - 10 == 2.0``. The two candidate answers differ by 0.235,
        which is 5 orders of magnitude above the 1e-6 tolerance.
        """
        probe = np.full((1, 1, 1, 1), 0.6, dtype="float32")
        actual = float(ops.convert_to_numpy(
            encoder._affine_sigmoid(probe)).reshape(-1)[0])

        affine_answer = 1.0 / (1.0 + np.exp(-2.0))
        bare_answer = 1.0 / (1.0 + np.exp(-0.6))

        assert affine_answer == pytest.approx(0.880797, abs=1e-6)
        assert bare_answer == pytest.approx(0.645656, abs=1e-6)
        assert actual == pytest.approx(affine_answer, abs=1e-6), (
            f"the mask affine produced {actual}; sigmoid(20x-10) at x=0.6 is "
            f"{affine_answer}, a bare sigmoid(x) is {bare_answer}"
        )
        assert abs(actual - bare_answer) > 0.2, (
            f"the transformed value is within 0.2 of a BARE sigmoid "
            f"({bare_answer}) -- the scale/bias is not being applied"
        )

    @pytest.mark.usefixtures("no_tf32")
    def test_affine_is_the_transform_the_forward_pass_actually_uses(
            self,
            encoder: SAM2MemoryEncoder,
            pix_feat: np.ndarray,
            mask_logits: np.ndarray,
    ) -> None:
        """End-to-end: ``call`` uses exactly ``sigmoid(20x - 10)``.

        The previous test reads a private method. This one proves the FORWARD
        PASS uses it, by feeding a ``skip_mask_sigmoid=True`` twin -- weights
        copied verbatim -- two candidate hand-transformed masks and asking
        which one the encoder reproduces.

        **The test is two-sided and self-calibrating**: it asserts agreement
        with the affine candidate AND disagreement with the bare-sigmoid
        candidate, with the measured margin between them in the message. A
        one-sided tolerance would be a hypothesis about float noise; this
        compares the two distances the tolerance has to separate.

        **Bit identity is deliberately NOT demanded.** MEASURED at step 5: the
        same comparison returns exactly 0.0 when this file runs alone and
        4.2e-4 when the whole ``test_sam2`` directory runs, because the two
        forward passes receive inputs differing at the 1e-8 level and TF32
        truncation amplifies that across the four strided convolutions. The
        ``no_tf32`` fixture removes the amplification; the margin assertion
        removes the dependence on how small the residue happens to be.
        """
        twin = SAM2MemoryEncoder(
            in_dim=IN_DIM, out_dim=OUT_DIM, skip_mask_sigmoid=True)
        twin.build([
            (None, FEATURE_GRID, FEATURE_GRID, IN_DIM),
            (None, MASK_GRID, MASK_GRID, 1),
        ])
        twin.set_weights(encoder.get_weights())

        logits = mask_logits.astype("float64")
        affine_mask = 1.0 / (1.0 + np.exp(
            -(SIGMOID_SCALE * logits + SIGMOID_BIAS)))
        bare_mask = 1.0 / (1.0 + np.exp(-logits))

        actual = ops.convert_to_numpy(
            encoder([pix_feat, mask_logits])[0]).astype("float64")
        as_affine = ops.convert_to_numpy(
            twin([pix_feat, affine_mask.astype("float32")])[0]
        ).astype("float64")
        as_bare = ops.convert_to_numpy(
            twin([pix_feat, bare_mask.astype("float32")])[0]
        ).astype("float64")

        to_affine = float(np.abs(actual - as_affine).max())
        to_bare = float(np.abs(actual - as_bare).max())

        assert to_affine < 1e-5, (
            f"the forward pass does not apply sigmoid(20x-10) to the mask: "
            f"distance to the affine twin {to_affine}, to the bare-sigmoid "
            f"twin {to_bare}"
        )
        assert to_bare > 1e-3, (
            f"the affine and bare-sigmoid candidates are only {to_bare} apart "
            f"at these weights -- this oracle cannot discriminate and the "
            f"probe must be re-seeded"
        )

    def test_affine_survives_a_logit_that_overflows_float16(
            self,
            pix_feat: np.ndarray,
    ) -> None:
        """``20 * x`` must not overflow before the sigmoid saturates it.

        Under ``mixed_float16`` the compute dtype's maximum is 65504, so a
        logit of 1e4 overflows at ``20 * x = 2e5``. Computing the affine in the
        VARIABLE dtype (float32 under this policy) keeps it finite.
        """
        previous = keras.mixed_precision.global_policy()
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            layer = SAM2MemoryEncoder(in_dim=IN_DIM, out_dim=OUT_DIM)
            layer.build([
                (None, FEATURE_GRID, FEATURE_GRID, IN_DIM),
                (None, MASK_GRID, MASK_GRID, 1),
            ])
            huge = np.full(
                (1, MASK_GRID, MASK_GRID, 1), 1.0e4, dtype="float32")
            transformed = ops.convert_to_numpy(layer._affine_sigmoid(huge))
        finally:
            keras.mixed_precision.set_global_policy(previous)

        assert np.isfinite(transformed).all(), (
            "the mask affine produced non-finite values under mixed_float16 -- "
            "20 * x overflowed the compute dtype before the sigmoid saturated"
        )
        assert float(transformed.min()) == pytest.approx(1.0, abs=1e-3)

# ---------------------------------------------------------------------
# G5.2 -- downsampler geometry
# ---------------------------------------------------------------------


class TestDownSamplerGeometry:
    """Layer COUNT and channel ladder, not output resolution."""

    def test_layer_count_is_derived_from_the_shipped_stride(self) -> None:
        """``log(16)/log(2) == 4`` convolutions, not 2."""
        down = SAM2MaskDownSampler(
            embed_dim=IN_DIM, **SHIPPED_DOWNSAMPLER)
        assert down.num_layers == 4, (
            f"the shipped k=3/s=2/p=1 configuration has 4 strided stages, got "
            f"{down.num_layers} -- the class SIGNATURE default k=4/s=4/p=0 has "
            f"2 and the SAME total stride"
        )
        assert len(down.convs) == 4
        assert all(
            conv.strides == (SHIPPED_DOWNSAMPLER["stride"],) * 2
            for conv in down.convs
        )

    def test_channel_ladder_grows_by_stride_squared(self) -> None:
        """The ladder is derived HERE, not copied from the implementation."""
        stride = SHIPPED_DOWNSAMPLER["stride"]
        expected: List[int] = []
        width = 1
        for _ in range(4):
            width *= stride ** 2
            expected.append(width)
        assert expected == [4, 16, 64, 256]

        down = SAM2MaskDownSampler(embed_dim=IN_DIM, **SHIPPED_DOWNSAMPLER)
        assert list(down.channel_sequence) == expected
        assert [conv.filters for conv in down.convs] == expected

    def test_final_projection_is_bare(self) -> None:
        """No normalization and no activation follow the final ``1x1``."""
        down = SAM2MaskDownSampler(embed_dim=IN_DIM, **SHIPPED_DOWNSAMPLER)
        assert down.final_conv.kernel_size == (1, 1)
        assert down.final_conv.filters == IN_DIM
        assert len(down.norms) == down.num_layers
        assert len(down.activations) == down.num_layers

    def test_output_grid_is_the_input_over_total_stride(self) -> None:
        """Shape check -- necessary, and on its own insufficient (see below)."""
        down = SAM2MaskDownSampler(embed_dim=IN_DIM, **SHIPPED_DOWNSAMPLER)
        output = down(np.zeros((1, MASK_GRID, MASK_GRID, 1), dtype="float32"))
        assert tuple(output.shape) == (1, FEATURE_GRID, FEATURE_GRID, IN_DIM)

    def test_padding_is_symmetric_not_keras_same(self) -> None:
        """``padding='same'`` at k=3/s=2 pads asymmetrically; this does not.

        Both produce the same output SHAPE, so only a VALUE comparison sees the
        difference. A constant input isolates the border: with symmetric
        padding the first output row sees one padded row above, with
        ``'same'`` it sees none.
        """
        down = SAM2MaskDownSampler(
            embed_dim=1, mask_in_chans=1, **SHIPPED_DOWNSAMPLER)
        ones = np.ones((1, MASK_GRID, MASK_GRID, 1), dtype="float32")
        down(ones)

        same = keras.layers.Conv2D(
            filters=down.convs[0].filters, kernel_size=3, strides=2,
            padding="same")
        same.build((1, MASK_GRID, MASK_GRID, 1))
        same.set_weights(down.convs[0].get_weights())

        symmetric = ops.convert_to_numpy(
            down.convs[0](down.pads[0](ones)))
        as_same = ops.convert_to_numpy(same(ones))

        assert symmetric.shape == as_same.shape
        assert float(np.abs(symmetric - as_same).max()) > 1e-6, (
            "symmetric padding and padding='same' agree -- one of them is not "
            "doing what this test assumes"
        )


class TestDownSamplerVacuity:
    """Executes the proof that a stride-only assertion cannot discriminate.

    This is not a guard on the implementation. It is a guard on the OTHER
    guards: it demonstrates by execution that the mutation named in the plan
    (the reference class's signature default) reaches the same total stride, so
    any future author who replaces ``TestDownSamplerGeometry`` with a
    resolution check has removed the only real assertion.
    """

    def test_both_configurations_reach_total_stride_16(self) -> None:
        shipped = SAM2MaskDownSampler(
            embed_dim=IN_DIM, **SHIPPED_DOWNSAMPLER)
        signature = SAM2MaskDownSampler(
            embed_dim=IN_DIM, **SIGNATURE_DEFAULT_DOWNSAMPLER)

        probe = np.zeros((1, MASK_GRID, MASK_GRID, 1), dtype="float32")
        shipped_shape = tuple(shipped(probe).shape)
        signature_shape = tuple(signature(probe).shape)

        assert shipped_shape == signature_shape, (
            "the two configurations no longer agree on output shape -- the "
            "vacuity this test documents has changed"
        )
        assert shipped.num_layers == 4 and signature.num_layers == 2
        assert list(signature.channel_sequence) == [16, 256]

# ---------------------------------------------------------------------
# G5.3 -- additive fusion
# ---------------------------------------------------------------------


class TestAdditiveFusion:
    """The fuser sees ``in_dim`` channels, never ``2 * in_dim``."""

    def test_fuser_width_is_in_dim_not_double(
            self, encoder: SAM2MemoryEncoder) -> None:
        assert encoder.fuser.dim == IN_DIM, (
            f"the fuser is configured for {encoder.fuser.dim} channels; "
            f"{2 * IN_DIM} would mean the fusion was concatenated"
        )
        block = encoder.fuser.blocks[0]
        assert block.filters == IN_DIM
        assert tuple(block.conv_1.kernel.shape)[2] == IN_DIM

    def test_concatenated_width_raises_at_build(self) -> None:
        """A concat fusion is not silently absorbed."""
        fuser = SAM2Fuser(dim=IN_DIM, num_layers=2)
        with pytest.raises(ValueError, match="ADDITIVELY"):
            fuser.build((None, FEATURE_GRID, FEATURE_GRID, 2 * IN_DIM))

    def test_scaling_the_mask_branch_moves_the_output(
            self,
            encoder: SAM2MemoryEncoder,
            pix_feat: np.ndarray,
            mask_logits: np.ndarray,
    ) -> None:
        """Liveness of the mask branch, as the plan specifies for G5.3."""
        base = ops.convert_to_numpy(encoder([pix_feat, mask_logits])[0])
        scaled = ops.convert_to_numpy(
            encoder([pix_feat, mask_logits * 2.0])[0])
        assert float(np.abs(base - scaled).max()) > 1e-5, (
            "doubling the mask logits did not move the memory -- the mask "
            "branch is not reaching the fusion"
        )

    def test_both_branches_reach_the_output(
            self,
            encoder: SAM2MemoryEncoder,
            pix_feat: np.ndarray,
            mask_logits: np.ndarray,
    ) -> None:
        """Perturbing EITHER branch alone must move the memory."""
        base = ops.convert_to_numpy(encoder([pix_feat, mask_logits])[0])
        moved_pix = ops.convert_to_numpy(
            encoder([pix_feat + 1.0, mask_logits])[0])
        assert float(np.abs(base - moved_pix).max()) > 1e-5, (
            "the pixel-feature branch is dead")

# ---------------------------------------------------------------------
# G5.4 -- positional-encoding width
# ---------------------------------------------------------------------


class TestPositionEncodingWidth:
    """64 channels HERE. The neck asserts 256 in its own file (H-11).

    The literal ``64`` is written in THIS file only. Sharing a constant with
    the neck's test would let one edit move both sites at once, which is the
    exact unification H-11 forbids.
    """

    def test_encoding_is_out_dim_channels(
            self,
            encoder: SAM2MemoryEncoder,
            pix_feat: np.ndarray,
            mask_logits: np.ndarray,
    ) -> None:
        _, position = encoder([pix_feat, mask_logits])
        assert int(position.shape[-1]) == OUT_DIM
        assert encoder.pos_enc_channels == OUT_DIM

    def test_shipped_memory_encoding_is_64_channels(self) -> None:
        """At the SHIPPED widths the encoding is 64 wide, from a 32 argument."""
        shipped = SAM2MemoryEncoder(in_dim=256, out_dim=64)
        assert shipped.num_pos_feats == 32, (
            "the reference config's literal `num_pos_feats: 64` belongs to a "
            "class that halves it internally; this repo's layer does not"
        )
        assert shipped.pos_enc_channels == 64

    def test_encoding_is_channels_last_and_matches_the_feature_dtype(
            self,
            encoder: SAM2MemoryEncoder,
            pix_feat: np.ndarray,
            mask_logits: np.ndarray,
    ) -> None:
        """The underlying layer emits channels-FIRST float32 at every policy."""
        memory, position = encoder([pix_feat, mask_logits])
        assert tuple(position.shape) == tuple(memory.shape)
        assert keras.backend.standardize_dtype(position.dtype) == \
            keras.backend.standardize_dtype(memory.dtype)

    def test_a_wrong_width_would_broadcast_rather_than_raise(
            self,
            pix_feat: np.ndarray,
            mask_logits: np.ndarray,
    ) -> None:
        """Documents WHY this site needs a guard and the neck does not.

        The neck's 512-vs-256 mistake fails loudly on the addition into the
        256-wide query stream. Here the encoding is returned alongside the
        memory rather than added to it, so a 2x-wide encoding produces no error
        at all -- only a wrong tensor.
        """
        wrong = SAM2MemoryEncoder(
            in_dim=IN_DIM, out_dim=OUT_DIM, num_pos_feats=OUT_DIM)
        _, position = wrong([pix_feat, mask_logits])
        assert int(position.shape[-1]) == 2 * OUT_DIM, (
            "the mis-specified encoding no longer doubles the width -- the "
            "silence this test documents has changed"
        )

# ---------------------------------------------------------------------
# A-6 -- the ConvNextV1Block / CXBlock equivalence, EXECUTED
# ---------------------------------------------------------------------


def _cxblock_reference(
        x: np.ndarray, weights: Dict[str, np.ndarray], kernel_size: int
) -> np.ndarray:
    """Float64 NumPy transcription of the reference fuser block's forward pass.

    ``dwconv(k, groups=dim) -> LayerNorm(eps=1e-6) -> Linear(4x) -> GELU ->
    Linear(reduce) -> gamma * x``, with NO GRN (that is the V2 block, which the
    reuse audit REJECTs) and NO residual add (the reference block adds it; the
    repo block is the branch only, so the fuser adds it instead).

    :param x: Input, ``(batch, height, width, channels)``.
    :param weights: The repo block's weights, keyed without the layer prefix.
    :param kernel_size: Depthwise kernel size.
    :return: The reference block's branch output, float64.
    """
    x = x.astype("float64")
    pad = kernel_size // 2
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    _, height, width, _ = x.shape

    depthwise = weights["depthwise_conv/kernel"]
    y = np.zeros_like(x)
    for u in range(kernel_size):
        for v in range(kernel_size):
            y += padded[:, u:u + height, v:v + width, :] * depthwise[u, v, :, 0]
    y = y + weights["depthwise_conv/bias"]

    mean = y.mean(axis=-1, keepdims=True)
    variance = y.var(axis=-1, keepdims=True)
    y = (y - mean) / np.sqrt(variance + 1e-6)
    y = y * weights["layer_norm/gamma"] + weights["layer_norm/beta"]

    y = y @ weights["expand_conv/kernel"][0, 0] + weights["expand_conv/bias"]
    # Exact GELU (erf form) -- both Keras and the reference default to exact,
    # not the tanh approximation.
    y = 0.5 * y * (1.0 + erf(y / np.sqrt(2.0)))
    y = y @ weights["reduce_conv/kernel"][0, 0] + weights["reduce_conv/bias"]
    return y * weights["gamma_scale/gamma"]


class TestConvNextBlockMatchesCXBlock:
    """Plan assumption A-6, EXECUTED for the first time.

    The reuse audit read ``ConvNextV1Block``'s computation order structurally
    and never ran a forward comparison. If this class fails, the fuser must be
    written SAM2-locally instead of reusing the repo block.
    """

    @pytest.mark.usefixtures("no_tf32")
    def test_forward_matches_a_float64_numpy_oracle(
            self, rng: np.random.Generator) -> None:
        channels, kernel = 8, 7
        x = rng.normal(size=(2, 9, 9, channels)).astype("float32")

        block = ConvNextV1Block(
            kernel_size=kernel, filters=channels, gamma_initial_value=1e-6)
        block.build((None, 9, 9, channels))
        for variable in block.weights:
            # gamma carries a non-negativity constraint; keep it positive.
            sample = rng.normal(scale=0.3, size=variable.shape)
            if "gamma_scale" in variable.path:
                sample = np.abs(sample)
            variable.assign(sample.astype("float32"))

        weights = {
            variable.path.split("/", 1)[1]:
                ops.convert_to_numpy(variable).astype("float64")
            for variable in block.weights
        }
        actual = ops.convert_to_numpy(
            block(x, training=False)).astype("float64")
        reference = _cxblock_reference(x, weights, kernel)

        difference = float(np.abs(actual - reference).max())
        assert difference < 1e-5, (
            f"ConvNextV1Block diverges from the reference CXBlock forward by "
            f"{difference} (float32 rounding alone is ~2e-7) -- A-6 is "
            f"FALSIFIED and the fuser needs a SAM2-local block"
        )

    def test_block_has_no_grn(self) -> None:
        """The V2 block's GRN is what makes it a REJECT; V1 must not have it."""
        block = ConvNextV1Block(kernel_size=7, filters=8)
        block.build((None, 4, 4, 8))
        names = {variable.path for variable in block.weights}
        assert not any("grn" in name.lower() for name in names), names
        assert len(names) == 9, sorted(names)

    def test_fuser_adds_the_residual_the_repo_block_omits(
            self, rng: np.random.Generator) -> None:
        """``ConvNextV1Block`` is the BRANCH only -- the fuser adds the skip.

        Zeroing every block's reduce-convolution kills the branch. With the
        residual present the fuser is then the identity; without it, the fuser
        would output zeros.
        """
        fuser = SAM2Fuser(dim=IN_DIM, num_layers=2)
        fuser.build((None, FEATURE_GRID, FEATURE_GRID, IN_DIM))
        probe = rng.normal(
            size=(1, FEATURE_GRID, FEATURE_GRID, IN_DIM)).astype("float32")

        killed = [
            variable for block in fuser.blocks
            for variable in block.conv_3.weights
        ]
        with zeroed_variables(killed):
            output = ops.convert_to_numpy(fuser(probe, training=False))

        assert float(np.abs(output - probe).max()) == pytest.approx(
            0.0, abs=1e-6), (
            "with the branch zeroed the fuser is not the identity -- the "
            "residual connection is missing"
        )

# ---------------------------------------------------------------------
# Authoring contract: serialization, shapes, raises
# ---------------------------------------------------------------------


class TestAuthoringContract:
    """Registration, config completeness, shape derivation, validation."""

    @pytest.mark.parametrize("layer, expected", [
        (SAM2MaskDownSampler(embed_dim=8), (None, 2, 2, 8)),
    ])
    def test_downsampler_compute_output_shape(
            self, layer: SAM2MaskDownSampler,
            expected: Tuple[Any, ...]) -> None:
        assert layer.compute_output_shape((None, 32, 32, 1)) == expected

    def test_encoder_compute_output_shape_matches_the_forward_pass(
            self,
            encoder: SAM2MemoryEncoder,
            pix_feat: np.ndarray,
            mask_logits: np.ndarray,
    ) -> None:
        memory, position = encoder([pix_feat, mask_logits])
        derived = encoder.compute_output_shape([
            (BATCH, FEATURE_GRID, FEATURE_GRID, IN_DIM),
            (BATCH, MASK_GRID, MASK_GRID, 1),
        ])
        assert derived[0] == tuple(memory.shape)
        assert derived[1] == tuple(position.shape)

    @pytest.mark.parametrize("cls, kwargs", [
        (SAM2MaskDownSampler, {"embed_dim": 8, "total_stride": 4}),
        (SAM2Fuser, {"dim": 8, "num_layers": 2}),
        (SAM2MemoryEncoder, {"in_dim": IN_DIM, "out_dim": OUT_DIM}),
    ])
    def test_config_round_trips(
            self, cls: Any, kwargs: Dict[str, Any]) -> None:
        original = cls(**kwargs)
        config = original.get_config()
        rebuilt = cls.from_config(config)
        assert rebuilt.get_config() == config

    def test_config_covers_every_init_parameter(self) -> None:
        """A dropped ``get_config`` key is a silent checkpoint break."""
        import inspect
        for cls in (SAM2MaskDownSampler, SAM2Fuser, SAM2MemoryEncoder):
            parameters = {
                name for name in
                inspect.signature(cls.__init__).parameters
                if name not in ("self", "kwargs")
            }
            config = cls().get_config()
            missing = parameters - set(config)
            assert not missing, f"{cls.__name__} get_config drops {missing}"

    def test_encoder_survives_a_keras_round_trip_by_value(
            self,
            pix_feat: np.ndarray,
            mask_logits: np.ndarray,
            tmp_path: Any,
    ) -> None:
        inputs = [
            keras.Input(shape=(FEATURE_GRID, FEATURE_GRID, IN_DIM)),
            keras.Input(shape=(MASK_GRID, MASK_GRID, 1)),
        ]
        outputs = SAM2MemoryEncoder(
            in_dim=IN_DIM, out_dim=OUT_DIM, name="memory_encoder")(inputs)
        model = keras.Model(inputs, outputs)

        before = ops.convert_to_numpy(model([pix_feat, mask_logits])[0])
        path = tmp_path / "memory_encoder.keras"
        model.save(path)
        restored = keras.models.load_model(path)
        after = ops.convert_to_numpy(restored([pix_feat, mask_logits])[0])

        assert float(np.abs(before - after).max()) == 0.0

    @pytest.mark.parametrize("kwargs, match", [
        ({"stride": 1}, "at least 2"),
        ({"stride": 3, "total_stride": 16}, "exact positive integer power"),
        ({"embed_dim": 0}, "embed_dim must be positive"),
        ({"padding": -1}, "padding must not be negative"),
    ])
    def test_downsampler_rejects_bad_geometry(
            self, kwargs: Dict[str, Any], match: str) -> None:
        base: Dict[str, Any] = {"embed_dim": 8}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            SAM2MaskDownSampler(**base)

    def test_encoder_rejects_a_mismatched_mask_grid(self) -> None:
        layer = SAM2MemoryEncoder(in_dim=IN_DIM, out_dim=OUT_DIM)
        with pytest.raises(ValueError, match="does not match"):
            layer.build([
                (None, FEATURE_GRID, FEATURE_GRID, IN_DIM),
                (None, MASK_GRID * 2, MASK_GRID * 2, 1),
            ])

    def test_encoder_call_traces_with_a_static_input_signature(self) -> None:
        """No dynamic-shape trap in the memory-encoder forward path."""
        layer = SAM2MemoryEncoder(in_dim=IN_DIM, out_dim=OUT_DIM)
        layer.build([
            (None, FEATURE_GRID, FEATURE_GRID, IN_DIM),
            (None, MASK_GRID, MASK_GRID, 1),
        ])

        @tf.function(input_signature=[
            tf.TensorSpec((1, FEATURE_GRID, FEATURE_GRID, IN_DIM), tf.float32),
            tf.TensorSpec((1, MASK_GRID, MASK_GRID, 1), tf.float32),
        ])
        def traced(features: Any, masks: Any) -> Any:
            return layer([features, masks], training=False)

        concrete = traced.get_concrete_function()
        memory_spec = concrete.structured_outputs[0]
        assert tuple(memory_spec.shape) == (
            1, FEATURE_GRID, FEATURE_GRID, OUT_DIM)

    def test_gradients_reach_every_component(
            self,
            pix_feat: np.ndarray,
            mask_logits: np.ndarray,
    ) -> None:
        layer = SAM2MemoryEncoder(in_dim=IN_DIM, out_dim=OUT_DIM)
        layer.build([
            (None, FEATURE_GRID, FEATURE_GRID, IN_DIM),
            (None, MASK_GRID, MASK_GRID, 1),
        ])
        with tf.GradientTape() as tape:
            memory, _ = layer(
                [tf.constant(pix_feat), tf.constant(mask_logits)],
                training=True)
            loss = ops.mean(ops.square(memory))
        gradients = tape.gradient(loss, layer.trainable_variables)
        dead = [
            variable.path
            for variable, gradient in zip(layer.trainable_variables, gradients)
            if gradient is None
        ]
        assert not dead, f"no gradient reaches {dead}"

# ---------------------------------------------------------------------
# G5.5 -- the MEASURED dead-component partition
# ---------------------------------------------------------------------


class TestDeadComponentPartition:
    """The real red/green partition, measured rather than predicted.

    Four steps running, this plan's "all guards go RED under a dead component"
    prediction has been falsified. These tests pin what a zeroed component
    ACTUALLY does, so a future reader does not re-derive the optimistic version.
    """

    def test_zeroing_the_downsampler_output_kills_the_mask_branch_only(
            self,
            encoder: SAM2MemoryEncoder,
            pix_feat: np.ndarray,
            mask_logits: np.ndarray,
    ) -> None:
        """G5.3's liveness arm goes RED; G5.1's value arms do NOT.

        The affine still computes the right quantity -- it is simply multiplied
        by a dead convolution downstream. This is exactly why G5.1 is a VALUE
        oracle rather than a liveness probe.
        """
        killed = list(encoder.mask_downsampler.final_conv.weights)
        with zeroed_variables(killed):
            base = ops.convert_to_numpy(encoder([pix_feat, mask_logits])[0])
            scaled = ops.convert_to_numpy(
                encoder([pix_feat, mask_logits * 2.0])[0])
            # G5.3's liveness assertion would now FAIL.
            assert float(np.abs(base - scaled).max()) == pytest.approx(
                0.0, abs=1e-7)

        # G5.1's value oracle is UNAFFECTED -- it reads the affine directly.
        probe = np.full((1, 1, 1, 1), 0.6, dtype="float32")
        with zeroed_variables(killed):
            value = float(ops.convert_to_numpy(
                encoder._affine_sigmoid(probe)).reshape(-1)[0])
        assert value == pytest.approx(0.880797, abs=1e-6)

    def test_zeroing_the_pixel_projection_leaves_the_mask_branch_alive(
            self,
            encoder: SAM2MemoryEncoder,
            pix_feat: np.ndarray,
            mask_logits: np.ndarray,
    ) -> None:
        """Additive fusion means one dead branch does not kill the other."""
        with zeroed_variables(list(encoder.pix_feat_proj.weights)):
            base = ops.convert_to_numpy(encoder([pix_feat, mask_logits])[0])
            moved_pix = ops.convert_to_numpy(
                encoder([pix_feat + 1.0, mask_logits])[0])
            scaled_mask = ops.convert_to_numpy(
                encoder([pix_feat, mask_logits * 2.0])[0])

        assert float(np.abs(base - moved_pix).max()) == pytest.approx(
            0.0, abs=1e-7), "the pixel branch is not actually dead"
        assert float(np.abs(base - scaled_mask).max()) > 1e-5, (
            "killing the pixel projection also killed the mask branch -- the "
            "fusion is not additive"
        )

    def test_zeroing_the_out_projection_kills_the_memory_but_not_the_encoding(
            self,
            encoder: SAM2MemoryEncoder,
            pix_feat: np.ndarray,
            mask_logits: np.ndarray,
    ) -> None:
        """G5.4 is structurally blind to a dead network.

        The sine encoding is a fixed function of the GRID, not of the values,
        so every width assertion in ``TestPositionEncodingWidth`` stays GREEN
        with the whole encoder zeroed. Recorded so nobody reads G5.4 as
        evidence that the encoder computes anything.
        """
        with zeroed_variables(list(encoder.out_proj.weights)):
            memory, position = encoder([pix_feat, mask_logits])
            memory = ops.convert_to_numpy(memory)
            position = ops.convert_to_numpy(position)

        assert float(np.abs(memory).max()) == 0.0
        assert int(position.shape[-1]) == OUT_DIM
        assert float(np.abs(position).max()) > 0.0
