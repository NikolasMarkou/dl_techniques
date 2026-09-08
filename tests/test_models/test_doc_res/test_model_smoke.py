"""Assembly guards for ``doc_res/model.py``.

This is the SMOKE suite for step 6: it exists to prove the backbone is wired
the way the reference implementation is wired. The full v2 arms (round-trip,
gradient flow, precision, knob sensitivity) arrive with the step-8 suite; what
is here is the set of claims that a shape test cannot see.

The load-bearing instrument is :func:`analytic_parameter_count` -- a closed
form written from the upstream PyTorch architecture, NOT read off this port's
``model.py``. It encodes all six of the asymmetries the port must reproduce, so
a single equality against ``model.count_params()`` fails if ANY of them is
symmetrized: adding the missing ``reduce_chan_level1`` changes it, running
``refinement`` at ``dim`` instead of ``2 * dim`` changes it, dropping
``skip_conv`` changes it, and switching the default depth changes it. The
structural tests below then localize which one broke, because an equality
failure alone says only "something is wrong".
"""

import inspect

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.image_restoration.doc_res.model import (
    SPATIAL_DIVISOR,
    DocRes,
    create_doc_res,
)

# ---------------------------------------------------------------------
# The independent oracle
# ---------------------------------------------------------------------


def _block_params(dim: int, num_heads: int, ffn_expansion_factor: float) -> int:
    """Parameters of one Restormer transformer block, from the paper's spec.

    Two ``LayerNorm``s (gamma and beta, one pair per channel); MDTA's ``1x1``
    qkv projection, its fully-depthwise ``3x3``, its ``1x1`` output projection
    and its per-head temperature; GDFN's ``1x1`` in-projection to twice the
    truncated hidden width, its fully-depthwise ``3x3`` on that doubled width,
    and its ``1x1`` out-projection from the halved (post-gating) width.
    """
    hidden = int(dim * ffn_expansion_factor)
    norms = 2 * (2 * dim)
    mdta = dim * (3 * dim) + 9 * (3 * dim) + dim * dim + num_heads
    gdfn = dim * (2 * hidden) + 9 * (2 * hidden) + hidden * dim
    return norms + mdta + gdfn


def analytic_parameter_count(
        dim: int = 48,
        num_blocks: tuple = (2, 3, 3, 4),
        num_refinement_blocks: int = 4,
        heads: tuple = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        input_channels: int = 6,
        output_channels: int = 3,
) -> int:
    """Total parameters of the Restormer/DocRes backbone, derived not measured.

    Every convolution in the backbone is bias-free, so each contributes exactly
    ``k*k*C_in*C_out``. ``Downsample(n)`` is a ``3x3`` from ``n`` to ``n//2``;
    ``Upsample(n)`` is a ``3x3`` from ``n`` to ``2n``; the pixel-(un)shuffles
    themselves carry no weights.
    """
    level = [int(dim * 2 ** k) for k in range(4)]

    total = 9 * input_channels * level[0]                 # patch_embed 3x3

    for k in range(3):                                    # encoder levels 1-3
        total += num_blocks[k] * _block_params(
            level[k], heads[k], ffn_expansion_factor)
        total += 9 * level[k] * (level[k] // 2)            # down(k -> k+1)

    total += num_blocks[3] * _block_params(               # latent
        level[3], heads[3], ffn_expansion_factor)

    total += 9 * level[3] * (2 * level[3])                # up4_3
    total += (2 * level[2]) * level[2]                    # reduce_chan_level3
    total += num_blocks[2] * _block_params(
        level[2], heads[2], ffn_expansion_factor)

    total += 9 * level[2] * (2 * level[2])                # up3_2
    total += (2 * level[1]) * level[1]                    # reduce_chan_level2
    total += num_blocks[1] * _block_params(
        level[1], heads[1], ffn_expansion_factor)

    total += 9 * level[1] * (2 * level[1])                # up2_1
    # NO reduce_chan_level1: decoder level 1 and refinement run at level[1].
    total += num_blocks[0] * _block_params(
        level[1], heads[0], ffn_expansion_factor)
    total += num_refinement_blocks * _block_params(
        level[1], heads[0], ffn_expansion_factor)

    total += level[0] * level[1]                          # skip_conv 1x1
    total += 9 * level[1] * output_channels               # output_conv 3x3
    return total


@pytest.fixture(scope="module")
def built_model() -> DocRes:
    """A symbolically-built ``docres`` model, shared across the shape tests."""
    model = create_doc_res("docres")
    model.build((None, None, None, 6))
    return model


# ---------------------------------------------------------------------
# Shape and parameter contract
# ---------------------------------------------------------------------


def test_the_forward_pass_maps_six_channels_to_three(built_model):
    inputs = np.random.default_rng(0).random((2, 64, 64, 6)).astype("float32")
    outputs = keras.ops.convert_to_numpy(built_model(inputs))

    assert outputs.shape == (2, 64, 64, 3)
    assert np.isfinite(outputs).all()


def test_compute_output_shape_agrees_with_the_forward_pass(built_model):
    assert built_model.compute_output_shape((None, 64, 64, 6)) == (
        None, 64, 64, 3)


@pytest.mark.parametrize("variant", ["docres", "restormer_base"])
def test_the_parameter_count_matches_the_independent_derivation(variant):
    model = DocRes.from_variant(variant)
    model.build((None, None, None, 6))

    config = dict(DocRes.MODEL_VARIANTS[variant])
    config.pop("description")
    config.pop("use_bias")
    expected = analytic_parameter_count(
        dim=config["dim"],
        num_blocks=tuple(config["num_blocks"]),
        num_refinement_blocks=config["num_refinement_blocks"],
        heads=tuple(config["heads"]),
        ffn_expansion_factor=config["ffn_expansion_factor"],
        input_channels=config["input_channels"],
        output_channels=config["output_channels"],
    )

    assert model.count_params() == expected


def test_the_shipped_docres_count_is_pinned_absolutely():
    """Anti-vacuity for the test above: pin the number the oracle produces.

    If the oracle and the model drifted together the equality test would still
    pass; this one would not.
    """
    assert analytic_parameter_count() == 15_203_680
    assert analytic_parameter_count(num_blocks=(4, 6, 6, 8)) == 26_132_548


def test_every_level_holds_the_per_head_channel_width_at_dim(built_model):
    """A Restormer invariant: ``dim * 2**k / heads[k]`` is constant at 48."""
    widths = {
        int(built_model.dim * 2 ** k) // built_model.heads[k]
        for k in range(4)
    }
    assert widths == {built_model.dim}


# ---------------------------------------------------------------------
# The six asymmetries
# ---------------------------------------------------------------------


def test_asymmetry_1_the_default_depth_is_the_docres_schedule():
    """[2,3,3,4], not the Restormer paper's [4,6,6,8] class default."""
    assert DocRes().num_blocks == [2, 3, 3, 4]
    assert DocRes.MODEL_VARIANTS["docres"]["num_blocks"] == [2, 3, 3, 4]
    assert DocRes.MODEL_VARIANTS["restormer_base"]["num_blocks"] == [4, 6, 6, 8]


def test_asymmetry_2_there_is_no_reduce_chan_level_one(built_model):
    """Levels 3 and 2 halve the concatenated width back; level 1 does not."""
    assert not hasattr(built_model, "reduce_chan_level1")
    assert built_model.reduce_chan_level3.filters == 192
    assert built_model.reduce_chan_level2.filters == 96

    # The observable consequence: decoder level 1 runs at 2*dim, not dim.
    assert [block.dim for block in built_model.decoder_level1] == [96, 96]


def test_asymmetry_3_refinement_runs_at_double_dim_with_level_one_heads(
        built_model):
    assert [block.dim for block in built_model.refinement] == [96] * 4
    assert [block.num_heads for block in built_model.refinement] == [1] * 4


def test_asymmetry_4_skip_conv_is_unconditional(built_model):
    """No ``dual_pixel_task`` knob, and the projection is really in ``call``.

    ``build()`` materializes sub-layers by tracing ``call``, so a ``skip_conv``
    that ``call`` never applied would still be unbuilt and hold no weights.
    Its weights existing is therefore evidence about the forward path, not
    merely about ``__init__``.
    """
    assert "dual_pixel_task" not in inspect.signature(DocRes.__init__).parameters
    assert built_model.skip_conv.filters == 96
    assert built_model.skip_conv.built
    assert len(built_model.skip_conv.weights) == 1


def test_asymmetry_5_there_is_no_input_residual():
    """Zeroing ``output_conv`` must make the whole network output exactly 0.

    If the port had added the usual ``+ inp_img`` restoration residual, the
    output would instead be the first three input channels. The control below
    shows the un-zeroed model does NOT output zeros, so this is not passing
    for a trivial reason.
    """
    model = create_doc_res("docres")
    inputs = (
        np.random.default_rng(1).random((1, 32, 32, 6)).astype("float32") + 1.0
    )

    before = keras.ops.convert_to_numpy(model(inputs))
    assert np.abs(before).max() > 0.0

    model.output_conv.kernel.assign(
        keras.ops.zeros_like(model.output_conv.kernel))
    after = keras.ops.convert_to_numpy(model(inputs))

    np.testing.assert_array_equal(after, np.zeros_like(after))


def test_asymmetry_6_decoder_levels_reuse_their_encoder_head_counts(
        built_model):
    assert {b.num_heads for b in built_model.decoder_level3} == {4}
    assert {b.num_heads for b in built_model.decoder_level2} == {2}
    assert {b.num_heads for b in built_model.decoder_level1} == {1}


# ---------------------------------------------------------------------
# Variant API
# ---------------------------------------------------------------------


def test_an_unknown_variant_names_the_available_keys():
    with pytest.raises(ValueError, match="Unknown DocRes variant") as excinfo:
        DocRes.from_variant("docres_large")

    message = str(excinfo.value)
    assert "docres" in message
    assert "restormer_base" in message


def test_pretrained_raises_and_names_the_variant():
    with pytest.raises(NotImplementedError, match="restormer_base"):
        DocRes.from_variant("restormer_base", pretrained=True)


def test_the_factory_delegates_and_forwards_overrides():
    model = create_doc_res("docres", num_refinement_blocks=1)

    assert isinstance(model, DocRes)
    assert model.num_refinement_blocks == 1
    # The override must not have mutated the shared variant table.
    assert DocRes.MODEL_VARIANTS["docres"]["num_refinement_blocks"] == 4


def test_get_config_round_trips_every_constructor_argument():
    model = create_doc_res("docres", num_refinement_blocks=2)
    config = model.get_config()

    constructor_args = set(
        inspect.signature(DocRes.__init__).parameters) - {"self", "kwargs"}
    assert constructor_args <= set(config)

    clone = DocRes.from_config(config)
    assert clone.get_config() == config


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"num_blocks": [2, 3, 3]}, "exactly 4 entries"),
        ({"heads": [1, 2, 4]}, "exactly 4 entries"),
        ({"dim": 0}, "must be positive"),
        ({"dim": 49}, "must be even"),
        ({"heads": [1, 2, 4, 7]}, "not divisible by its head count"),
        ({"num_refinement_blocks": 0}, "must be positive"),
    ],
)
def test_an_unbuildable_configuration_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DocRes(**kwargs)


# ---------------------------------------------------------------------
# Build contract
# ---------------------------------------------------------------------


def test_the_model_builds_symbolically_at_fully_dynamic_spatial_dims():
    model = create_doc_res("docres")
    model.build((None, None, None, 6))

    assert model.built
    assert model.count_params() == analytic_parameter_count()


def test_explicit_build_matches_lazy_build():
    def relative(model):
        return sorted(w.path.split("/", 1)[-1] for w in model.weights)

    explicit = create_doc_res("docres")
    explicit.build((None, None, None, 6))

    lazy = create_doc_res("docres")
    lazy(np.zeros((1, 32, 32, 6), "float32"))

    assert relative(explicit) == relative(lazy)
    assert len(explicit.weights) > 0


def test_a_wrong_channel_count_is_refused_at_build():
    with pytest.raises(ValueError, match="Expected 6 input channels"):
        create_doc_res("docres").build((None, 64, 64, 3))


# ---------------------------------------------------------------------
# The divisibility contract
# ---------------------------------------------------------------------


@pytest.mark.parametrize("height,width", [(63, 64), (64, 60), (12, 12)])
def test_a_non_multiple_of_eight_fails_loudly(height, width):
    """A deliberate refusal, not internal padding. See D-016."""
    model = create_doc_res("docres")

    with pytest.raises(ValueError, match="divisible by 8"):
        model(np.zeros((1, height, width, 6), "float32"))


def test_the_refusal_survives_a_prior_legal_call():
    """The check must live in ``call``, not only in ``build``.

    On a FRESH model the test above is satisfied by ``build``'s copy of the
    check alone -- measured: deleting the ``call``-site check leaves it green.
    Once the model is built (here by a legal call; equally by a symbolic
    ``build((None, None, None, 6))``, where the extents are unknown and
    ``build`` cannot check anything), ``call`` is the only line of defence.
    """
    model = create_doc_res("docres")
    model(np.zeros((1, 32, 32, 6), "float32"))
    assert model.built

    with pytest.raises(ValueError, match="divisible by 8"):
        model(np.zeros((1, 36, 32, 6), "float32"))


def test_a_multiple_of_eight_is_accepted():
    """Control for the test above: 8 is the real boundary, not 32 or 64."""
    model = create_doc_res("docres")
    outputs = model(np.zeros((1, 24, 40, 6), "float32"))

    assert outputs.shape == (1, 24, 40, 3)
    assert SPATIAL_DIVISOR == 8
