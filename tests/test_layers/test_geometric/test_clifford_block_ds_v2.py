"""Tests for ``CliffordNetBlockDSv2`` (downsampling design-space sibling)."""

from __future__ import annotations

import os

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.blur_pool import BlurPool2D
from dl_techniques.layers.pixel_unshuffle import PixelUnshuffle2D
from dl_techniques.layers.geometric.clifford_block import (
    CliffordNetBlockDSv2,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def input_4d() -> tf.Tensor:
    return tf.random.uniform((2, 32, 32, 32), seed=0)


# ---------------------------------------------------------------------
# Init / validation
# ---------------------------------------------------------------------


def test_default_init_uses_v1_winner():
    """Defaults reflect the V1 winner of the downsampling sweep
    (see src/train/cliffordnet/DOWNSAMPLING.md): stream_pool=blur,
    skip_pool=blur."""
    blk = CliffordNetBlockDSv2(channels=32, shifts=[1, 2])
    assert blk.channels == 32
    assert blk.stream_pool_kind == "blur"
    assert blk.skip_pool_kind == "blur"
    assert blk.out_channels is None
    assert blk.ctx_norm_type == "bn"
    assert blk.ctx_mode == "diff"
    assert blk.strides == 1


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"ctx_mode": "garbage"},
        {"stream_pool": "garbage"},
        {"skip_pool": "garbage"},
        {"ctx_norm_type": "garbage"},
        {"strides": 0},
        {"kernel_size": 0},
        {"out_channels": 0},
    ],
)
def test_invalid_kwargs_raise(bad_kwargs):
    with pytest.raises(ValueError):
        CliffordNetBlockDSv2(channels=32, shifts=[1, 2], **bad_kwargs)


def test_invalid_channels_raises():
    with pytest.raises(ValueError):
        CliffordNetBlockDSv2(channels=0, shifts=[1, 2])


# ---------------------------------------------------------------------
# Forward shape per pool kind (axes A and B)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "stream_pool,skip_pool",
    [
        ("avg", "avg"),
        ("max", "max"),
        ("blur", "blur"),
        ("blur", "pixel_unshuffle"),
        ("gaussian_dw", "pixel_unshuffle"),
        ("avg", "resnetd"),
        ("resnetd", "resnetd"),
    ],
)
def test_pool_combinations_strides_2(input_4d, stream_pool, skip_pool):
    blk = CliffordNetBlockDSv2(
        channels=32,
        shifts=[1, 2],
        strides=2,
        stream_pool=stream_pool,
        skip_pool=skip_pool,
    )
    y = blk(input_4d)
    assert tuple(y.shape) == (2, 16, 16, 32)


def test_strides_1_identity_shape(input_4d):
    blk = CliffordNetBlockDSv2(
        channels=32, shifts=[1, 2], strides=1,
        stream_pool="blur", skip_pool="pixel_unshuffle",
    )
    y = blk(input_4d)
    assert tuple(y.shape) == (2, 32, 32, 32)


# ---------------------------------------------------------------------
# Internal channel expansion (axis E)
# ---------------------------------------------------------------------


def test_out_channels_expansion(input_4d):
    blk = CliffordNetBlockDSv2(
        channels=32,
        shifts=[1, 2],
        strides=2,
        stream_pool="blur",
        skip_pool="pixel_unshuffle",
        out_channels=64,
    )
    # Transform-only contract (plan_2026-07-03_eb53492e, D-002): call() returns
    # h_mix at `channels` width; the channels->out_channels projection (out_proj)
    # is applied externally by the model, POST-SUM.
    h = blk(input_4d)
    assert tuple(h.shape) == (2, 16, 16, 32)
    assert blk.out_proj is not None
    out = blk.out_proj(blk.skip_pool(input_4d) + blk(input_4d))
    assert tuple(out.shape) == (2, 16, 16, 64)


def test_out_channels_equal_channels_no_proj(input_4d):
    blk = CliffordNetBlockDSv2(
        channels=32, shifts=[1, 2], strides=2, out_channels=32,
    )
    assert blk.out_proj is None
    y = blk(input_4d)
    assert tuple(y.shape) == (2, 16, 16, 32)


# ---------------------------------------------------------------------
# Norm type (axis G)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("norm_type", ["bn", "gn", "ln", "none"])
def test_ctx_norm_types(input_4d, norm_type):
    blk = CliffordNetBlockDSv2(
        channels=32, shifts=[1, 2], strides=2,
        ctx_norm_type=norm_type,
    )
    y = blk(input_4d)
    assert tuple(y.shape) == (2, 16, 16, 32)


def test_gn_groups_divides_channels():
    """Group count must divide the channel dim."""
    blk = CliffordNetBlockDSv2(
        channels=96, shifts=[1, 2], strides=2, ctx_norm_type="gn",
    )
    # ctx_norm exists and is a GroupNormalization
    assert isinstance(blk.ctx_norm, keras.layers.GroupNormalization)
    assert 96 % blk.ctx_norm.groups == 0


def test_norm_none_dwconv_has_bias():
    blk = CliffordNetBlockDSv2(
        channels=32, shifts=[1, 2], ctx_norm_type="none",
    )
    assert blk.ctx_norm is None
    assert blk.dw_conv.use_bias is True


# ---------------------------------------------------------------------
# ctx_mode (axis D)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["diff", "abs", "pyramid_diff"])
def test_ctx_modes_strides_2(input_4d, mode):
    blk = CliffordNetBlockDSv2(
        channels=32, shifts=[1, 2], strides=2, ctx_mode=mode,
    )
    y = blk(input_4d)
    assert tuple(y.shape) == (2, 16, 16, 32)


def test_pyramid_diff_falls_back_at_strides_1(input_4d):
    """At strides=1, pyramid_diff has no pyramid layers; uses plain diff."""
    blk = CliffordNetBlockDSv2(
        channels=32, shifts=[1, 2], strides=1, ctx_mode="pyramid_diff",
    )
    assert blk._pyr_pool is None and blk._pyr_up is None
    y = blk(input_4d)
    assert tuple(y.shape) == (2, 32, 32, 32)


# ---------------------------------------------------------------------
# V12 negative control: k=3, s=2 (violates k>=2s) must build, not raise
# ---------------------------------------------------------------------


def test_v12_negative_control_builds(input_4d):
    blk = CliffordNetBlockDSv2(
        channels=32,
        shifts=[1, 2],
        kernel_size=3,
        strides=2,
        stream_pool="blur",
        skip_pool="pixel_unshuffle",
    )
    y = blk(input_4d)
    assert tuple(y.shape) == (2, 16, 16, 32)


# ---------------------------------------------------------------------
# Defaults produce an isotropic-shape output at strides=1
# ---------------------------------------------------------------------


def test_defaults_match_ds_at_strides_1(input_4d):
    """At strides=1 with all defaults, DSv2 must produce a tensor with the
    same shape as its input (isotropic reduction)."""
    v2 = CliffordNetBlockDSv2(channels=32, shifts=[1, 2])
    y = v2(input_4d)
    assert tuple(y.shape) == tuple(input_4d.shape)


# ---------------------------------------------------------------------
# Compute output shape
# ---------------------------------------------------------------------


def test_compute_output_shape_matches_call():
    blk = CliffordNetBlockDSv2(
        channels=32, shifts=[1, 2], strides=2, out_channels=64,
    )
    x = tf.random.uniform((2, 32, 32, 32), seed=0)
    y = blk(x)
    inferred = blk.compute_output_shape(tuple(x.shape))
    assert tuple(y.shape) == inferred


# ---------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        # V0 baseline
        dict(strides=2, stream_pool="avg", skip_pool="avg"),
        # V2 high-priority
        dict(strides=2, stream_pool="blur", skip_pool="pixel_unshuffle"),
        # V4: V2 + internal expansion
        dict(
            strides=2,
            stream_pool="blur",
            skip_pool="pixel_unshuffle",
            out_channels=64,
        ),
        # V5: + GroupNorm
        dict(
            strides=2,
            stream_pool="blur",
            skip_pool="pixel_unshuffle",
            out_channels=64,
            ctx_norm_type="gn",
        ),
        # V6: + pyramid_diff
        dict(
            strides=2,
            stream_pool="blur",
            skip_pool="pixel_unshuffle",
            out_channels=64,
            ctx_mode="pyramid_diff",
        ),
        # V10: ResNet-D both paths
        dict(strides=2, stream_pool="resnetd", skip_pool="resnetd"),
        # V12: k=3 s=2 negative control
        dict(
            kernel_size=3,
            strides=2,
            stream_pool="blur",
            skip_pool="pixel_unshuffle",
        ),
    ],
)
def test_save_load_round_trip(tmp_path, kwargs):
    inp = keras.layers.Input(shape=(32, 32, 32))
    blk = CliffordNetBlockDSv2(channels=32, shifts=[1, 2], **kwargs)
    out = blk(inp)
    model = keras.Model(inp, out)

    x = np.random.RandomState(0).randn(1, 32, 32, 32).astype(np.float32)
    pre = model.predict(x, verbose=0)

    path = os.path.join(tmp_path, "v2.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    post = loaded.predict(x, verbose=0)
    np.testing.assert_allclose(pre, post, atol=1e-5)


# ---------------------------------------------------------------------
# get_config faithful
# ---------------------------------------------------------------------


def test_get_config_round_trip():
    blk = CliffordNetBlockDSv2(
        channels=64,
        shifts=[1, 2, 3],
        strides=2,
        stream_pool="gaussian_dw",
        skip_pool="pixel_unshuffle",
        out_channels=128,
        ctx_norm_type="gn",
        ctx_mode="pyramid_diff",
        kernel_size=5,
        layer_scale_init=1e-2,
    )
    cfg = blk.get_config()
    rebuilt = CliffordNetBlockDSv2.from_config(cfg)
    assert rebuilt.channels == 64
    assert rebuilt.shifts == [1, 2, 3]
    assert rebuilt.strides == 2
    assert rebuilt.stream_pool_kind == "gaussian_dw"
    assert rebuilt.skip_pool_kind == "pixel_unshuffle"
    assert rebuilt.out_channels == 128
    assert rebuilt.ctx_norm_type == "gn"
    assert rebuilt.ctx_mode == "pyramid_diff"
    assert rebuilt.kernel_size == 5
    assert rebuilt.layer_scale_init == 1e-2


# ---------------------------------------------------------------------
# SC4: DSv2 external POST-SUM orchestration parity guard
# (plan_2026-07-03_eb53492e, D-002)
# ---------------------------------------------------------------------


def test_dsv2_post_sum_orchestration_guard():
    """SC4: the NEW external orchestration ``out_proj(skip_pool(x) + block(x))``
    (using the block's own public ``skip_pool``/``out_proj`` and the transform-only
    ``block(x)``) yields a finite tensor of shape ``(B, H/2, W/2, out_channels)``.

    The pre-change monolithic call() is gone, so we cannot compare against it
    directly. Instead we assert (a) shape correctness, (b) finiteness, and
    (c) that out_proj is applied POST-SUM: projecting the summed tensor differs
    from projecting each branch separately (out_proj uses bias -> non-distributive),
    which pins the sum-before-projection order.

    THE BIAS MUST BE MADE NON-ZERO FIRST, AND THAT IS THE WHOLE TEST.
    `out_proj` is a `Conv2D` whose bias initializes to ZEROS. With a zero bias the
    projection IS distributive -- `W(a+b) == Wa + Wb` exactly -- so `out` and `wrong` are
    mathematically IDENTICAL and this test cannot detect the thing it claims to test.

    It nonetheless "passed" for a long time, on GPU only, purely on TF32 rounding noise:
    measured `max|out - wrong|` = 2.6e-4 with TF32 enabled (the GPU default) versus 2.4e-7
    with true fp32 -- and the assertion's tolerance is 1e-4. So the guard was passing on
    numerical garbage that happened to clear the threshold. It began failing in full-suite
    runs the moment another test module called
    `tf.config.experimental.enable_tensor_float_32_execution(False)` -- a PROCESS-WIDE
    setting with no teardown -- which removed the noise it was living on. The test was
    always broken; TF32 was masking it.

    Assigning a real bias makes `out - wrong == -bias`, an actual signal of the property
    under test, deterministic and independent of TF32, GPU/CPU and test ordering.
    """
    keras.utils.set_random_seed(0)
    blk = CliffordNetBlockDSv2(
        channels=32, shifts=[1, 2], strides=2,
        stream_pool="avg", skip_pool="avg", out_channels=64,
    )
    x = tf.random.uniform((2, 32, 32, 32), seed=0)

    # Build the block + its skip_pool / out_proj sub-layers.
    h_mix = blk(x)
    assert tuple(h_mix.shape) == (2, 16, 16, 32)  # transform-only at `channels`
    assert blk.out_proj is not None

    # Give the bias a real value -- see the docstring. Without this the two orderings are
    # algebraically equal and the assertion below is vacuous.
    bias = blk.out_proj.bias
    bias.assign(keras.ops.ones_like(bias) * 0.5)

    # External POST-SUM orchestration via the block's own public sub-layers.
    summed = blk.skip_pool(x) + blk(x)
    out = blk.out_proj(summed)
    assert tuple(out.shape) == (2, 16, 16, 64)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    # POST-SUM (project the sum) must differ from projecting each branch separately then
    # summing -- the bias is double-counted in the latter, so the two differ by exactly
    # one bias (0.5). Assert the MEASURED gap, not merely "not close": a bare
    # `not allclose` is satisfiable by float noise, which is precisely how this guard used
    # to pass while testing nothing.
    wrong = blk.out_proj(blk.skip_pool(x)) + blk.out_proj(blk(x))
    gap = np.abs(
        keras.ops.convert_to_numpy(out) - keras.ops.convert_to_numpy(wrong)
    ).max()
    assert np.isclose(gap, 0.5, atol=1e-3), (
        f"out_proj must be applied POST-SUM, not per-branch: expected the two orderings "
        f"to differ by exactly one bias (0.5), measured {gap:.6g}. A gap near 0 means "
        f"out_proj was applied per-branch (or the bias is zero, which makes this test "
        f"vacuous)."
    )
