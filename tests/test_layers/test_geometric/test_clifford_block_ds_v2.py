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
    CliffordNetBlockDS,
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
    skip_pool=blur. Other knobs unchanged from CliffordNetBlockDS."""
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
    y = blk(input_4d)
    assert tuple(y.shape) == (2, 16, 16, 64)


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
# Defaults match CliffordNetBlockDS at the same surface
# ---------------------------------------------------------------------


def test_defaults_match_ds_at_strides_1(input_4d):
    """At strides=1 with all defaults, v2 must produce a tensor with the
    same shape as the v1 block. (Forward values need not match — different
    weight init seeds.)"""
    v2 = CliffordNetBlockDSv2(channels=32, shifts=[1, 2])
    y = v2(input_4d)
    assert tuple(y.shape) == tuple(input_4d.shape)
    # And v1 stays untouched / still works:
    v1 = CliffordNetBlockDS(channels=32, shifts=[1, 2])
    y1 = v1(input_4d)
    assert tuple(y1.shape) == tuple(input_4d.shape)


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
# Backward compatibility: legacy CliffordNetBlockDS still works untouched
# ---------------------------------------------------------------------


def test_v1_block_still_works(input_4d):
    blk = CliffordNetBlockDS(
        channels=32, shifts=[1, 2], strides=2, skip_pool="avg",
    )
    y = blk(input_4d)
    assert tuple(y.shape) == (2, 16, 16, 32)
