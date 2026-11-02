"""Vision heads refuse a configuration or input they would run wrongly.

Measured on 2026-11-02 before the fixes (audit of ``layers/heads``):

* ``DepthEstimationHead(min_depth=0.0)`` returned NaN in log mode, and
  ``min_depth=5, max_depth=1`` was accepted.
* Under a global ``mixed_float16`` policy the log mode raised
  ``InvalidArgumentError``: ``keras.ops.log`` of a Python float is float32.
* ``SegmentationHead(skip_channel_mode='explict')`` and
  ``DetectionHead(input_format='seq')`` stored the typo and took another branch.
* ``explicit_skip_channels`` with three entries raised ``IndexError``.
* ``SegmentationHead`` raised ``AttributeError`` on a tuple where a list works.
* With five input maps the first had no effect on the output.
* ``ClassificationHead(pooling_type='cls_token')`` on a feature map returned
  ``(B, H, num_classes)`` while reporting ``(B, num_classes)``.
* ``EnhancementHead`` at scale 4 left 7/16 of its pixels unreached, and at
  scale 8 55/64, because its 3x3 transposed conv is narrower than its stride.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.vision import (
    ClassificationHead,
    DepthEstimationHead,
    DetectionHead,
    EnhancementHead,
    SegmentationHead,
)

HIDDEN_DIM = 16


def _features(*shape):
    return np.random.default_rng(0).random(shape).astype("float32")


@pytest.fixture
def half_precision():
    previous = keras.config.dtype_policy()
    keras.config.set_dtype_policy("mixed_float16")
    try:
        yield
    finally:
        keras.config.set_dtype_policy(previous)


@pytest.mark.parametrize("kwargs", [
    {"min_depth": 0.0},
    {"min_depth": -1.0},
    {"min_depth": 5.0, "max_depth": 1.0},
    {"min_depth": 1.0, "max_depth": 1.0},
    {"min_depth": 0.0, "max_depth": float("inf")},
    {"min_depth": True},
    {"max_depth": "far"},
])
def test_a_log_depth_head_refuses_a_range_it_cannot_scale_into(kwargs) -> None:
    with pytest.raises(ValueError, match="DepthEstimationHead"):
        DepthEstimationHead(hidden_dim=HIDDEN_DIM, **kwargs)


def test_a_linear_depth_range_may_start_at_zero() -> None:
    head = DepthEstimationHead(hidden_dim=HIDDEN_DIM, min_depth=0.0,
                               use_log_depth=False)
    depth = np.array(head(_features(1, 2, 2, 8))["depth"])
    assert np.isfinite(depth).all()
    assert depth.min() >= 0.0


def test_a_linear_depth_range_still_refuses_an_inverted_range() -> None:
    with pytest.raises(ValueError, match="DepthEstimationHead"):
        DepthEstimationHead(hidden_dim=HIDDEN_DIM, min_depth=5.0, max_depth=1.0,
                            use_log_depth=False)


@pytest.mark.parametrize("use_log_depth", [True, False])
def test_a_depth_head_stays_in_range_under_mixed_float16(
        half_precision, use_log_depth) -> None:
    head = DepthEstimationHead(hidden_dim=HIDDEN_DIM, min_depth=0.1,
                               max_depth=1000.0, use_log_depth=use_log_depth)
    depth = np.array(head(_features(1, 2, 2, 8))["depth"], dtype="float32")
    assert np.isfinite(depth).all()
    assert depth.min() >= 0.09
    assert depth.max() <= 1001.0


def test_a_depth_head_refuses_a_range_float16_cannot_hold(half_precision) -> None:
    with pytest.raises(ValueError, match="float16"):
        DepthEstimationHead(hidden_dim=HIDDEN_DIM, max_depth=1e5)


@pytest.mark.parametrize("make", [
    lambda: SegmentationHead(num_classes=2, hidden_dim=HIDDEN_DIM,
                             skip_channel_mode="explict"),
    lambda: SegmentationHead(num_classes=2, hidden_dim=HIDDEN_DIM,
                             input_format="seq"),
    lambda: DetectionHead(num_classes=2, hidden_dim=HIDDEN_DIM,
                          input_format="seq"),
], ids=["skip_mode", "segmentation_format", "detection_format"])
def test_a_misspelled_mode_is_refused(make) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        make()


@pytest.mark.parametrize("channels", [[8, 8, 8], [1, 8, 8, 8], [8, 8, 0, 8]])
def test_unusable_explicit_skip_channels_are_refused(channels) -> None:
    with pytest.raises(ValueError, match="explicit_skip_channels"):
        SegmentationHead(num_classes=2, skip_channel_mode="explicit",
                         explicit_skip_channels=channels, use_ffn=False)


def test_a_tuple_of_maps_runs_as_a_list_does() -> None:
    maps = [_features(1, 8, 8, 8), _features(1, 8, 8, 8)]
    head = SegmentationHead(num_classes=2, hidden_dim=HIDDEN_DIM, use_ffn=False)
    np.testing.assert_array_equal(np.array(head(tuple(maps))), np.array(head(maps)))


def test_more_skips_than_refine_blocks_are_refused() -> None:
    head = SegmentationHead(num_classes=2, hidden_dim=HIDDEN_DIM, use_ffn=False)
    with pytest.raises(ValueError, match="3 refine blocks"):
        head([_features(1, 8, 8, 8) for _ in range(5)])


def test_three_skips_all_reach_the_output() -> None:
    maps = [_features(1, 8, 8, 8) for _ in range(4)]
    head = SegmentationHead(num_classes=2, hidden_dim=HIDDEN_DIM, use_ffn=False)
    base = np.array(head(maps))
    for index in range(4):
        changed = list(maps)
        changed[index] = changed[index] + 100.0
        assert np.abs(np.array(head(changed)) - base).max() > 0, index


@pytest.mark.parametrize("pooling_type,shape", [
    ("cls_token", (2, 4, 4, 8)),
    ("avg", (2, 4, 8)),
    ("max", (2, 4, 8)),
])
def test_pooling_refuses_an_input_rank_it_cannot_reduce(pooling_type, shape) -> None:
    head = ClassificationHead(num_classes=3, hidden_dim=HIDDEN_DIM,
                              pooling_type=pooling_type)
    with pytest.raises(ValueError, match=f"pooling_type='{pooling_type}'"):
        head(_features(*shape))


def test_a_cls_token_head_built_at_the_pooled_shape_runs_on_the_sequence() -> None:
    """ViT, ViT-HMLP and BEiT build at ``(None, D)`` and call with the sequence."""
    head = ClassificationHead(num_classes=3, hidden_dim=HIDDEN_DIM,
                              pooling_type="cls_token")
    head.build((None, 8))
    assert head(_features(2, 4, 8))["logits"].shape == (2, 3)


def test_a_cls_token_head_built_at_the_pooled_shape_still_refuses_a_map() -> None:
    head = ClassificationHead(num_classes=3, hidden_dim=HIDDEN_DIM,
                              pooling_type="cls_token")
    head.build((None, 8))
    with pytest.raises(ValueError, match="pooling_type='cls_token'"):
        head(_features(2, 4, 4, 8))


def test_the_flatten_path_owns_its_flatten_layer() -> None:
    head = ClassificationHead(num_classes=3, hidden_dim=HIDDEN_DIM,
                              use_global_pooling=False)
    outputs = head(_features(2, 4, 4, 8))
    assert outputs["logits"].shape == (2, 3)
    assert head.flatten.built
    assert head.flatten.name == "flatten"


@pytest.mark.parametrize("scale", [2, 3, 4, 5, 8])
def test_upsampling_reaches_every_output_pixel(scale) -> None:
    head = EnhancementHead(scale_factor=scale, block_dims=(), output_channels=1,
                           output_use_bias=False)
    enhanced = np.array(head(np.ones((1, 8, 8, 4), "float32"))["enhanced"])
    assert enhanced.shape == (1, 8 * scale, 8 * scale, 1)
    assert np.mean(enhanced == 0.0) == 0.0


@pytest.mark.parametrize("scale,kernel", [(1, None), (2, 3), (3, 3), (4, 4), (8, 8)])
def test_the_upsample_kernel_is_unchanged_up_to_scale_three(scale, kernel) -> None:
    head = EnhancementHead(scale_factor=scale, hidden_dim=HIDDEN_DIM)
    if kernel is None:
        assert head.output_conv.kernel_size == (3, 3)
    else:
        assert head.upsample.kernel_size == (kernel, kernel)
