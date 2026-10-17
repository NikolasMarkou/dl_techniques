"""DepthAnything's depth is the reference's: a ReLU'd, non-negative map with no
range knobs (plan-2026-10-14T062616-81938e81 step 7.1, decision D-031).

Re-derived from the official code (fetched 2026-10-15):

* Depth Anything V1, ``depth_anything/dpt.py``: ``output_conv2`` ends
  ``Conv2d(head_features_2, 1, 1) -> nn.ReLU(True) -> nn.Identity()``, and
  ``DPT_DINOv2.forward`` applies ``F.relu`` again after the bilinear resize.
* Depth Anything V2, ``depth_anything_v2/dpt.py``: the same ``nn.ReLU`` tail and
  ``F.relu`` after the resize.
* Only V2's metric head (``metric_depth/depth_anything_v2/dpt.py``) maps the
  output into a range, ``nn.Sigmoid()`` times ``max_depth``; V1's metric models
  are ZoeDepth metric bins. Neither is a ``[min_depth, max_depth]`` log or
  linear rescale, and this package claims relative (affine-invariant) depth.

Before this step the decoder was linear (MEASURED at c39e444a3: min -0.25 at
init on the placeholder subject below) and received ``min_depth=0.1``,
``max_depth=100.0`` and ``use_log_depth=True``, which the DPT decoder never
applied. The value pin compares the shipped output with the ReLU of the same
model's linear output, so it fails if the ReLU goes or moves.

Since step 8 of iteration 2 (D-128) the decoder is ``DPTHead``, the reference
head itself: its ``output_conv2_2`` carries the ``nn.ReLU`` of ``output_conv2``,
and ``DepthAnything`` applies the ``F.relu`` after the resize.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.depth_anything import DepthAnything
from dl_techniques.models.vision.depth_anything.components import DPTHead


IMAGE_SHAPE = (32, 32, 3)


def _model(**kwargs) -> DepthAnything:
    keras.utils.set_random_seed(0)
    m = DepthAnything(
        encoder_kind="placeholder", encoder_type="vit_s",
        image_shape=IMAGE_SHAPE, dpt_features=8, dpt_out_channels=[8, 8, 8, 8],
        use_feature_alignment=False, **kwargs,
    )
    _ = m(keras.ops.zeros((1,) + IMAGE_SHAPE))
    return m


def _image() -> np.ndarray:
    return np.random.default_rng(1).uniform(
        0.0, 1.0, (2,) + IMAGE_SHAPE).astype("float32")


def test_the_depth_is_the_relu_of_the_output_conv() -> None:
    model = _model()
    conv = model.decoder.output_conv2_2
    assert conv.activation is keras.activations.relu
    x = _image()
    depth = keras.ops.convert_to_numpy(model(x, training=False))
    conv.activation = keras.activations.linear
    try:
        linear = keras.ops.convert_to_numpy(
            model.decoder(model._encoder_layers(model.encoder, x, False)))
    finally:
        conv.activation = keras.activations.relu
    assert linear.min() < 0.0 < linear.max(), (
        "degenerate probe: the linear output never goes negative, so a "
        "missing ReLU would pass")
    np.testing.assert_array_equal(depth, np.maximum(linear, 0.0))
    assert depth.min() == 0.0


def test_the_decoder_carries_no_range_knobs() -> None:
    decoder = _model().decoder
    config = decoder.get_config()
    for key in ("min_depth", "max_depth", "use_log_depth"):
        assert not hasattr(decoder, key) and key not in config, key
    with pytest.raises((TypeError, ValueError)):
        DPTHead(min_depth=0.1)


def _convs(layer):
    return [sub for sub in layer._flatten_layers(include_self=False)
            if isinstance(sub, (keras.layers.Conv2D, keras.layers.Conv2DTranspose))]


def test_the_model_initializer_and_regularizer_reach_the_decoder() -> None:
    model = _model(kernel_initializer=keras.initializers.Constant(0.125),
                   kernel_regularizer=keras.regularizers.L2(1e-4))
    convs = _convs(model.decoder)
    assert len(convs) == 32  # 4 projects, 3 resize, 4 layer_rn, 5 x 3 + 3 in refinenets, 3 output
    for conv in convs:
        assert np.all(keras.ops.convert_to_numpy(conv.kernel) == 0.125), conv.name
    assert len(model.decoder.losses) == len(convs)


def test_without_the_head_no_decoder_is_created() -> None:
    """call() returns before the decoder when include_top=False, so the model
    owns none (an unrun decoder was created, unbuilt and weightless)."""
    model = _model(include_top=False)
    assert model.decoder is None
    assert not any("depth_head" in w.path for w in model.weights)
