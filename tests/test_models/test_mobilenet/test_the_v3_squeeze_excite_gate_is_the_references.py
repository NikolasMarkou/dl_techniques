"""MobileNetV3's squeeze-excite gate is the reference's: biased 1x1 convs and a
hard-sigmoid gate.

All three references build it the same way:

* Keras Applications ``mobilenet_v3._se_block``: ``Conv2D(_depth(filters *
  se_ratio), 1)`` and ``Conv2D(filters, 1)`` with Keras' default
  ``use_bias=True``, ReLU between them, then ``hard_sigmoid``
  (``ReLU(6.0)(x + 3.0) * (1.0 / 6.0)``);
* torchvision ``ops.SqueezeExcitation(..., scale_activation=nn.Hardsigmoid)``
  with ``nn.Conv2d`` (``bias=True`` by default);
* the paper (Howard et al. 2019, Section 5.2 and Figure 4): hard-sigmoid
  replaces the sigmoid in the SE gate.

Until plan-2026-10-14T062616-81938e81 step 9.3.1 the port's SE convs took the
block's ``use_bias=False`` and its gate was a plain sigmoid (the module
docstring recorded the gate as a known deviation). The count against a live
``keras.applications`` model, which this file pinned as "plus the first row's
320" until step 10.3, is now exact at every width and lives in
``test_the_v3_widths_follow_keras_applications_at_every_width.py``.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.mobilenet.mobilenet_v3 import MobileNetV3


def _model(variant: str) -> MobileNetV3:
    m = MobileNetV3(num_classes=1000, variant=variant)
    m(np.zeros((1, 224, 224, 3), "float32"))
    return m


def _se_layers(model):
    return [l for l in model._flatten_layers(include_self=False)
            if l.name == "se" and hasattr(l, "conv_reduce")]


@pytest.mark.parametrize("variant", ["small", "large"])
def test_every_se_conv_carries_a_bias(variant):
    se = _se_layers(_model(variant))
    assert len(se) == (9 if variant == "small" else 8)
    for layer in se:
        assert layer.conv_reduce.use_bias and layer.conv_restore.use_bias


def test_the_gate_is_the_reference_hard_sigmoid(golden_reference_device):
    """The SE output, recomputed by hand from the layer's own weights with
    the reference gate ``relu6(z + 3) / 6``, equals the layer's output. On the
    CPU golden device: the GPU runs the 1x1 convs in TF32."""
    with keras.device(golden_reference_device):
        _check_the_gate()


def _check_the_gate():
    keras.utils.set_random_seed(0)
    model = _model("small")
    layer = _se_layers(model)[3]
    rng = np.random.default_rng(0)
    c = layer.input_channels
    x = rng.standard_normal((2, 5, 5, c)).astype("float32")
    w1, b1 = (np.asarray(v) for v in layer.conv_reduce.weights)
    w2, b2 = (np.asarray(v) for v in layer.conv_restore.weights)
    # Non-zero biases so the bias path is exercised by value.
    b1 = rng.standard_normal(b1.shape).astype("float32")
    b2 = 3.0 * rng.standard_normal(b2.shape).astype("float32")
    layer.conv_reduce.bias.assign(b1)
    layer.conv_restore.bias.assign(b2)
    pooled = x.mean(axis=(1, 2), keepdims=True)
    hidden = np.maximum(pooled @ w1[0, 0] + b1, 0.0)
    z = hidden @ w2[0, 0] + b2
    gate = np.clip(z + 3.0, 0.0, 6.0) / 6.0
    assert (gate == 0.0).any() and (gate == 1.0).any(), (
        "anti-vacuity: both saturated ends of the hard sigmoid are reached")
    out = np.asarray(layer(x))
    np.testing.assert_allclose(out, x * gate, rtol=1e-5, atol=1e-6)
