"""SCUNet pads its input the way the official code does: edge replication.

cszn/SCUNet ``models/network_scunet.py`` ``SCUNet.forward``::

    h, w = x0.size()[-2:]
    paddingBottom = int(np.ceil(h/64)*64-h)
    paddingRight = int(np.ceil(w/64)*64-w)
    x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

``nn.ReplicationPad2d`` repeats the last row and column, which is
``np.pad(mode='edge')``. Until plan-2026-10-14T062616-81938e81 step 9.1 this port
used ``keras.ops.pad(mode="REFLECT")``: a reflect pad must be smaller than the
extent it pads, so the model refused every height or width below 33, and on any
other size that is not a multiple of 64 it fed the network different border
content than the reference does.

The guard is by value, end to end: the model's output on an ``H x W`` input
must equal, exactly, its output on ``np.pad(x, mode='edge')`` (already a multiple
of 64, so the model pads nothing) cropped back to ``H x W``. Every output pixel
depends on the pad (the U-Net's bottleneck sees the whole padded image), so a
different pad content moves the output, and a refusal fails the call. The
sizes are the plan's: 5x7 and 1x1 (below the old floor), 16x16, 33x40 (legal
under the old pad, different content), 64x64 (no pad at all). The same holds
through a ``keras.Input((None, None, 3))`` functional model and through one
``tf.function`` trace with a fully dynamic ``TensorSpec``, where the pad sizes
are tensors.

Runs on the ``golden_reference_device`` (CPU), where both evaluations run the
same kernels on the same 64-multiple tensor, so equality is exact.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vision.image_restoration.scunet.model import SCUNet

SIZES = [(5, 7), (33, 40), (64, 64), (1, 1), (16, 16)]


def _edge_padded(x: np.ndarray) -> np.ndarray:
    h, w = x.shape[1], x.shape[2]
    return np.pad(x, [(0, 0), (0, (-h) % 64), (0, (-w) % 64), (0, 0)], mode="edge")


def _images(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((2, h, w, 3)).astype("float32")


@pytest.fixture(scope="module")
def model(golden_reference_device):
    with keras.device(golden_reference_device):
        keras.utils.set_random_seed(0)
        m = SCUNet(in_nc=3, config=[1] * 7, dim=16, head_dim=8, window_size=8,
                   input_resolution=64)
        m(_images(64, 64), training=False)
    return m


def _assert_equals_the_edge_padded_reference(run, x: np.ndarray) -> None:
    h, w = x.shape[1], x.shape[2]
    out = np.asarray(keras.ops.convert_to_numpy(run(x)))
    reference = np.asarray(keras.ops.convert_to_numpy(run(_edge_padded(x))))
    assert out.shape == x.shape, out.shape
    np.testing.assert_array_equal(out, reference[:, :h, :w, :])


@pytest.mark.parametrize("h,w", SIZES)
def test_the_output_is_the_edge_padded_references(model, golden_reference_device, h, w):
    with keras.device(golden_reference_device):
        _assert_equals_the_edge_padded_reference(
            lambda t: model(t, training=False), _images(h, w))


def test_the_same_holds_through_a_dynamic_shape_input(model, golden_reference_device):
    with keras.device(golden_reference_device):
        inp = keras.Input((None, None, 3))
        functional = keras.Model(inp, model(inp))
        assert functional.output_shape == (None, None, None, 3)

        @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)])
        def infer(t):
            return model(t, training=False)

        for h, w in SIZES:
            x = _images(h, w, seed=h * 100 + w)
            _assert_equals_the_edge_padded_reference(
                lambda t: functional(t, training=False), x)
            _assert_equals_the_edge_padded_reference(infer, x)


def test_the_pad_content_moves_the_output(model, golden_reference_device):
    """Anti-vacuity: the equality above could pass if the output ignored the
    pad. At 33x40 a reflect-padded input gives a different output."""
    with keras.device(golden_reference_device):
        x = _images(33, 40)
        reflect = np.pad(x, [(0, 0), (0, 31), (0, 24), (0, 0)], mode="reflect")
        a = np.asarray(model(_edge_padded(x), training=False))[:, :33, :40]
        b = np.asarray(model(reflect, training=False))[:, :33, :40]
        assert float(np.max(np.abs(a - b))) > 1e-3
