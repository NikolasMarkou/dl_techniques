"""The segmentation and classification heads follow the detection head and the blocks.

Two conventions hold everywhere else in `layers/yolo12/`:

* BatchNorm takes ``YOLO12_NORM_KWARGS`` (epsilon 1e-3, momentum 0.97, the
  Ultralytics values), not the Keras defaults (momentum 0.99).
* Every weighted sub-layer gets its own clone of ``kernel_initializer``. One
  seedless Keras initializer instance replays the same sample at every site, so
  two sub-layers with the same kernel shape come out bit-identical.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.yolo12.yolo12_blocks import YOLO12_NORM_KWARGS
from dl_techniques.layers.yolo12.yolo12_heads import (
    YOLOv12ClassificationHead,
    YOLOv12SegmentationHead,
)

_SHAPES = [(None, 16, 16, 32), (None, 8, 8, 32), (None, 4, 4, 32)]
_WEIGHTED = (keras.layers.Conv2D, keras.layers.Conv2DTranspose, keras.layers.Dense)


def _built_segmentation_head(**kwargs):
    head = YOLOv12SegmentationHead(
        num_classes=2, intermediate_filters=[16, 16, 16, 16], **kwargs
    )
    head.build(_SHAPES)
    return head


def _built_classification_head(**kwargs):
    head = YOLOv12ClassificationHead(num_classes=3, hidden_dims=[64, 64], **kwargs)
    head.build(_SHAPES)
    return head


def _sublayers(head, kinds):
    """Every sub-layer of ``head``, including the Sequentials it fills in ``build``."""
    found, seen = [], set()

    def visit(layer):
        if id(layer) in seen:
            return
        seen.add(id(layer))
        if isinstance(layer, kinds):
            found.append(layer)
        for child in getattr(layer, "layers", []):
            visit(child)

    for layer in head._flatten_layers():
        visit(layer)
    for name in ("upconv_blocks", "skip_convs", "dense_layers"):
        for layer in getattr(head, name, []):
            if layer is not None:
                visit(layer)
    return found


def _kernels(layers):
    return [keras.ops.convert_to_numpy(layer.kernel) for layer in layers]


class TestSegmentationHead:

    def test_batch_norm_uses_the_house_epsilon_and_momentum(self):
        norms = _sublayers(_built_segmentation_head(), keras.layers.BatchNormalization)
        assert len(norms) == 10
        for norm in norms:
            assert norm.epsilon == YOLO12_NORM_KWARGS["epsilon"], norm.name
            assert norm.momentum == YOLO12_NORM_KWARGS["momentum"], norm.name

    def test_no_two_sublayers_share_an_initializer_instance(self):
        head = _built_segmentation_head()
        layers = _sublayers(head, _WEIGHTED)
        ids = [id(layer.kernel_initializer) for layer in layers]
        assert len(layers) >= 9
        assert len(set(ids)) == len(ids)
        assert id(head.kernel_initializer) not in ids

    def test_equal_shape_kernels_draw_independently(self):
        head = _built_segmentation_head()
        refine = [
            layer for layer in _sublayers(head, keras.layers.Conv2D)
            if layer.name.endswith("_refine")
        ]
        kernels = _kernels(refine)
        assert len(kernels) == 4 and len({k.shape for k in kernels}) == 1
        for i in range(len(kernels)):
            for j in range(i + 1, len(kernels)):
                assert not np.array_equal(kernels[i], kernels[j]), (i, j)

    def test_the_configured_initializer_is_still_what_each_site_uses(self):
        head = _built_segmentation_head(kernel_initializer="glorot_uniform")
        for layer in _sublayers(head, _WEIGHTED):
            assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)


class TestClassificationHead:

    def test_no_two_sublayers_share_an_initializer_instance(self):
        head = _built_classification_head()
        layers = _sublayers(head, keras.layers.Dense)
        ids = [id(layer.kernel_initializer) for layer in layers]
        assert len(layers) >= 4
        assert len(set(ids)) == len(ids)
        assert id(head.kernel_initializer) not in ids

    def test_equal_shape_dense_kernels_draw_independently(self):
        head = _built_classification_head(use_attention=False)
        hidden = [d for d in head.dense_layers]
        kernels = _kernels(hidden)
        # dense_1 is (64, 64); dense_0 is (in, 64). Compare any equal-shaped pair.
        shapes = [k.shape for k in kernels]
        pairs = [
            (i, j) for i in range(len(kernels)) for j in range(i + 1, len(kernels))
            if shapes[i] == shapes[j]
        ]
        for i, j in pairs:
            assert not np.array_equal(kernels[i], kernels[j])
