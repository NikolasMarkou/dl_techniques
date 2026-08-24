"""MobileNet: `summary()` must report the real model, and a headless model must
return a feature MAP.

Guard for C-10 + C-11 (plan-2026-08-14T233721-d4f9beb2, step 34).

(a) ``mobilenet_v4.py``'s ``summary()`` called ``self.build(self._input_shape)``
    with the stored 3-tuple ``(224, 224, 3)``, where ``Model.build()`` expects the
    BATCH input shape. On a never-called model that marks the model built without
    materializing any sub-layer weights, so the printed summary and the
    ``count_params()`` line under it reported a near-zero total instead of raising.
    ``mobilenet_v1.py`` already does ``self.build((None, *self._input_shape))``.

(b) ``mobilenet_v1.py`` applied ``global_avg_pool`` unconditionally, including
    under ``include_top=False``, so V1 returned a 2-D pooled vector where
    V2/V3/V4 all return the 4-D feature map a detection or segmentation head
    needs.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.mobilenet.mobilenet_v1 import MobileNetV1
from dl_techniques.models.mobilenet.mobilenet_v2 import MobileNetV2
from dl_techniques.models.mobilenet.mobilenet_v3 import MobileNetV3
from dl_techniques.models.mobilenet.mobilenet_v4 import MobileNetV4


SMALL_SHAPE = (32, 32, 3)
# 64px, so that after the family's 32x downsampling a real feature map still has
# a 2x2 spatial extent: at 32px every version collapses to 1x1 and a rank-2 vs
# rank-4 confusion is much easier to wave away.
HEADLESS_SHAPE = (64, 64, 3)


def _headless(cls):
    return cls(include_top=False, input_shape=HEADLESS_SHAPE)


class TestSummaryBuildsForReal:
    @pytest.mark.parametrize(
        "cls", [MobileNetV1, MobileNetV2, MobileNetV3, MobileNetV4]
    )
    def test_summary_on_a_fresh_model_reports_the_real_parameter_count(self, cls, capsys):
        fresh = cls(num_classes=10, input_shape=SMALL_SHAPE)
        assert not fresh.built
        fresh.summary()
        after_summary = fresh.count_params()

        # Reference: the same model, built the way a caller builds it.
        reference = cls(num_classes=10, input_shape=SMALL_SHAPE)
        reference(np.zeros((1, *SMALL_SHAPE), dtype="float32"))

        assert after_summary == reference.count_params(), (
            f"{cls.__name__}.summary() left the model with {after_summary} "
            f"parameters, but a really-built model has {reference.count_params()}"
        )
        assert after_summary > 1000, "summary() reported a near-empty model"


class TestHeadlessReturnsAFeatureMap:
    @pytest.mark.parametrize(
        "cls", [MobileNetV1, MobileNetV2, MobileNetV3, MobileNetV4]
    )
    def test_include_top_false_is_rank_four(self, cls):
        model = _headless(cls)
        out = model(np.random.rand(2, *HEADLESS_SHAPE).astype("float32"))
        shape = tuple(keras.ops.shape(out))
        assert len(shape) == 4, (
            f"{cls.__name__}(include_top=False) returned rank {len(shape)} "
            f"{shape}; the family contract is a 4-D feature map"
        )
        assert shape[0] == 2
        assert shape[1] > 1 and shape[2] > 1, (
            f"{cls.__name__} headless output has no spatial extent: {shape}"
        )

    def test_v1_top_still_returns_class_probabilities(self):
        """Anti-vacuity: gating the pool must not disturb the classifier path."""
        model = MobileNetV1(num_classes=7, input_shape=SMALL_SHAPE)
        out = keras.ops.convert_to_numpy(
            model(np.random.rand(3, *SMALL_SHAPE).astype("float32"))
        )
        assert out.shape == (3, 7)
        np.testing.assert_allclose(out.sum(axis=-1), np.ones(3), rtol=1e-5, atol=1e-5)

    def test_v1_headless_pooled_manually_matches_the_old_output(self):
        """The pooled vector is still one `GlobalAveragePooling2D` away, so the
        behaviour change costs a caller exactly one layer."""
        model = _headless(MobileNetV1)
        x = np.random.rand(2, *HEADLESS_SHAPE).astype("float32")
        fmap = model(x)
        pooled = keras.layers.GlobalAveragePooling2D()(fmap)
        assert tuple(keras.ops.shape(pooled)) == (2, tuple(keras.ops.shape(fmap))[-1])
