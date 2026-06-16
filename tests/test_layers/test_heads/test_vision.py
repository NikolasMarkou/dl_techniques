"""Tests for the merged ``heads.vision`` sub-package.

Covers the two opportunistic bugfixes landed during the merge (SC6):

* **EnhancementHead module-scope serialization.** The class was previously
  defined inside ``create_enhancement_head()`` (a closure-local
  ``@register_keras_serializable`` class). It is now at module scope with the
  same name, so ``Custom>EnhancementHead`` is registered and a build via the
  factory round-trips through ``.keras`` save/load.
* **MultiTaskHead dict non-mutation.** ``_create_task_heads`` used to
  ``config.pop('task_type')`` on the caller's dict; it now copies first. We
  assert the caller's input config dict is unchanged after
  ``create_multi_task_head``.

Plus a vision factory smoke + save/load round-trip for ``ClassificationHead``.
"""

import os
import copy
import tempfile

import numpy as np
import keras
from keras import ops

from dl_techniques.layers.heads.vision import (
    VisionTaskType,
    ClassificationHead,
    DetectionHead,
    SegmentationHead,
    DepthEstimationHead,
    InstanceSegmentationHead,
    EnhancementHead,
    MultiTaskHead,
    create_vision_head,
    create_enhancement_head,
    create_multi_task_head,
)

# Feature-map input: (B, H, W, C).
B, H, W, C = 2, 8, 8, 16


def _feature_map() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.standard_normal((B, H, W, C)).astype("float32")


# ---------------------------------------------------------------------
# SC6 — EnhancementHead module-scope serialization
# ---------------------------------------------------------------------

class TestEnhancementHeadSerialization:

    def test_registered_at_module_scope(self) -> None:
        assert (
            keras.saving.get_registered_object("Custom>EnhancementHead")
            is not None
        )

    def test_create_and_roundtrip(self) -> None:
        head = create_enhancement_head(
            VisionTaskType.DENOISING, hidden_dim=16, output_channels=3
        )
        assert isinstance(head, EnhancementHead)

        inp = keras.Input(shape=(H, W, C))
        out = head(inp)
        model = keras.Model(inp, out)

        x = _feature_map()
        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "enh.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0["enhanced"]),
            ops.convert_to_numpy(y1["enhanced"]),
            atol=1e-4,
        )


# ---------------------------------------------------------------------
# SC6 — MultiTaskHead dict non-mutation
# ---------------------------------------------------------------------

class TestMultiTaskHeadDictNonMutation:

    def test_input_config_not_mutated(self) -> None:
        task_configs = {
            "cls": {
                "task_type": VisionTaskType.CLASSIFICATION,
                "num_classes": 10,
                "hidden_dim": 16,
            }
        }
        reference = copy.deepcopy(task_configs)

        head = create_multi_task_head(task_configs)

        # The caller's dict (and its nested 'task_type') must be intact.
        assert task_configs == reference
        assert "task_type" in task_configs["cls"]
        # And the head was actually constructed.
        assert "cls" in head.task_heads


# ---------------------------------------------------------------------
# Vision factory smoke + save/load round-trip
# ---------------------------------------------------------------------

class TestVisionFactoryAndRoundtrip:

    def test_factory_returns_classification_head(self) -> None:
        head = create_vision_head(
            VisionTaskType.CLASSIFICATION, num_classes=10, hidden_dim=16
        )
        assert isinstance(head, ClassificationHead)

    def test_factory_from_string(self) -> None:
        head = create_vision_head("classification", num_classes=4, hidden_dim=16)
        assert isinstance(head, ClassificationHead)

    def test_model_save_load_roundtrip(self) -> None:
        inp = keras.Input(shape=(H, W, C))
        out = create_vision_head(
            VisionTaskType.CLASSIFICATION, num_classes=5, hidden_dim=16
        )(inp)
        model = keras.Model(inp, out)

        x = _feature_map()
        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cls.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0["logits"]),
            ops.convert_to_numpy(y1["logits"]),
            atol=1e-4,
        )


# ---------------------------------------------------------------------
# compute_output_shape agreement with actual call() output
# ---------------------------------------------------------------------

class TestVisionHeadComputeOutputShape:
    """Every concrete vision head's compute_output_shape must match call()."""

    def _assert_shapes_match(self, head, x) -> None:
        out = head(x)
        declared = head.compute_output_shape(x.shape)
        if isinstance(out, dict):
            assert set(out.keys()) == set(declared.keys())
            for key in out:
                assert tuple(out[key].shape) == tuple(declared[key])
        else:
            assert tuple(out.shape) == tuple(declared)

    def test_classification_head(self) -> None:
        head = ClassificationHead(num_classes=7, hidden_dim=16, use_ffn=False)
        self._assert_shapes_match(head, _feature_map())

    def test_detection_head(self) -> None:
        head = DetectionHead(num_classes=5, hidden_dim=16, use_ffn=False)
        self._assert_shapes_match(head, _feature_map())

    def test_segmentation_head(self) -> None:
        head = SegmentationHead(
            num_classes=4, hidden_dim=16, use_ffn=False,
            use_skip_connections=False,
        )
        self._assert_shapes_match(head, _feature_map())

    def test_depth_estimation_head(self) -> None:
        head = DepthEstimationHead(output_channels=1, hidden_dim=16, use_ffn=False)
        self._assert_shapes_match(head, _feature_map())

    def test_instance_segmentation_head(self) -> None:
        head = InstanceSegmentationHead(
            num_classes=5, num_instances=12, hidden_dim=16, use_ffn=False,
        )
        self._assert_shapes_match(head, _feature_map())

    def test_enhancement_head_denoise(self) -> None:
        head = EnhancementHead(output_channels=3, scale_factor=1, hidden_dim=16)
        self._assert_shapes_match(head, _feature_map())

    def test_enhancement_head_super_resolution(self) -> None:
        head = EnhancementHead(output_channels=3, scale_factor=2, hidden_dim=16)
        self._assert_shapes_match(head, _feature_map())

    def test_multi_task_head(self) -> None:
        head = MultiTaskHead(
            task_configs={
                "cls": {
                    "task_type": VisionTaskType.CLASSIFICATION,
                    "num_classes": 6,
                    "hidden_dim": 16,
                    "use_ffn": False,
                },
                "det": {
                    "task_type": VisionTaskType.DETECTION,
                    "num_classes": 3,
                    "hidden_dim": 16,
                    "use_ffn": False,
                },
            },
            use_task_specific_attention=False,
        )
        x = _feature_map()
        out = head(x)
        declared = head.compute_output_shape(x.shape)
        assert set(out.keys()) == set(declared.keys())
        for task_name in out:
            for key in out[task_name]:
                assert tuple(out[task_name][key].shape) == tuple(declared[task_name][key])
