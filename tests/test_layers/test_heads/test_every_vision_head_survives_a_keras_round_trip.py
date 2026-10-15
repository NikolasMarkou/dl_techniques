"""Every vision head survives a ``.keras`` save and load, by value.

Guard for plan-2026-10-14T062616-81938e81 step 7 (decision D-027). Checking the
vision heads README against the code at ``84438bee6`` found three heads that
did not:

* ``SegmentationHead``: ``load_model`` rebuilds a layer from its JSON
  ``build_config``, where ONE input shape arrives as a LIST of ints.
  ``build`` read "a list" as "several shapes", took the last int as the base
  shape and raised ``TypeError: 'int' object is not iterable`` (spatial input)
  or ``cannot unpack non-iterable NoneType object`` (sequence input). Every
  saved model holding the head, FFTNet's segmenter included, failed to load.
* ``DepthEstimationHead`` (progressive decoder): its three ``ConvBlock`` s and
  three ``Conv2DTranspose`` layers sat in a list of ``[conv, upsample]``
  pairs. Keras does not save a layer inside a nested list, so a reload
  silently ran with fresh kernels there: measured max |depth diff| 17.3, no
  warning, the biases agreeing only because both copies were zero-initialised.
* ``MultiTaskHead``: ``get_config`` returned the caller's ``VisionTaskType``
  members, which are not JSON-serializable, so ``save`` raised ``TypeError``;
  and a string ``task_type`` was refused, so no stored config could rebuild it.
  Its task heads were also unnamed, so their process-global Keras suffixes
  ('detection_head_2' saved, 'detection_head_3' reloaded) moved every weight
  path under them across the round trip.

The oracle is two-part. Every weight is compared by path with
``np.array_equal`` (a restore that skipped a layer fails here even when that
layer's init happens to match), and the outputs on a fixed input are compared
exactly, on the golden-reference device, where both models run the same
deterministic kernels.
"""

from typing import Any, Callable, Dict, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.vision.factory import (
    ClassificationHead,
    DepthEstimationHead,
    DetectionHead,
    EnhancementHead,
    InstanceSegmentationHead,
    MultiTaskHead,
    OpticalFlowHead,
    SegmentationHead,
    SurfaceNormalsHead,
    create_vision_head,
)
from dl_techniques.layers.heads.vision.task_types import VisionTaskType

HIDDEN_DIM = 16
SPATIAL = (2, 6, 10, HIDDEN_DIM)        # non-square on purpose
SEQUENCE = (2, 12, HIDDEN_DIM)          # 12 tokens on a (3, 4) patch grid


def _multi_task(task_type_form: Callable[[VisionTaskType], Any]) -> MultiTaskHead:
    return MultiTaskHead(
        {
            "det": {"task_type": task_type_form(VisionTaskType.DETECTION),
                    "num_classes": 3, "hidden_dim": HIDDEN_DIM},
            "seg": {"task_type": task_type_form(VisionTaskType.SEGMENTATION),
                    "num_classes": 3, "hidden_dim": HIDDEN_DIM,
                    "use_skip_connections": False},
            "dep": {"task_type": task_type_form(VisionTaskType.DEPTH_ESTIMATION),
                    "hidden_dim": HIDDEN_DIM},
            "nrm": {"task_type": task_type_form(VisionTaskType.SURFACE_NORMALS),
                    "hidden_dim": HIDDEN_DIM},
            "flw": {"task_type": task_type_form(VisionTaskType.OPTICAL_FLOW),
                    "hidden_dim": HIDDEN_DIM},
            "kpt": {"task_type": task_type_form(VisionTaskType.KEYPOINT_DETECTION),
                    "num_classes": 5, "hidden_dim": HIDDEN_DIM},
            "sr": {"task_type": task_type_form(VisionTaskType.SUPER_RESOLUTION),
                   "hidden_dim": HIDDEN_DIM},
        },
        use_task_specific_attention=False,
    )


#: (id, builder, input shapes). A list of shapes means a list input.
_SUBJECTS: Tuple[Tuple[str, Callable[[], keras.layers.Layer], Any], ...] = (
    ("classification", lambda: ClassificationHead(
        num_classes=5, hidden_dim=HIDDEN_DIM), SPATIAL),
    ("segmentation", lambda: SegmentationHead(
        num_classes=5, hidden_dim=HIDDEN_DIM, use_skip_connections=False), SPATIAL),
    ("segmentation_skip_default_single_input", lambda: SegmentationHead(
        num_classes=5, hidden_dim=HIDDEN_DIM), SPATIAL),
    ("segmentation_with_a_skip_list", lambda: SegmentationHead(
        num_classes=5, hidden_dim=HIDDEN_DIM, upsampling_factor=4),
        [(2, 6, 10, 8), SPATIAL]),
    ("segmentation_sequence", lambda: SegmentationHead(
        num_classes=5, hidden_dim=HIDDEN_DIM, use_skip_connections=False,
        input_format="sequence", patch_grid_size=(3, 4)), SEQUENCE),
    ("instance_segmentation", lambda: InstanceSegmentationHead(
        num_classes=5, hidden_dim=HIDDEN_DIM, num_instances=4), SPATIAL),
    ("detection", lambda: DetectionHead(
        num_classes=5, hidden_dim=HIDDEN_DIM), SPATIAL),
    ("detection_sequence", lambda: DetectionHead(
        num_classes=5, hidden_dim=HIDDEN_DIM, input_format="sequence",
        patch_grid_size=(3, 4)), SEQUENCE),
    ("depth_progressive", lambda: DepthEstimationHead(
        hidden_dim=HIDDEN_DIM), SPATIAL),
    ("depth_dpt", lambda: DepthEstimationHead(
        decoder_style="dpt", dpt_dims=[16, 8], dpt_upsample_factor=4), SPATIAL),
    ("enhancement", lambda: EnhancementHead(hidden_dim=HIDDEN_DIM), SPATIAL),
    ("surface_normals", lambda: SurfaceNormalsHead(hidden_dim=HIDDEN_DIM), SPATIAL),
    ("optical_flow", lambda: OpticalFlowHead(hidden_dim=HIDDEN_DIM), SPATIAL),
    ("keypoint_detection", lambda: create_vision_head(
        "keypoint_detection", num_classes=5, hidden_dim=HIDDEN_DIM), SPATIAL),
    ("multi_task_enum_task_types", lambda: _multi_task(lambda t: t), SPATIAL),
    ("multi_task_string_task_types", lambda: _multi_task(lambda t: t.value), SPATIAL),
)


def _inputs(shapes: Any) -> Any:
    rng = np.random.default_rng(17)
    if isinstance(shapes, list):
        return [rng.normal(size=s).astype("float32") for s in shapes]
    return rng.normal(size=shapes).astype("float32")


def _flat(out: Any) -> np.ndarray:
    if isinstance(out, dict):
        return np.concatenate([_flat(out[k]).ravel() for k in sorted(out)])
    return np.asarray(keras.ops.convert_to_numpy(out)).ravel()


def _weights_by_path(model: keras.Model) -> Dict[str, np.ndarray]:
    return {w.path: keras.ops.convert_to_numpy(w) for w in model.weights}


@pytest.mark.parametrize(
    "label,builder,shapes", _SUBJECTS, ids=[s[0] for s in _SUBJECTS])
def test_the_head_round_trips_every_weight_and_its_output(
        label: str, builder, shapes, tmp_path, golden_reference_device
) -> None:
    data = _inputs(shapes)
    with keras.device(golden_reference_device):
        keras.utils.set_random_seed(3)
        if isinstance(shapes, list):
            inputs = [keras.Input(shape=s[1:]) for s in shapes]
        else:
            inputs = keras.Input(shape=shapes[1:])
        model = keras.Model(inputs, builder()(inputs))
        before = model.predict(data, verbose=0)

        path = tmp_path / f"{label}.keras"
        model.save(path)
        restored = keras.models.load_model(path)
        after = restored.predict(data, verbose=0)

    saved, loaded = _weights_by_path(model), _weights_by_path(restored)
    assert saved, f"{label}: the model holds no weights; the probe is vacuous"
    assert sorted(loaded) == sorted(saved), (
        f"{label}: weight paths differ across the round trip: "
        f"only saved {sorted(set(saved) - set(loaded))}, "
        f"only loaded {sorted(set(loaded) - set(saved))}"
    )
    not_restored = [p for p in saved if not np.array_equal(saved[p], loaded[p])]
    assert not not_restored, (
        f"{label}: {len(not_restored)} weights were not restored by value: "
        f"{not_restored}"
    )
    np.testing.assert_array_equal(
        _flat(after), _flat(before),
        err_msg=f"{label}: the reloaded head's output differs",
    )


def test_multi_task_head_config_holds_strings_and_leaves_the_caller_dict_alone() -> None:
    caller = {"det": {"task_type": VisionTaskType.DETECTION, "num_classes": 3,
                      "hidden_dim": HIDDEN_DIM}}
    head = MultiTaskHead(caller, use_task_specific_attention=False)

    assert head.get_config()["task_configs"]["det"]["task_type"] == "detection"
    assert caller["det"]["task_type"] is VisionTaskType.DETECTION
    rebuilt = MultiTaskHead.from_config(head.get_config())
    assert isinstance(rebuilt.task_heads["det"], DetectionHead)


def test_multi_task_head_refuses_an_unknown_task_string() -> None:
    with pytest.raises(ValueError, match="not_a_task"):
        MultiTaskHead({"x": {"task_type": "not_a_task", "num_classes": 3}})
