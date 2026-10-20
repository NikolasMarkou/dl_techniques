"""Every ``head_config_overrides`` key of an FFTNet head reaches its head or raises.

Measured at 69b42275b: the detection and segmentation branches read four
override keys by hand (``num_anchors`` for detection, ``use_attention``,
``attention_type``, ``use_ffn``) and dropped every other key in silence, so
``head_config_overrides={"dropout_rate": 0.0}`` built a head at 0.1; the
classification branch read ``dropout_rate`` and dropped the rest. Both
heads now go through ``build_model_vision_head`` (iteration 2 step 6.6,
D-210), which applies every key last; the classification head refuses a key
it has no reader for. Since step 10.1 every task head, the classifier's too,
is built by ``build_model_vision_head`` inside ``FFTNet.__init__``.
"""

import pytest

from dl_techniques.models.vision.fftnet.model import create_fftnet


def _head(model, name):
    assert model.head.name == name
    return model.head


@pytest.mark.parametrize("task,head_name", [
    ("detection", "detection_head"), ("segmentation", "segmentation_head")])
def test_a_head_keyword_override_reaches_the_head(task, head_name):
    model = create_fftnet(
        "tiny", task_type=task, num_classes=4, image_size=32, patch_size=16,
        head_config_overrides={"dropout_rate": 0.0, "use_ffn": True, "ffn_type": "glu"})
    head = _head(model, head_name)
    assert (head.dropout_rate, head.use_ffn, head.ffn_type) == (0.0, True, "glu")


@pytest.mark.parametrize("task", ["detection", "segmentation"])
def test_an_unknown_override_is_refused(task):
    with pytest.raises(ValueError, match="not_a_head_keyword"):
        create_fftnet(
            "tiny", task_type=task, num_classes=4, image_size=32, patch_size=16,
            head_config_overrides={"not_a_head_keyword": 1})


def test_the_classification_head_refuses_a_key_it_cannot_read():
    with pytest.raises(ValueError, match="use_ffn"):
        create_fftnet(
            "tiny", task_type="classification", num_classes=4, image_size=32,
            patch_size=16, head_config_overrides={"use_ffn": True})


@pytest.mark.parametrize("task,head_name,default", [
    ("detection", "detection_head", "multi_head"),
    ("segmentation", "segmentation_head", "cbam")])
def test_the_attention_defaults_hold_and_turn_off_cleanly(task, head_name, default):
    # Liveness: the factory's attention default is unchanged, and turning
    # attention off drops it rather than tripping the head's refusal.
    on = create_fftnet("tiny", task_type=task, num_classes=4,
                                 image_size=32, patch_size=16)
    assert _head(on, head_name).attention_type == default
    off = create_fftnet("tiny", task_type=task, num_classes=4, image_size=32,
                                  patch_size=16, head_config_overrides={"use_attention": False})
    assert _head(off, head_name).attention_type is None
