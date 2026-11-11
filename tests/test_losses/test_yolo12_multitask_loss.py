"""YOLOv12MultiTaskLoss: no dead uncertainty weighting, no silent 0.0.

Two defects, both measured at cb96b75b5 (plan-2026-11-11T131349-13b28b37,
D-005):

1. ``use_uncertainty_weighting=True`` created ``log_var`` ``keras.Variable``s on a
   ``Loss``. A ``Loss`` is not a tracked layer, so the optimizer never saw them:
   they stayed at 0.0 and the loss history was identical to the off run. Both
   trainers exposed ``--uncertainty-weighting``, a silent no-op. The feature is
   deleted outright (no shim), so the keyword is now refused.
2. ``call`` returned ``0.0`` for an output whose rank is not 2/3/4, or whose task
   is not enabled; a rank-5 output fit with ``loss: [0.0]`` and no warning.
"""

import inspect

import keras
import numpy as np
import pytest

import train.yolo12.train_coco as train_coco
import train.yolo12.train_multitask as train_multitask
from dl_techniques.losses.yolo12_multitask_loss import (
    ClassificationFocalLoss,
    DiceFocalSegmentationLoss,
    YOLOv12MultiTaskLoss,
    YOLOv12ObjectDetectionLoss,
    create_yolov12_multitask_loss,
)

TASKS = ["detection", "segmentation", "classification"]


def _loss(tasks=TASKS, **kwargs):
    return YOLOv12MultiTaskLoss(
        tasks=tasks, num_classes=3, input_shape=(64, 64), **kwargs
    )


class TestUncertaintyWeightingIsGone:
    def test_the_keyword_is_refused(self):
        with pytest.raises(TypeError):
            _loss(use_uncertainty_weighting=True)

    def test_the_keyword_is_refused_when_false_too(self):
        # No shim: the knob does not exist, so even the off value is unknown.
        with pytest.raises(TypeError):
            _loss(use_uncertainty_weighting=False)

    def test_the_factory_refuses_it(self):
        with pytest.raises(TypeError):
            create_yolov12_multitask_loss(
                tasks=TASKS, num_classes=3, input_shape=(64, 64),
                use_uncertainty_weighting=True,
            )

    def test_the_config_does_not_carry_it(self):
        assert "use_uncertainty_weighting" not in _loss().get_config()

    def test_no_log_var_variable_exists(self):
        loss = _loss()
        assert not [k for k in vars(loss) if "log_var" in k]
        assert not hasattr(loss, "_build_uncertainty_weights")

    def test_task_weights_report_the_configured_static_weights(self):
        loss = _loss(
            detection_weight=2.0, segmentation_weight=0.5,
            classification_weight=0.25,
        )
        assert loss.get_task_weights() == {
            "detection": 2.0, "segmentation": 0.5, "classification": 0.25,
        }

    def test_task_weights_only_name_enabled_tasks(self):
        loss = _loss(tasks=["detection"], detection_weight=3.0)
        assert loss.get_task_weights() == {"detection": 3.0}

    def test_static_weight_scales_the_sub_loss(self):
        y_true = np.eye(3, dtype="float32")[[0, 1, 2, 0]]
        y_pred = np.random.default_rng(0).standard_normal((4, 3)).astype("float32")
        one = float(_loss(tasks=["classification"])(y_true, y_pred))
        three = float(
            _loss(tasks=["classification"], classification_weight=3.0)(y_true, y_pred)
        )
        assert one > 0.0
        np.testing.assert_allclose(three, 3.0 * one, rtol=1e-5)


class TestAnUnroutableOutputRaises:
    @pytest.mark.parametrize("shape", [(2, 3, 4, 5, 6), (6,)])
    def test_an_unrecognized_rank_raises_and_names_it(self, shape):
        loss = _loss()
        y = np.zeros(shape, dtype="float32")
        with pytest.raises(ValueError, match=rf"rank {len(shape)}") as err:
            loss(y, y)
        # The message names the enabled tasks so the mismatch is findable.
        for task in TASKS:
            assert task in str(err.value)

    def test_a_rank_whose_task_is_not_enabled_raises(self):
        loss = _loss(tasks=["detection"])
        y = np.zeros((2, 3), dtype="float32")  # classification rank, not enabled
        with pytest.raises(ValueError, match="classification"):
            loss(y, y)

    def test_the_error_survives_graph_tracing(self):
        loss = _loss()
        model = keras.Sequential([keras.layers.Input((2, 2, 2, 2)), keras.layers.Flatten(),
                                  keras.layers.Reshape((2, 2, 2, 2))])
        model.compile(optimizer="sgd", loss=loss)
        x = np.zeros((2, 2, 2, 2, 2), dtype="float32")
        with pytest.raises(ValueError, match="rank 5"):
            model.fit(x, x, epochs=1, verbose=0)

    def test_a_routable_rank_still_returns_a_finite_loss(self):
        y_true = np.eye(3, dtype="float32")[[0, 1]]
        y_pred = np.zeros((2, 3), dtype="float32")
        assert np.isfinite(float(_loss()(y_true, y_pred)))


class TestTheTrainersNoLongerOfferTheFlag:
    def test_train_multitask_rejects_the_flag(self, monkeypatch):
        calls = []
        monkeypatch.setattr(train_multitask, "train_model", lambda a: calls.append(a))
        monkeypatch.setattr(train_multitask, "set_seeds", lambda seed: None)
        monkeypatch.setattr(
            "sys.argv",
            ["train_multitask", "--data-dir", ".", "--uncertainty-weighting"],
        )
        with pytest.raises(SystemExit) as err:
            train_multitask.main()
        assert err.value.code == 2
        assert calls == []

    def test_train_multitask_still_parses_without_the_flag(self, monkeypatch):
        seen = []
        monkeypatch.setattr(train_multitask, "train_model", lambda a: seen.append(a))
        monkeypatch.setattr(train_multitask, "set_seeds", lambda seed: None)
        monkeypatch.setattr("sys.argv", ["train_multitask", "--data-dir", "."])
        train_multitask.main()
        assert len(seen) == 1 and not hasattr(seen[0], "uncertainty_weighting")

    def test_train_coco_rejects_the_flag(self, monkeypatch):
        class Reached(Exception):
            pass

        def reached(*a, **k):
            raise Reached

        # First thing main() does after parse_args that is not argparse.
        monkeypatch.setattr(train_coco, "refuse_a_dropped_weight_decay", reached)
        monkeypatch.setattr("sys.argv", ["train_coco", "--uncertainty-weighting"])
        with pytest.raises(SystemExit) as err:
            train_coco.main()
        assert err.value.code == 2

    def test_train_coco_still_parses_without_the_flag(self, monkeypatch):
        class Reached(Exception):
            pass

        def reached(*a, **k):
            raise Reached

        monkeypatch.setattr(train_coco, "refuse_a_dropped_weight_decay", reached)
        monkeypatch.setattr("sys.argv", ["train_coco"])
        with pytest.raises(Reached):
            train_coco.main()

    def test_the_builders_have_no_uncertainty_parameter(self):
        assert "use_uncertainty_weighting" not in inspect.signature(
            train_coco.create_coco_model_and_loss
        ).parameters
        assert "use_uncertainty_weighting" not in inspect.signature(
            train_multitask.create_model_and_loss
        ).parameters


class TestTheFixedReductionSubLossesRoundTrip:
    """``reduction="none"`` is hard-coded, yet ``get_config`` emits it.

    Measured at 8ad64d2bd: ``from_config(get_config())`` raised ``TypeError ... got
    multiple values for keyword argument 'reduction'`` for all three classes
    (D-006). The fix lives in one shared ``from_config``, not three copies.
    """

    @staticmethod
    def _instances():
        return [
            YOLOv12ObjectDetectionLoss(num_classes=3, input_shape=(64, 64)),
            DiceFocalSegmentationLoss(num_classes=3),
            ClassificationFocalLoss(),
        ]

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_from_config_accepts_the_emitted_config(self, index):
        original = self._instances()[index]
        config = original.get_config()
        assert config["reduction"] == "none"
        restored = type(original).from_config(config)
        assert type(restored) is type(original)
        assert restored.reduction == "none"
        assert restored.get_config() == config

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_keras_serialization_round_trips(self, index):
        original = self._instances()[index]
        restored = keras.saving.deserialize_keras_object(
            keras.saving.serialize_keras_object(original)
        )
        assert type(restored) is type(original)

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_from_config_does_not_mutate_the_callers_dict(self, index):
        config = self._instances()[index].get_config()
        snapshot = dict(config)
        type(self._instances()[index]).from_config(config)
        assert config == snapshot
