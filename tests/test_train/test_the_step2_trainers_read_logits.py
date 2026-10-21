"""The trainers of the models moved to logits compile their losses on logits.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 2.1 (D-142, D-144,
D-145, D-146). Each arm drives the trainer's own code, not a re-implementation:

- CoShNet: ``train_model`` runs up to its compile on a tiny stand-in dataset;
  the compiled loss must read logits.
- AccUNet: ``compile_model`` builds every segmentation loss with
  ``LossConfig(from_logits=True)``, thresholds binary accuracy at the logit 0,
  and the binary Dice metric thresholds the logit at 0. The synthetic
  multi-class labels are one-hot (D-145), the layout the losses validate, and
  one multi-class epoch of ``run_training`` completes.
- DocScanner: the segmenter stage's loss is ``BinaryCrossentropy(from_logits=True)``.

The PowerMLP and NAM trainers are guarded in their own directories.
"""

import argparse

import keras
import numpy as np
import pytest


class _StopAfterCompile(Exception):
    pass


def test_the_coshnet_trainer_compiles_from_logits(monkeypatch):
    from train.coshnet import train_coshnet as trainer

    x = np.random.default_rng(0).random((4, 32, 32, 3)).astype("float32")
    y = np.arange(4) % 10
    monkeypatch.setattr(trainer, "setup_gpu", lambda **kwargs: None)
    monkeypatch.setattr(trainer, "load_dataset",
                        lambda *a, **k: ((x, y), (x, y), (32, 32, 3), 10))
    seen = {}

    def stop(*args, **kwargs):
        raise _StopAfterCompile

    original_compile = keras.Model.compile

    def spy(self, *args, **kwargs):
        seen["loss"] = kwargs.get("loss")
        return original_compile(self, *args, **kwargs)

    monkeypatch.setattr(keras.Model, "compile", spy)
    monkeypatch.setattr(trainer, "create_callbacks", stop)
    args = argparse.Namespace(gpu=None, dataset="cifar10", batch_size=4, variant="tiny",
                              lr_schedule="constant", learning_rate=1e-3, epochs=1, patience=1,
                              seed=42)
    with pytest.raises(_StopAfterCompile):
        trainer.train_model(args)
    assert isinstance(seen["loss"], keras.losses.SparseCategoricalCrossentropy)
    assert seen["loss"].get_config()["from_logits"] is True


@pytest.mark.parametrize("num_classes", [1, 3])
def test_the_accunet_trainer_compiles_on_logits(num_classes):
    from train.accunet import train_accunet as trainer

    config = trainer.AccUNetTrainingConfig(
        num_classes=num_classes, image_size=32, base_filters=8, mlfc_iterations=1)
    model = trainer.build_model(config)
    trainer.compile_model(model, config, steps_per_epoch=1)
    assert model.loss.config.from_logits is True
    if num_classes == 1:
        accuracy = [m for m in model._compile_metrics._user_metrics if getattr(m, "name", "") == "binary_accuracy"]
        assert accuracy and accuracy[0].threshold == 0.0
        dice = trainer.BinaryDiceMetric()
        dice.update_state(np.array([[1.0], [0.0]]), np.array([[0.3], [-0.3]]))  # logits
        assert float(dice.result()) == pytest.approx((2.0 + 1.0) / (2.0 + 1.0))


def test_the_accunet_multiclass_labels_are_one_hot():
    from train.accunet import train_accunet as trainer

    x, y = trainer.generate_synthetic_segmentation_data(
        4, 32, 32, 3, rng=np.random.default_rng(0))
    assert y.shape == (4, 32, 32, 3)
    np.testing.assert_array_equal(y.sum(-1), 1.0)


def test_a_multiclass_accunet_run_completes(tmp_path):
    """Exit 1 with a y_true/y_pred shape mismatch at the first step before D-145."""
    from train.accunet import train_accunet as trainer

    config = trainer.AccUNetTrainingConfig(
        num_classes=3, image_size=32, num_samples=8, base_filters=8, mlfc_iterations=1,
        epochs=1, batch_size=4, output_dir=str(tmp_path), grid_every_n_epochs=1)
    model, results_dir = trainer.run_training(config)
    assert model.output_shape[-1] == 3


def test_the_doc_scanner_segmenter_loss_reads_logits():
    from train.doc_scanner import common

    config = common.DocScannerTrainingConfig(stage=common.STAGE_SEGMENTER)
    loss = common.build_loss(config)
    assert isinstance(loss, keras.losses.BinaryCrossentropy)
    assert loss.get_config()["from_logits"] is True
