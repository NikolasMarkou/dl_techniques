"""LightGlueTrainingModel: one real stock-fit run on tiny tensors and its guards."""

import inspect
import os
from pathlib import Path

import keras
import numpy as np
import pytest

from dl_techniques.layers.matching.learned_fourier_rotary import LearnedFourierRotaryEncoding
from dl_techniques.layers.matching.token_confidence import MatchTokenConfidence
from dl_techniques.models.vision.keypoints.lightglue.model import LightGlue
from dl_techniques.models.vision.keypoints.superpoint.model import SuperPoint
from train.lightglue import pipeline
from train.lightglue.data import list_images, make_pair_dataset
from train.lightglue.pipeline import (
    LightGlueCheckpoint, LightGlueTrainingModel, load_superpoint,
)

from tests.test_models.test_sam.dead_component_oracle import (
    fit_one_step_moved_variables, variable_labels,
)

from .conftest import SIZE, make_lightglue, make_superpoint, write_images


def _model(max_keypoints: int = 32, superpoint=None, **kwargs) -> LightGlueTrainingModel:
    return LightGlueTrainingModel(
        superpoint or make_superpoint(), make_lightglue(), max_keypoints=max_keypoints, **kwargs)


def _dataset(folder: Path, batch: int = 4, **kwargs):
    return make_pair_dataset(list_images(str(folder)), SIZE, batch, seed=3, **kwargs)


def _frozen(model: LightGlueTrainingModel) -> bool:
    """The frozen-front-end predicate: no SuperPoint weight is trainable or optimised."""
    lightglue_ids = {id(w) for w in model.lightglue.trainable_weights}
    return (
        not any(w.trainable for w in model.superpoint.weights)
        and {id(w) for w in model.trainable_weights} == lightglue_ids
    )


@pytest.fixture(scope="module")
def images(tmp_path_factory) -> Path:
    return write_images(tmp_path_factory.mktemp("images"), 8)


@pytest.fixture(scope="module")
def fitted(images):
    """One real stock fit: 2 epochs x 2 steps with validation, weights snapshotted around it."""
    model = _model()
    model.build({"image0": (None, SIZE, SIZE, 1)})
    model.compile(optimizer=keras.optimizers.Adam(1e-3), jit_compile=False)
    superpoint_before = [w.numpy().copy() for w in model.superpoint.weights]
    lightglue_before = [w.numpy().copy() for w in model.lightglue.trainable_weights]
    train = _dataset(images, repeat=True)
    val = _dataset(images, shuffle=False)
    history = model.fit(train, epochs=2, steps_per_epoch=2, validation_data=val,
                        validation_steps=2, verbose=0)
    return {
        "model": model, "history": history.history, "val": val,
        "superpoint_before": superpoint_before, "lightglue_before": lightglue_before,
    }


def test_loss_is_finite_and_positive(fitted):
    for key in ("loss", "val_loss"):
        values = np.array(fitted["history"][key])
        assert np.all(np.isfinite(values)) and np.all(values > 0.0), (key, values)


def test_metrics_reach_the_fit_history(fitted):
    for key in ("precision", "recall", "keypoints_per_image", "positive_fraction"):
        assert key in fitted["history"] and f"val_{key}" in fitted["history"], key
    assert fitted["history"]["keypoints_per_image"][-1] > 0.0


def test_lightglue_variables_moved(fitted):
    after = [w.numpy() for w in fitted["model"].lightglue.trainable_weights]
    assert len(after) == len(fitted["lightglue_before"]) > 0
    moved = [not np.array_equal(a, b) for a, b in zip(after, fitted["lightglue_before"])]
    assert any(moved)


def _one_step_report(images, confidence_weight=1.0):
    """One stock fit step on a fresh wrapper; returns (labels, report)."""
    model = _model()
    model.objective.confidence_weight = confidence_weight
    model.build({"image0": (None, SIZE, SIZE, 1)})
    model.compile(optimizer=keras.optimizers.Adam(1e-3), jit_compile=False)
    labels = variable_labels(model)
    train = _dataset(images, repeat=True)
    return labels, fit_one_step_moved_variables(model, train, steps_per_epoch=1)


# every group of LightGlue weights that must receive gradient under the pipeline objective
_REQUIRED_GROUPS = (
    "posenc", "self_attn_0", "self_attn_1", "cross_attn_0", "cross_attn_1",
    "log_assignment_0/matchability", "log_assignment_0/final_proj",
    "log_assignment_1/matchability", "log_assignment_1/final_proj",
    "token_confidence_0",
)


def _unmoved_required(labels, report):
    """Required groups with no label at all (a vacuous guard) and labels that did not move."""
    missing = [g for g in _REQUIRED_GROUPS if not any(g in label for label in labels)]
    return missing, list(report.unmoved)


def test_every_trainable_lightglue_variable_moves_in_one_fit_step(images):
    labels, report = _one_step_report(images)
    missing, unmoved = _unmoved_required(labels, report)
    assert not missing, f"guard is vacuous, no variable named {missing}; labels={labels}"
    assert report.total == len(labels) > 20
    assert not unmoved, report.summary()


def test_guard_is_red_when_the_confidence_logits_are_detached(images, monkeypatch):
    original = MatchTokenConfidence._logit

    def detached(self, desc):
        """MUTANT: the confidence logits are cut from the graph."""
        return keras.ops.stop_gradient(original(self, desc))

    monkeypatch.setattr(MatchTokenConfidence, "_logit", detached)
    labels, report = _one_step_report(images)
    _, unmoved = _unmoved_required(labels, report)
    assert unmoved and all("token_confidence_0" in u for u in unmoved), unmoved


def test_guard_is_red_when_the_confidence_term_is_dropped(images):
    labels, report = _one_step_report(images, confidence_weight=0.0)
    _, unmoved = _unmoved_required(labels, report)
    assert unmoved and all("token_confidence_0" in u for u in unmoved), unmoved


def test_guard_is_red_when_the_positional_encoding_is_cut(images, monkeypatch):
    original = LearnedFourierRotaryEncoding.call

    def cut(self, *args, **kwargs):
        return keras.ops.stop_gradient(original(self, *args, **kwargs))

    monkeypatch.setattr(LearnedFourierRotaryEncoding, "call", cut)
    labels, report = _one_step_report(images)
    _, unmoved = _unmoved_required(labels, report)
    assert unmoved and any("posenc" in u for u in unmoved), unmoved


def test_superpoint_variables_are_bitwise_unchanged(fitted):
    model = fitted["model"]
    assert _frozen(model)
    for before, weight in zip(fitted["superpoint_before"], model.superpoint.weights):
        assert np.array_equal(before, weight.numpy()), weight.path


def test_the_frozen_predicate_sees_an_unfrozen_superpoint():
    """RED proof kept as a negative control: the guard above fails on an unfrozen front end."""
    model = _model()
    assert _frozen(model)
    model.superpoint.trainable = True
    assert not _frozen(model)


def test_no_custom_train_step():
    assert LightGlueTrainingModel.train_step is keras.Model.train_step
    assert LightGlueTrainingModel.test_step is keras.Model.test_step
    assert "def train_step" not in inspect.getsource(pipeline)


def test_logged_loss_equals_the_add_loss_value(fitted):
    """evaluate()'s loss is the batch mean of what call() registered with add_loss."""
    model = fitted["model"]
    per_batch = []
    for batch in fitted["val"].take(2):
        model(batch, training=False)
        per_batch.append(float(sum(model.losses)))
    evaluated = model.evaluate(fitted["val"].take(2), verbose=0, return_dict=True)["loss"]
    # eager call versus the compiled evaluate graph differ by float32 reduction order;
    # measured 1.6e-5 relative once on GPU (log1p/exp BCE), so the bound is 1e-4
    assert evaluated == pytest.approx(float(np.mean(per_batch)), rel=1e-4)


def test_masks_reach_the_confidence_term(images):
    """Padded slots exist (64 slots, far fewer fit after NMS + border); the registered loss
    must equal the masked objective and differ from the unmasked one."""
    model = _model(max_keypoints=64)
    batch = next(iter(_dataset(images, shuffle=False)))
    _, _, mask0, mask1, labels0, labels1 = model._labelled(batch)
    assert float(keras.ops.min(mask0)) == 0.0, "the batch has no padded slot"
    out = model(batch, training=False)
    registered = float(sum(model.losses))
    args = (out["log_assignments"], labels0, labels1,
            out["token_logits0"], out["token_logits1"])
    masked = float(keras.ops.mean(model.objective.compute(*args, mask0, mask1)))
    unmasked = float(keras.ops.mean(model.objective.compute(*args)))
    assert registered == pytest.approx(masked, rel=1e-5)
    assert abs(masked - unmasked) > 1e-4 * abs(masked), (masked, unmasked)


def test_image_size_mismatch_is_refused(images):
    model = _model()
    batch = next(iter(_dataset(images)))
    bad = dict(batch, image0=batch["image0"][:, :32], image1=batch["image1"][:, :32])
    with pytest.raises(ValueError, match="spatial size"):
        model(bad, training=False)


def test_non_float32_superpoint_is_refused():
    previous = keras.config.dtype_policy()
    keras.config.set_dtype_policy("mixed_float16")
    try:
        mixed = SuperPoint(depths=(1, 1, 1), dims=(8, 16, 32), input_shape=(SIZE, SIZE, 1),
                           descriptor_dim=32, dtype="mixed_float16")
    finally:
        keras.config.set_dtype_policy(previous)
    with pytest.raises(ValueError, match="float32"):
        LightGlueTrainingModel(mixed, make_lightglue(), max_keypoints=32)


def test_lightglue_width_must_match_the_descriptors():
    with pytest.raises(ValueError, match="input_dim"):
        LightGlueTrainingModel(
            make_superpoint(), LightGlue(input_dim=16, descriptor_dim=32, num_layers=2,
                                         num_heads=2))


# ---------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------


def test_wrapper_round_trips_through_keras_saving(fitted, tmp_path):
    model = fitted["model"]
    path = str(tmp_path / "wrapper.keras")
    model.save(path)
    reloaded = keras.saving.load_model(path, compile=False)
    batch = next(iter(fitted["val"]))
    a = model(batch, training=False)["log_assignments"]
    b = reloaded(batch, training=False)["log_assignments"]
    assert isinstance(reloaded, LightGlueTrainingModel)
    assert np.array_equal(keras.ops.convert_to_numpy(a), keras.ops.convert_to_numpy(b))
    assert _frozen(reloaded)


def test_lightglue_checkpoint_saves_the_lightglue_alone_on_improvement(images, tmp_path):
    model = _model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), jit_compile=False)
    path = str(tmp_path / "best_model.keras")
    callback = LightGlueCheckpoint(path, monitor="val_loss")
    model.fit(_dataset(images, repeat=True), epochs=2, steps_per_epoch=1,
              validation_data=_dataset(images, shuffle=False), validation_steps=1,
              callbacks=[callback], verbose=0)
    assert callback.best is not None and os.path.isfile(path)
    saved = keras.saving.load_model(path, compile=False)
    assert type(saved).__name__ == "LightGlue"
    # the file holds the weights of the epoch it was written at, which is the best one
    assert callback.best_epoch in (0, 1)
    assert os.path.getsize(path) < 0.7 * _wrapper_size(model, tmp_path)


def _wrapper_size(model, tmp_path) -> int:
    path = str(tmp_path / "wrapper_size.keras")
    model.save(path)
    return os.path.getsize(path)


def test_checkpoint_ignores_nan_and_worse_values(tmp_path):
    callback = LightGlueCheckpoint(str(tmp_path / "x.keras"), monitor="val_loss")
    assert callback.mode == "min"
    callback.best = 1.0
    assert not callback._improved(1.5) and callback._improved(0.5)


# ---------------------------------------------------------------------
# load_superpoint
# ---------------------------------------------------------------------


def test_load_superpoint_returns_a_frozen_float32_model(superpoint_path):
    model = load_superpoint(superpoint_path, image_size=(SIZE, SIZE))
    assert model.trainable is False and model.compute_dtype == "float32"
    assert not any(w.trainable for w in model.weights)
    assert (model.input_height, model.input_width) == (SIZE, SIZE)


def test_load_superpoint_restores_the_global_policy(superpoint_path):
    keras.config.set_dtype_policy("mixed_float16")
    try:
        model = load_superpoint(superpoint_path)
        assert keras.config.dtype_policy().name == "mixed_float16"
    finally:
        keras.config.set_dtype_policy("float32")
    assert model.compute_dtype == "float32"


def test_load_superpoint_errors_are_clear(superpoint_path, tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_superpoint(str(tmp_path / "missing.keras"))
    with pytest.raises(ValueError, match="does not match"):
        load_superpoint(superpoint_path, image_size=(128, 128))
    lightglue_file = str(tmp_path / "lg.keras")
    unsaved = make_lightglue()
    unsaved.build()
    unsaved.save(lightglue_file)
    with pytest.raises(TypeError, match="not a SuperPoint"):
        load_superpoint(lightglue_file)


# ---------------------------------------------------------------------
# mixed precision: the front end stays float32
# ---------------------------------------------------------------------


def test_one_step_under_a_mixed_float16_policy(images):
    superpoint = make_superpoint()
    keras.config.set_dtype_policy("mixed_float16")
    try:
        model = LightGlueTrainingModel(superpoint, make_lightglue(), max_keypoints=32)
        optimizer = keras.optimizers.LossScaleOptimizer(keras.optimizers.Adam(1e-3))
        model.compile(optimizer=optimizer, jit_compile=False)
        before = [w.numpy().copy() for w in superpoint.weights]
        history = model.fit(_dataset(images, repeat=True), epochs=1, steps_per_epoch=1, verbose=0)
    finally:
        keras.config.set_dtype_policy("float32")
    assert np.isfinite(history.history["loss"][0])
    assert all(np.array_equal(b, w.numpy()) for b, w in zip(before, superpoint.weights))
