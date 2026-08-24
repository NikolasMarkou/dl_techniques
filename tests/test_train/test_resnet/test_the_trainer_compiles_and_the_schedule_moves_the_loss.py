"""Guards for ``train.resnet.train_resnet``'s compile spec and DS weight schedule.

Two MEASURED defects lived here, both in the gap left by the fact that nothing
in ``tests/test_train/test_resnet/`` had ever touched ``compile``,
``create_callbacks`` or ``DeepSupervisionWeightScheduler``.

1. The single-output branch -- the DEFAULT configuration -- passed
   ``metrics = ['accuracy', 'top_5_accuracy']``. ``'top_5_accuracy'`` is not a
   Keras 3 alias. ``compile()`` ACCEPTS it (Keras 3 resolves metrics lazily);
   the FIRST ``fit`` step then dies with::

       ValueError: Could not interpret metric identifier: top_5_accuracy

   So the trainer could never reach a training step in its default
   configuration. Note the review that raised this reported the raise at
   ``compile()``; re-measured here, ``compile()`` returns cleanly and the raise
   lands one step later, inside ``fit``. That is precisely why an arm that only
   compiles would NOT see this defect, and why every arm below runs a real
   ``fit`` step.

2. ``DeepSupervisionWeightScheduler.on_epoch_begin`` did
   ``self.model.loss_weights = new_weights``. Keras 3 ``Model`` has no
   ``loss_weights`` attribute -- reading it raises
   ``AttributeError: 'Functional' object has no attribute 'loss_weights'``
   (MEASURED) -- and ``CompileLoss`` snapshots the weights into
   ``_flat_losses`` when it builds. The assignment created a dead Python
   attribute that nothing read, so the deep-supervision weight SCHEDULE, the
   entire point of the feature, never reached the loss. The fix makes the
   weights non-trainable ``keras.Variable`` objects that
   ``CompileLoss.call``'s ``ops.multiply(value, loss_weight)`` reads live; see
   the ``D-020`` anchors in ``train_resnet.py``.

Why each arm is not satisfied by construction:

``test_the_default_single_output_spec_reaches_a_fit_step``
    The RED arm for defect 1. It calls ``build_compile_spec`` -- it does NOT
    restate the metrics list -- so if the source ever regains
    ``'top_5_accuracy'`` this arm fails, with the text quoted above.
``test_the_deep_supervision_spec_reaches_a_fit_step``
    The multi-output branch was already fine; this pins it so a shared repair
    cannot break it.
``test_the_history_keys_are_exactly_what_create_callbacks_plots``
    The three-way agreement between the compiled metric names, the history
    keys and ``PLOTTED_METRIC_NAMES``. ``'top_5_accuracy'`` vs
    ``'top5_accuracy'`` was a live disagreement between the trainer and its
    plotter; nothing checked it.
``test_the_schedule_actually_moves_the_weighted_loss``
    The RED arm for defect 2, and deliberately NOT an "it ran without raising"
    assertion -- that is exactly the shape of assertion the defect survived
    for. The optimizer runs at ``learning_rate=0.0`` so the model's weights,
    and therefore every per-output UNWEIGHTED loss, are identical in both
    epochs. The only thing that can move the total loss between epoch 1 and
    epoch 2 is the weight schedule.
``test_a_constant_schedule_leaves_the_loss_where_it_was``
    The anti-vacuity pair for the arm above. Same model, same frozen
    optimizer, same callback -- only the schedule is swapped for
    ``constant_equal``. If the "loss moved" arm were passing for some reason
    OTHER than the schedule (a non-deterministic forward, a BatchNorm update,
    a live learning rate), this arm would move too and fail. It measures
    exactly 0.0 drift.
``test_the_scheduler_refuses_to_be_built_without_its_variables``
    Pins the fix's own failure mode: the schedule can no longer be silently
    disconnected. Passing ``None`` is what the OLD wiring effectively did.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.resnet import create_resnet
from dl_techniques.utils.deep_supervision import get_model_output_info

# Import the module, not just the names: every arm must exercise the SOURCE's
# own compile specification. Restating the metrics list in this file would
# make the guard stop tracking the code it guards.
import train.resnet.train_resnet as train_resnet
from train.resnet.train_resnet import (
    PLOTTED_METRIC_NAMES,
    DeepSupervisionWeightScheduler,
    TrainingConfig,
    build_compile_spec,
)

INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
BATCH = 4

# Two epochs at learning_rate=0.0: enough for the scheduler to fire at
# progress 0.0 and progress 1.0, with the model frozen in between.
FROZEN_EPOCHS = 2


def _resnet(enable_deep_supervision: bool) -> keras.Model:
    return create_resnet(
        variant="resnet18",
        num_classes=NUM_CLASSES,
        input_shape=INPUT_SHAPE,
        enable_deep_supervision=enable_deep_supervision,
    )


def _batch(num_outputs: int):
    rng = np.random.RandomState(1234)
    x = rng.randn(BATCH, *INPUT_SHAPE).astype("float32")
    y = rng.randint(0, NUM_CLASSES, size=(BATCH,)).astype("int32")
    if num_outputs > 1:
        return x, tuple(y for _ in range(num_outputs))
    return x, y


def _ds_config(schedule_type: str, tmp_path) -> TrainingConfig:
    """A valid ``TrainingConfig``. ``__post_init__`` requires the data dirs to
    EXIST, so they are created under ``tmp_path``; nothing outside it is
    touched and repo-root ``results/`` is never involved."""
    train_dir = tmp_path / "imagenet_train"
    val_dir = tmp_path / "imagenet_val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    return TrainingConfig(
        train_data_dir=str(train_dir),
        val_data_dir=str(val_dir),
        output_dir=str(tmp_path / "run_output"),
        experiment_name="guard",
        epochs=FROZEN_EPOCHS,
        enable_deep_supervision=True,
        deep_supervision_schedule_type=schedule_type,
        deep_supervision_schedule_config={},
    )


def _fit_one_epoch(model, num_outputs, epochs=1, callbacks=None, learning_rate=1e-4):
    x, y = _batch(num_outputs)
    return model.fit(
        x,
        y,
        epochs=epochs,
        batch_size=BATCH,
        verbose=0,
        callbacks=callbacks or [],
    )


# ---------------------------------------------------------------------------
# Defect 1 -- the compile spec must survive a real fit step in BOTH branches
# ---------------------------------------------------------------------------


def test_the_default_single_output_spec_reaches_a_fit_step():
    """The DEFAULT (deep supervision OFF) configuration must reach ``fit``."""
    model = _resnet(enable_deep_supervision=False)
    info = get_model_output_info(model, input_shape=INPUT_SHAPE)
    assert info["num_outputs"] == 1, info

    spec = build_compile_spec(info["has_deep_supervision"], info["num_outputs"])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-4), **spec)

    history = _fit_one_epoch(model, 1)

    assert "top5_accuracy" in history.history, (
        "the single-output branch did not produce a 'top5_accuracy' history "
        f"key; got {sorted(history.history)}. If this arm instead FAILED with "
        "'Could not interpret metric identifier: top_5_accuracy', the "
        "non-alias metric string is back in build_compile_spec()."
    )
    assert "accuracy" in history.history, sorted(history.history)


def test_the_deep_supervision_spec_reaches_a_fit_step():
    """The deep-supervision branch must reach ``fit`` and name its metrics."""
    model = _resnet(enable_deep_supervision=True)
    info = get_model_output_info(model, input_shape=INPUT_SHAPE)
    num_outputs = info["num_outputs"]
    assert num_outputs > 1, info

    spec = build_compile_spec(info["has_deep_supervision"], num_outputs)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-4), **spec)

    history = _fit_one_epoch(model, num_outputs)

    for key in ("primary_accuracy", "primary_top5_accuracy"):
        assert key in history.history, (
            f"missing history key {key!r}; got {sorted(history.history)}"
        )


def test_the_history_keys_are_exactly_what_create_callbacks_plots():
    """Compiled metric names, history keys and the plotted list must agree.

    The trainer used to compile ``'top_5_accuracy'`` while
    ``create_callbacks`` plotted ``'top5_accuracy'`` -- two spellings, no
    overlap, nothing checking. This arm pins that every accuracy-family key a
    real ``fit`` produces, in EITHER branch, is a name the plot callback asks
    for.
    """
    produced = set()

    for deep_supervision in (False, True):
        model = _resnet(enable_deep_supervision=deep_supervision)
        info = get_model_output_info(model, input_shape=INPUT_SHAPE)
        spec = build_compile_spec(info["has_deep_supervision"], info["num_outputs"])
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-4), **spec)
        history = _fit_one_epoch(model, info["num_outputs"])
        produced |= {k for k in history.history if "accuracy" in k}

    missing = sorted(produced - set(PLOTTED_METRIC_NAMES))
    assert not missing, (
        f"fit produced accuracy metric(s) {missing} that "
        f"PLOTTED_METRIC_NAMES {PLOTTED_METRIC_NAMES} does not plot -- the "
        "compile spec and create_callbacks have drifted apart again"
    )
    # And the reverse direction: every plotted name is actually produced by
    # one of the two branches, so the plot list cannot rot into fiction.
    unproduced = sorted(set(PLOTTED_METRIC_NAMES) - produced)
    assert not unproduced, (
        f"PLOTTED_METRIC_NAMES asks to plot {unproduced}, which no branch of "
        f"build_compile_spec produces; produced={sorted(produced)}"
    )


# ---------------------------------------------------------------------------
# Defect 2 -- the schedule must actually reach the loss
# ---------------------------------------------------------------------------


class _RecordEpochLoss(keras.callbacks.Callback):
    """Record the total ``loss`` reported at the end of each epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.losses: list = []

    def on_epoch_end(self, epoch, logs=None) -> None:
        self.losses.append(float((logs or {})["loss"]))


def _run_frozen_two_epochs(schedule_type: str, tmp_path):
    """Fit two epochs at ``learning_rate=0.0`` under ``schedule_type``.

    Freezing the optimizer is what makes the measurement clean: every
    per-output UNWEIGHTED loss is identical in both epochs, so any change in
    the reported total loss can only come from the loss WEIGHTS.

    :param schedule_type: a ``ScheduleType`` value, e.g. ``linear_low_to_high``.
    :returns: ``(losses, weights_seen)`` -- the per-epoch total loss and the
        weight vector the scheduler wrote at the start of each epoch.
    """
    keras.utils.set_random_seed(20260824)
    model = _resnet(enable_deep_supervision=True)
    info = get_model_output_info(model, input_shape=INPUT_SHAPE)
    num_outputs = info["num_outputs"]
    spec = build_compile_spec(True, num_outputs)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0), **spec)

    config = _ds_config(schedule_type, tmp_path)
    scheduler = DeepSupervisionWeightScheduler(
        config, num_outputs, spec["loss_weights"]
    )
    recorder = _RecordEpochLoss()

    weights_seen: list = []

    class _Snapshot(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            weights_seen.append([float(v) for v in spec["loss_weights"]])

    _fit_one_epoch(
        model,
        num_outputs,
        epochs=FROZEN_EPOCHS,
        callbacks=[scheduler, _Snapshot(), recorder],
    )
    return recorder.losses, weights_seen


def test_the_schedule_actually_moves_the_weighted_loss(tmp_path):
    """A changing schedule must change the total loss the model reports."""
    losses, weights_seen = _run_frozen_two_epochs("linear_low_to_high", tmp_path)

    assert len(losses) == FROZEN_EPOCHS, losses
    assert weights_seen[0] != weights_seen[1], (
        "the linear_low_to_high schedule wrote the same weights in both "
        f"epochs ({weights_seen}); the arm below cannot mean anything"
    )
    drift = abs(losses[1] - losses[0])
    assert drift > 1e-6, (
        f"the deep-supervision weight schedule did NOT reach the loss: with "
        f"the model frozen at learning_rate=0.0 the reported loss went "
        f"{losses[0]!r} -> {losses[1]!r} (drift {drift:.3e}) while the weights "
        f"went {weights_seen[0]} -> {weights_seen[1]}. This is the original "
        "defect: on_epoch_begin writing to a dead model attribute."
    )


def test_a_constant_schedule_leaves_the_loss_where_it_was(tmp_path):
    """Anti-vacuity pair: with the weights frozen, the loss must not move.

    Same model, same frozen optimizer, same real callback -- only the schedule
    differs. If the arm above passed for any reason other than the weights
    (a non-deterministic forward, a live learning rate, a BatchNorm update),
    this arm would move too.
    """
    losses, weights_seen = _run_frozen_two_epochs("constant_equal", tmp_path)

    assert weights_seen[0] == weights_seen[1], (
        f"constant_equal was expected to hold the weights fixed; got "
        f"{weights_seen}"
    )
    drift = abs(losses[1] - losses[0])
    assert drift == pytest.approx(0.0, abs=1e-6), (
        f"with constant weights and learning_rate=0.0 the loss still moved "
        f"{losses[0]!r} -> {losses[1]!r} (drift {drift:.3e}); the "
        "'schedule moves the loss' arm above is therefore not measuring the "
        "schedule"
    )


def test_the_scheduler_refuses_to_be_built_without_its_variables(tmp_path):
    """The schedule can no longer be silently disconnected from the loss."""
    config = _ds_config("linear_low_to_high", tmp_path)

    with pytest.raises(ValueError, match="build_compile_spec"):
        DeepSupervisionWeightScheduler(config, 4, None)

    with pytest.raises(ValueError, match="must match"):
        DeepSupervisionWeightScheduler(
            config, 4, build_compile_spec(True, 2)["loss_weights"]
        )


def test_create_callbacks_forwards_the_loss_weight_variables(monkeypatch, tmp_path):
    """``create_callbacks`` must hand the scheduler the compiled variables.

    Without this arm the trainer could compile live variables and still build
    a scheduler pointed at nothing -- the same silent no-op in a new place.
    ``create_common_callbacks`` is stubbed so this writes nothing to disk and
    NEVER touches the repo-root ``results/`` tree.
    """
    monkeypatch.setattr(
        train_resnet, "create_common_callbacks", lambda **kw: ([], str(tmp_path))
    )
    spec = build_compile_spec(True, 4)
    config = _ds_config("linear_low_to_high", tmp_path)

    callbacks, _ = train_resnet.create_callbacks(
        config, 4, loss_weight_vars=spec["loss_weights"]
    )
    schedulers = [
        c for c in callbacks if isinstance(c, DeepSupervisionWeightScheduler)
    ]
    assert len(schedulers) == 1, [type(c).__name__ for c in callbacks]
    assert schedulers[0].loss_weight_vars is spec["loss_weights"], (
        "create_callbacks built the scheduler around a DIFFERENT weight list "
        "than the one compiled into the loss"
    )
