"""The ``dino`` task's run callbacks and probe wiring: ``train.common.dino_run`` (plan-2026-11-26T224507-e3381378, Phase 4k, Step 4).

Four things are held here, each by a test that sees the failure it prevents:

1. **The compiled teacher EMA is the legacy EMA, bit for bit** (``CompiledTeacherEMA``; D-024, D-033). The legacy ``update_teacher_ema`` and
   ``TeacherEMACallback`` are the oracle. The tolerance is ZERO: a compiled form that formed ``1 - decay`` in float32 inside the graph was 9 to
   12 float32 steps away from the legacy and would have passed any band of 1e-6. It reads the step from ``optimizer.iterations`` (so a resumed
   run continues the schedule where the weights are, where the legacy callback replays its warmup freeze), the decay is a tensor argument
   (a Python float would retrace the graph at every decay), and the whole thing traces once.
2. **The centre file** (``CentreFile``; D-027). ``DINOLoss`` keeps its centre in a variable that no weights file carries, so a task file
   beside the sidecar holds it: written atomically at each epoch end, restored only on a resume, refused when its shape is not the loss's.
3. **The probe is one instrument.** ``build_probe`` is the only place the k-NN probe is configured; epoch zero, every epoch and the reload
   check read that one configuration, so a number from one is comparable to a number from another.
4. **Epoch zero leaves the run alone.** ``model.evaluate`` on a DINO model advances the loss centre (the loss moves it inside ``call``), so
   the epoch-zero control scores the untrained model through ``evaluate_split`` and takes its loss from one eager forward through a
   throwaway loss. The run's centre, weights and random state are untouched.

Sizes: a tiny ``DINOTrainingModel`` two blocks deep (the callbacks do not care about the depth, and 12 blocks cost about a minute of tracing
per fit), 32 px, 64 outputs, fake CIFAR-10 rows (``_dino_world``). The tests write only under ``tmp_path``.
"""

import hashlib
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.dino_loss import DINOLoss
from dl_techniques.models.vision.depth_anything.teacher_ema import TeacherEMACallback, cosine_ema_schedule
from train.common import classification_run, dino_run
from train.dino.knn_eval import KNNEvalCallback, knn_top1_accuracy

from ._dino_world import (  # noqa: F401  (``fake_rows`` is a fixture)
    BANK_BATCHES, BATCH, OUT_DIM, QUERY_BATCHES, build, compile_model, fake_rows, labels_of, make_loss, make_model,
    same_arrays, views_and_labels, weights_of)

REPO_ROOT = Path(__file__).resolve().parents[3]
WARMUP = 266                       # one epoch of Imagenette at batch 32: what ``ema_warmup_epochs=1.0`` resolves to
SCHEDULE = cosine_ema_schedule(0.996, 0.9999, 266 * 60)


class Recording:
    """A schedule that records every ``(post-warmup index, decay)`` it is asked for."""

    def __init__(self, schedule: Any) -> None:
        self.schedule, self.calls = schedule, []

    def __call__(self, step: int) -> float:
        decay = self.schedule(step)
        self.calls.append((int(step), decay))
        return decay


class Silent:
    """A stand-in model for the legacy callback: it accepts an update and does nothing (the schedule records what was asked)."""

    def update_teacher_ema(self, decay: float) -> None:
        pass


def sha256(path: Any) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------
# 1. the compiled teacher EMA
# ---------------------------------------------------------------------

def ema_pair() -> Tuple[Any, Any, Any]:
    """Two identical models and a bound compiled callback on the second."""
    eager, compiled = make_model(), make_model()
    assert same_arrays(weights_of(eager), weights_of(compiled)), "the same seed gives the same weights"
    callback = dino_run.CompiledTeacherEMA(SCHEDULE, warmup_steps=0)
    callback.set_model(compiled)
    callback.on_train_begin()
    return eager, compiled, callback


def decays_over_the_schedule() -> List[float]:
    """Fifty decays spread over the whole schedule, and the edges the legacy method accepts (0 copies the student, 1 freezes the teacher)."""
    schedule = cosine_ema_schedule(0.996, 0.9999, 15960)
    return [schedule(i * 320) for i in range(50)] + [0.0, 0.5, 1.0]


def test_the_compiled_update_equals_the_legacy_eager_update_bit_for_bit_after_every_update():
    eager, compiled, callback = ema_pair()
    initial = weights_of(eager.teacher)
    rng = np.random.default_rng(5)
    worst = 0.0
    for decay in decays_over_the_schedule():
        for student_eager, student_compiled in zip(eager.student.weights, compiled.student.weights):
            moved = student_eager.numpy() + rng.normal(size=student_eager.shape).astype(np.float32) * 0.05
            student_eager.assign(moved)
            student_compiled.assign(moved)
        eager.update_teacher_ema(decay)
        callback.update(decay)
        worst = max(worst, max(float(np.abs(a - b).max()) for a, b in zip(weights_of(eager.teacher), weights_of(compiled.teacher))))
    print(f"EMA PARITY isolated update: {callback.updates} updates, max abs gap of every teacher weight after every update = {worst!r}")
    assert worst == 0.0, "the compiled update is the legacy update, not a number within a band of it"
    assert callback.updates == len(decays_over_the_schedule())
    moved_away = max(float(np.abs(a - b).max()) for a, b in zip(initial, weights_of(compiled.teacher)))
    assert moved_away > 1e-3, "the control: the update did move the teacher, so a gap of 0.0 compares something"


def test_the_decay_is_a_tensor_argument_so_the_function_traces_once_for_any_number_of_decays():
    _, _, callback = ema_pair()
    for decay in decays_over_the_schedule():
        callback.update(decay)
    assert callback.traces == 1 and callback.updates == len(decays_over_the_schedule())
    traces: List[int] = []

    @tf.function
    def by_python_float(decay):  # the control: a Python float is a constant of the graph, so every new value is a new trace
        traces.append(1)
        return decay * 2.0

    for decay in decays_over_the_schedule()[:5]:
        by_python_float(decay)
    assert len(traces) == 5, "the instrument can see a retrace"


def drive(legacy: Any, compiled: Any, model: Any, steps: range) -> Tuple[List[int], List[int]]:
    """Run both callbacks over the absolute ``steps`` of ``model.optimizer`` and return the running update counts of each."""
    legacy_counts, compiled_counts = [], []
    for step in steps:
        model.optimizer.iterations.assign(step)  # what `fit` has done to the optimizer when `on_train_batch_end` runs
        legacy.on_train_batch_end(step - 1)
        compiled.on_train_batch_end(step - 1)
        legacy_counts.append(legacy._post_warmup_count)
        compiled_counts.append(compiled.updates)
    return legacy_counts, compiled_counts


def callbacks_and_model() -> Tuple[Any, Any, Any, Recording, Recording]:
    legacy_schedule, compiled_schedule = Recording(SCHEDULE), Recording(SCHEDULE)
    legacy = TeacherEMACallback(schedule=legacy_schedule, warmup_steps=WARMUP)
    legacy.set_model(Silent())
    model = compile_model(make_model(), make_loss())
    compiled = dino_run.CompiledTeacherEMA(compiled_schedule, warmup_steps=WARMUP)
    compiled.set_model(model)
    compiled.on_train_begin()
    return legacy, compiled, model, legacy_schedule, compiled_schedule


def test_the_decay_sequence_equals_the_legacy_one_over_600_steps_and_the_first_update_follows_the_warmup():
    legacy, compiled, model, legacy_schedule, compiled_schedule = callbacks_and_model()
    legacy_counts, compiled_counts = drive(legacy, compiled, model, range(1, 601))
    assert compiled_schedule.calls == legacy_schedule.calls, "the same post-warmup index and the same decay at every update"
    assert compiled_counts == legacy_counts, "an update at exactly the same steps"
    assert compiled.updates == legacy._post_warmup_count == 600 - WARMUP
    assert compiled_counts.index(1) + 1 == WARMUP + 1, "no update while step <= warmup; the first one is step warmup + 1"
    assert compiled_schedule.calls[0] == (0, 0.996), "and its post-warmup index restarts at 0: the schedule's first decay"
    assert compiled.traces == 1


def test_after_a_resume_the_compiled_callback_continues_the_schedule_where_the_weights_are():
    legacy, compiled, model, legacy_schedule, compiled_schedule = callbacks_and_model()
    k = 300
    legacy._step = k  # the legacy callback as a process that never stopped counting would hold it: the state a resume has to reproduce
    legacy_counts, compiled_counts = drive(legacy, compiled, model, range(k + 1, k + 11))
    assert compiled_schedule.calls == legacy_schedule.calls == [(step - WARMUP - 1, SCHEDULE(step - WARMUP - 1)) for step in range(k + 1, k + 11)]
    assert compiled_counts == legacy_counts == list(range(1, 11)), "an update at every one of the ten steps"
    assert compiled.updates == 10


def test_a_restarted_legacy_callback_replays_the_warmup_freeze_which_is_why_the_compiled_one_reads_the_optimizer():
    restarted = TeacherEMACallback(schedule=Recording(SCHEDULE), warmup_steps=WARMUP)
    restarted.set_model(Silent())
    compiled_schedule = Recording(SCHEDULE)
    model = compile_model(make_model(), make_loss())
    compiled = dino_run.CompiledTeacherEMA(compiled_schedule, warmup_steps=WARMUP)
    compiled.set_model(model)
    compiled.on_train_begin()
    drive(restarted, compiled, model, range(301, 311))
    assert restarted._post_warmup_count == 0, "the legacy callback restarts its private counter, so a resumed run freezes the teacher again"
    assert compiled.updates == 10, "the compiled callback reads the optimizer's step, which the weights file restores"


@pytest.fixture(scope="module")
def parity_fits(fake_rows):
    """Two real 20-step fits of one model (same seed, same arrays, no shuffle), one with each callback: warmup 5, 15 updates each."""
    steps, epochs, warmup = 5, 4, 5
    schedule = cosine_ema_schedule(0.996, 0.9999, steps * epochs)
    x, y = views_and_labels(steps * epochs)

    def one(make_callback: Any) -> Dict[str, Any]:
        model, loss = make_model(), make_loss()
        compile_model(model, loss)
        before = weights_of(model.student)
        callback = make_callback(schedule, warmup)
        history = model.fit(x, y, batch_size=BATCH, epochs=epochs, steps_per_epoch=steps, shuffle=False, callbacks=[callback], verbose=0)
        return {"model": model, "loss": loss, "history": history.history["loss"], "callback": callback, "student_before": before}

    return one(lambda s, w: TeacherEMACallback(schedule=s, warmup_steps=w)), one(lambda s, w: dino_run.CompiledTeacherEMA(s, warmup_steps=w))


def test_a_real_fit_with_each_callback_ends_bitwise_equal_in_the_teacher_the_student_the_centre_and_the_loss(parity_fits):
    legacy, compiled = parity_fits
    teacher_gap = max(float(np.abs(a - b).max()) for a, b in zip(weights_of(legacy["model"].teacher), weights_of(compiled["model"].teacher)))
    student_gap = max(float(np.abs(a - b).max()) for a, b in zip(weights_of(legacy["model"].student), weights_of(compiled["model"].student)))
    centre_gap = float(np.abs(legacy["loss"].center.numpy() - compiled["loss"].center.numpy()).max())
    print(f"EMA PARITY real 20-step fit: teacher {teacher_gap!r} student {student_gap!r} centre {centre_gap!r} "
          f"loss {legacy['history']} against {compiled['history']}")
    assert (teacher_gap, student_gap, centre_gap) == (0.0, 0.0, 0.0)
    assert legacy["history"] == compiled["history"], "the four per-epoch losses are identical to the last digit"
    assert legacy["callback"]._post_warmup_count == compiled["callback"].updates == 15 and compiled["callback"].traces == 1
    # the controls: the fits did something and the EMA did something
    assert not same_arrays(weights_of(compiled["model"].student), compiled["student_before"]), "the student trained"
    assert max(float(np.abs(t - s).max()) for t, s in zip(weights_of(compiled["model"].teacher), weights_of(compiled["model"].student))) > 1e-4
    assert float(np.abs(compiled["loss"].center.numpy()).max()) > 0.0, "the centre moved"


def test_a_model_with_no_teacher_and_a_decay_outside_the_unit_interval_are_refused_and_nothing_self_disables():
    callback = dino_run.CompiledTeacherEMA(SCHEDULE, warmup_steps=0)
    callback.set_model(keras.Sequential([keras.layers.Input((2,)), keras.layers.Dense(1)]))
    with pytest.raises(ValueError, match="teacher"):  # the legacy callback logs one warning and disables itself; a probe that quietly stops is worse
        callback.on_train_begin()
    _, _, bound = ema_pair()
    for decay in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="decay"):
            bound.update(decay)
    with pytest.raises(ValueError, match="warmup"):
        dino_run.CompiledTeacherEMA(SCHEDULE, warmup_steps=-1)


# ---------------------------------------------------------------------
# 2. the centre file
# ---------------------------------------------------------------------

def loss_with_centre(seed: int = 0) -> DINOLoss:
    loss = make_loss()
    loss.center.assign(np.random.default_rng(seed).normal(size=(1, OUT_DIM)).astype(np.float32))
    return loss


def centre_of(loss: DINOLoss) -> np.ndarray:
    return np.array(loss.center.numpy(), copy=True)


def test_the_centre_is_written_at_each_epoch_end_and_the_file_holds_the_loss_centre(tmp_path):
    path = tmp_path / "centre.npy"
    loss = loss_with_centre(1)
    callback = dino_run.CentreFile(loss, path, completed_epochs=None)
    callback.on_epoch_end(0, {})
    saved = np.load(path)
    assert saved.dtype == np.float32 and saved.shape == (1, OUT_DIM)
    np.testing.assert_array_equal(saved, centre_of(loss))
    loss.center.assign(np.full((1, OUT_DIM), 2.5, np.float32))
    callback.on_epoch_end(1, {})
    np.testing.assert_array_equal(np.load(path), np.full((1, OUT_DIM), 2.5, np.float32))
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["centre.npy"], "no temporary file is left beside it"


def test_a_failing_weights_write_after_the_centre_write_stops_the_run_so_the_file_is_at_most_one_epoch_ahead_and_a_resume_restores_it_as_it_is(
        tmp_path, fake_rows):
    """D-104: the file carries no epoch stamp, and the bound that makes it unnecessary is the callback order: the centre is written BEFORE the weights backup,
    and a backup write that fails raises, which ends ``fit``, so no later epoch writes a newer centre over weights that were never saved."""
    path = tmp_path / "centre.npy"
    loss, model = make_loss(), make_model()
    compile_model(model, loss)
    ends: List[np.ndarray] = []

    class BackupThatFails(keras.callbacks.Callback):
        """Stands for the weights backup: it runs after ``CentreFile`` at each epoch end and fails at the second one (the first one 'succeeds')."""

        def on_epoch_end(self, epoch, logs=None):
            ends.append(centre_of(loss))
            if epoch == 1:
                raise OSError("the weights backup could not be written")

    x, y = views_and_labels(2)
    with pytest.raises(OSError, match="weights backup"):
        model.fit(x, y, batch_size=BATCH, epochs=4, shuffle=False, verbose=0, callbacks=[dino_run.CentreFile(loss, path, completed_epochs=None), BackupThatFails()])
    assert len(ends) == 2, "no third epoch ran: a failed backup ends the run"
    assert not np.array_equal(ends[0], ends[1]), "the control: the centre moved between the two epoch ends"
    np.testing.assert_array_equal(np.load(path), ends[1], err_msg="the file holds the LATEST epoch end's centre, one epoch ahead of the last backup")
    # The backup that settled is the one of epoch 0 (one finished epoch): the resume restores the file's centre whatever epoch it is from.
    resumed = make_loss()
    assert dino_run.CentreFile(resumed, path, completed_epochs=1).restore() is True
    np.testing.assert_array_equal(centre_of(resumed), ends[1])


def test_a_resume_restores_the_centre_exactly_and_a_fresh_run_reads_no_file(tmp_path, monkeypatch):
    path = tmp_path / "centre.npy"
    written = loss_with_centre(2)
    dino_run.CentreFile(written, path, completed_epochs=None).on_epoch_end(0, {})
    resumed = make_loss()
    callback = dino_run.CentreFile(resumed, path, completed_epochs=3)
    assert not centre_of(resumed).any()
    callback.on_train_begin()
    np.testing.assert_array_equal(centre_of(resumed), centre_of(written))  # bitwise

    def never(*args: Any, **kwargs: Any):
        raise AssertionError("a fresh run read the centre file")

    fresh = make_loss()
    monkeypatch.setattr(np, "load", never)
    dino_run.CentreFile(fresh, path, completed_epochs=None).on_train_begin()
    dino_run.CentreFile(fresh, path, completed_epochs=0).on_train_begin()  # a resume with no finished epoch has nothing to restore either
    assert not centre_of(fresh).any(), "the stale file did not leak into a run that did not resume"


def test_an_interrupted_write_leaves_the_old_file_whole_and_the_next_write_succeeds(tmp_path, monkeypatch):
    path = tmp_path / "centre.npy"
    old = loss_with_centre(3)
    callback = dino_run.CentreFile(old, path, completed_epochs=None)
    callback.on_epoch_end(0, {})
    before, old_centre = sha256(path), centre_of(old)
    old.center.assign(np.full((1, OUT_DIM), 9.0, np.float32))
    real_save = np.save

    def torn(handle: Any, array: Any, **kwargs: Any) -> None:  # the disk fills up while the temporary file is being written
        handle.write(b"torn")
        raise OSError("injected: no space left on device")

    monkeypatch.setattr(np, "save", torn)
    with pytest.raises(OSError, match="injected"):
        callback.on_epoch_end(1, {})
    assert sha256(path) == before, "the file of the last epoch is byte-identical"
    np.testing.assert_array_equal(np.load(path), old_centre)

    monkeypatch.setattr(np, "save", real_save)
    real_replace = os.replace

    def refuse(source: Any, target: Any) -> None:  # the process cannot move the finished temporary file into place
        raise OSError("injected: replace failed")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError, match="replace failed"):
        callback.on_epoch_end(2, {})
    assert sha256(path) == before
    monkeypatch.setattr(os, "replace", real_replace)
    callback.on_epoch_end(3, {})
    np.testing.assert_array_equal(np.load(path), np.full((1, OUT_DIM), 9.0, np.float32))
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["centre.npy"], "the leftover temporary file was overwritten and moved"


def test_a_real_sigkill_while_the_temporary_file_is_half_written_leaves_the_old_file_whole(tmp_path):
    path = tmp_path / "centre.npy"
    old = loss_with_centre(4)
    dino_run.CentreFile(old, path, completed_epochs=None).on_epoch_end(0, {})
    before, old_centre = sha256(path), centre_of(old)
    child_code = (
        "import sys, time, numpy as np\n"
        "from dl_techniques.losses.dino_loss import DINOLoss\n"
        "from train.common import dino_run\n"
        f"loss = DINOLoss(out_dim={OUT_DIM})\n"
        f"loss.center.assign(np.full((1, {OUT_DIM}), 7.0, np.float32))\n"
        "def half(handle, array, **kwargs):\n"
        "    handle.write(b'torn'); handle.flush()\n"
        "    print('HALF', flush=True); time.sleep(600)\n"
        "np.save = half\n"
        "dino_run.CentreFile(loss, sys.argv[1], completed_epochs=None).on_epoch_end(0, {})\n")
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(path)], stdout=subprocess.PIPE, text=True, cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""})
    try:
        seen = None
        for line in child.stdout:
            if line.strip() == "HALF":
                seen = line.strip()
                os.kill(child.pid, signal.SIGKILL)  # only the PID this test started
                break
    finally:
        if child.poll() is None:
            os.kill(child.pid, signal.SIGKILL)
        child.wait()
    assert seen == "HALF" and child.returncode == -signal.SIGKILL, "the child was killed by the signal, not by an error of its own"
    assert sha256(path) == before
    np.testing.assert_array_equal(np.load(path), old_centre)
    assert (tmp_path / "centre.partial.npy").read_bytes() == b"torn", "the torn temporary file is what the kill left"
    dino_run.CentreFile(old, path, completed_epochs=None).on_epoch_end(1, {})  # the next write goes over the leftover
    np.testing.assert_array_equal(np.load(path), old_centre)
    assert not (tmp_path / "centre.partial.npy").exists()


def test_a_file_that_is_not_the_losss_centre_is_refused_and_leaves_the_loss_untouched(tmp_path):
    path = tmp_path / "centre.npy"
    for bad, reason in ((np.zeros((1, OUT_DIM // 2), np.float32), "32"), (np.zeros((OUT_DIM,), np.float32), "shape"),
                        (np.zeros((2, OUT_DIM), np.float32), "shape"), (np.full((1, OUT_DIM), np.nan, np.float32), "finite")):
        np.save(path, bad)
        loss = make_loss()
        with pytest.raises(ValueError, match=reason) as caught:
            dino_run.CentreFile(loss, path, completed_epochs=2).on_train_begin()
        assert str(path) in str(caught.value) or "centre.npy" in str(caught.value)
        assert not centre_of(loss).any()
    np.save(path, np.zeros((1, OUT_DIM // 2), np.float32))
    with pytest.raises(ValueError, match=str(OUT_DIM)):
        dino_run.CentreFile(make_loss(), path, completed_epochs=2).on_train_begin()  # the message names the out_dim of the loss


def test_a_resume_after_two_epochs_with_no_centre_file_is_refused_and_after_one_it_starts_from_zero_and_says_so(tmp_path, monkeypatch):
    path = tmp_path / "centre.npy"
    said: List[str] = []

    class Said:
        def warning(self, message: str) -> None:
            said.append(message)

        def info(self, message: str) -> None:
            pass

    monkeypatch.setattr(dino_run, "logger", Said())
    with pytest.raises(FileNotFoundError, match="centre.npy"):
        dino_run.CentreFile(make_loss(), path, completed_epochs=2).on_train_begin()
    with pytest.raises(FileNotFoundError):
        dino_run.CentreFile(make_loss(), path, completed_epochs=7).on_train_begin()
    assert not said
    one = make_loss()
    dino_run.CentreFile(one, path, completed_epochs=1).on_train_begin()
    assert not centre_of(one).any() and len(said) == 1 and "zero" in said[0]
    dino_run.CentreFile(make_loss(), path, completed_epochs=0).on_train_begin()
    assert len(said) == 1, "no finished epoch, nothing to say"


def test_the_centre_file_is_a_npy_path_and_the_epoch_count_is_a_count(tmp_path):
    with pytest.raises(ValueError, match="npy"):
        dino_run.CentreFile(make_loss(), tmp_path / "centre.txt", completed_epochs=None)
    for bad in (-1, True, 1.5):
        with pytest.raises((TypeError, ValueError), match="completed_epochs"):
            dino_run.CentreFile(make_loss(), tmp_path / "centre.npy", completed_epochs=bad)


# ---------------------------------------------------------------------
# 3 and 4. the probe, epoch zero and the reload seam
# ---------------------------------------------------------------------

class CountingLoss(DINOLoss):
    """A ``DINOLoss`` that counts its executions (a Python counter: any call, eager or traced, adds one)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.executions = 0

    def call(self, y_true: Any, y_pred: Any) -> Any:
        self.executions += 1
        return super().call(y_true, y_pred)


@pytest.fixture(scope="module")
def data(fake_rows):
    return build()


@pytest.fixture(scope="module")
def zero_world(data):
    """A built model compiled with a counting loss whose centre is NOT zero, so that "unchanged" says something."""
    loss = CountingLoss(out_dim=OUT_DIM, student_temp=0.2, teacher_temp=0.06, center_momentum=0.8)
    loss.center.assign(np.random.default_rng(9).normal(size=(1, OUT_DIM)).astype(np.float32))
    model = compile_model(make_model(), loss)
    return model, loss


def test_the_probe_is_one_configuration_over_the_cached_probe_streams(data, zero_world):
    _, loss = zero_world
    probe = dino_run.build_probe(data, loss)
    assert isinstance(probe, KNNEvalCallback) and probe.compiled is True
    assert probe.control_json_path is None and probe.random_init_repeats == 0 and probe.every_n_epochs == 1
    assert probe.bank_dataset is data.probe_bank_ds and probe.query_dataset is data.probe_query_ds
    assert (probe.bank_batches, probe.query_batches) == (data.bank_batches, data.query_batches) == (BANK_BATCHES, min(QUERY_BATCHES, data.val_steps))
    assert probe.ks == (10, 20) and probe.dino_loss is loss
    assert "dino_knn_top1_k20" in probe.log_keys
    assert dino_run.build_probe(data, loss) is not probe, "a new object each time: the extractor of one model is never reused for another"


def test_epoch_zero_leaves_the_centre_the_weights_the_step_and_the_random_state_alone(data, zero_world, monkeypatch):
    model, loss = zero_world
    centre, teacher, student = centre_of(loss), weights_of(model.teacher), weights_of(model.student)
    steps = int(model.optimizer.iterations)
    generator = keras.src.random.seed_generator.global_seed_generator()
    state = np.array(generator.state.numpy(), copy=True)
    for name in ("evaluate", "test_on_batch", "predict", "predict_on_batch", "train_on_batch", "fit"):
        def boom(*args: Any, _name: str = name, **kwargs: Any):
            raise AssertionError(f"epoch zero called model.{_name}: the loss moves its centre inside call()")

        monkeypatch.setattr(keras.Model, name, boom)
    dino_run.epoch_zero(model, data, loss)
    np.testing.assert_array_equal(centre_of(loss), centre)  # bitwise
    assert loss.executions == 0, "the run's own loss was never executed"
    assert same_arrays(weights_of(model.teacher), teacher) and same_arrays(weights_of(model.student), student)
    assert int(model.optimizer.iterations) == steps
    np.testing.assert_array_equal(generator.state.numpy(), state, err_msg="a diagnostic pass must not perturb the run's random state")


def test_epoch_zero_is_scored_with_the_probes_own_extractor_so_it_equals_the_per_epoch_probe_on_the_same_weights(data, zero_world):
    model, loss = zero_world
    result = dino_run.epoch_zero(model, data, loss)
    probe = dino_run.build_probe(data, loss)
    probe.set_model(model)
    logs: Dict[str, float] = {}
    probe.on_epoch_end(0, logs)
    assert set(result.probe) == set(probe.log_keys) and {key: logs[key] for key in probe.log_keys} == result.probe, \
        "every key of the control equals the epoch's, to the last digit: one extractor, one instrument"
    assert all(isinstance(value, float) and np.isfinite(value) for value in result.probe.values())
    assert result.probe["dino_teacher_entropy_is_centered"] == 1.0


def test_the_epoch_zero_loss_is_one_eager_forward_through_a_throwaway_loss_of_the_runs_hyperparameters(data, zero_world):
    model, loss = zero_world
    result = dino_run.epoch_zero(model, data, loss)
    views, labels = next(iter(data.loss_batch_ds))
    reference_loss = DINOLoss(out_dim=OUT_DIM, student_temp=0.2, teacher_temp=0.06, center_momentum=0.8)
    reference = compile_model(make_model(), reference_loss).evaluate(views, labels, batch_size=len(views), verbose=0)
    default = compile_model(make_model(), make_loss()).evaluate(views, labels, batch_size=len(views), verbose=0)
    print(f"EPOCH ZERO loss {result.loss!r}; evaluate of an identical model with the same loss {float(reference)!r}")
    assert np.isfinite(result.loss) and result.loss == pytest.approx(float(reference), rel=1e-5)
    assert abs(float(default) - float(reference)) > 1e-3, "the control: another student temperature gives another loss, so the hyperparameters matter"
    runs_centre = centre_of(loss)
    loss.center.assign(np.zeros((1, OUT_DIM), np.float32))
    try:
        assert dino_run.epoch_zero(model, data, loss).loss == result.loss, "it is the loss of a zero centre, whatever the run's centre holds"
    finally:
        loss.center.assign(runs_centre)


def test_the_constant_feature_baseline_is_the_k_nn_of_identical_features_against_the_real_labels(data, zero_world):
    model, loss = zero_world
    result = dino_run.epoch_zero(model, data, loss)
    bank, queries = labels_of(data.probe_bank_ds), labels_of(data.probe_query_ds)
    expected = {f"dino_knn_top1_k{k}": knn_top1_accuracy(np.ones((len(bank), 3)), bank, np.ones((len(queries), 3)), queries, k=k)
                for k in (10, 20)}
    assert result.baseline == expected and set(result.baseline) == {"dino_knn_top1_k10", "dino_knn_top1_k20"}
    assert all(0.0 <= value <= 1.0 for value in result.baseline.values())


def test_a_model_whose_features_are_exactly_zero_scores_the_constant_feature_baseline(data):
    model = make_model(seed=11)
    for weight in model.student.weights:
        weight.assign(np.zeros(weight.shape, np.float32))
    features = keras.Model(model.student.inputs[0], model.student.head.input)(next(iter(data.select_ds))[0])
    assert float(np.abs(features.numpy()).max()) == 0.0, "the control: the features really are exactly zero"
    result = dino_run.epoch_zero(model, data, make_loss())
    assert result.probe["dino_feat_mean_cos"] == 0.0
    for key, baseline in result.baseline.items():
        assert result.probe[key] == baseline, f"{key}: zero features carry no information, so they score what the labels alone score"


def test_the_knn_seam_scores_a_dataset_through_a_fresh_probe_and_never_through_evaluate(data, zero_world, monkeypatch):
    model, loss = zero_world
    centre = centre_of(loss)
    probe = dino_run.build_probe(data, loss)
    probe.set_model(model)
    expected = probe.evaluate_split(data.test_ds, data.test_steps)
    monkeypatch.setattr(keras.Model, "evaluate", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("evaluate was called")))
    scored = dino_run.knn_of(model, data.test_ds, data=data, loss=loss)
    assert scored == {key: float(value) for key, value in expected.items()} and set(scored) == set(probe.log_keys)
    assert all(isinstance(value, float) for value in scored.values())
    np.testing.assert_array_equal(centre_of(loss), centre)
    with pytest.raises(ValueError, match="length"):
        dino_run.knn_of(model, data.train_ds, data=data, loss=loss)  # an endless stream has no pass to score


def test_the_knn_seam_is_the_reload_check_of_a_best_checkpoint_and_the_reloaded_model_scores_the_same(data, tmp_path):
    loss = make_loss()
    model = compile_model(make_model(seed=7), loss)
    model.optimizer.build(model.trainable_variables)  # a best checkpoint is saved after training, so its optimizer holds its slots
    model.save(str(tmp_path / "best_model.keras"))
    reloaded, load_error, gap = classification_run.check_best_checkpoint(
        model, tmp_path, data.test_ds, lambda m, ds: dino_run.knn_of(m, ds, data=data, loss=loss))
    print(f"RELOAD gap of every k-NN key between the in-memory model and best_model.keras: {gap!r}")
    assert load_error is None and reloaded is not None and set(reloaded) >= {"dino_knn_top1_k10", "dino_knn_top1_k20"}
    assert gap is not None and gap <= classification_run.WEIGHT_MISMATCH_TOLERANCE
