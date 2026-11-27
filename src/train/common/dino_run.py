"""What a DINO run needs beside stock ``fit``: a compiled teacher EMA, a centre file, and the k-NN probe's wiring.

``DINOTrainingModel`` has only ``call`` and ``update_teacher_ema``, and ``DINOLoss`` keeps its centre in a variable that its own ``call``
moves, so a run is stock ``fit`` plus three things: a callback that moves the teacher toward the student after every batch, a way to keep the
centre across a kill (no weights file carries it, measured), and a k-NN probe that is the validation signal and never touches the loss. The
legacy trainer (``train.dino.train_dino``) does the first with ``TeacherEMACallback`` and the third with ``KNNEvalCallback``; this module
holds the versions the front door runs and ties them to the legacy ones, which stay the oracle.

Callbacks subclass Keras classes and the module imports TensorFlow, so it cannot live in ``train.api``, which imports it inside functions only.

Interface contract
------------------
``CompiledTeacherEMA(schedule, warmup_steps=0, log_every=0)``
    A ``keras.callbacks.Callback``: after every training batch moves every teacher weight toward its student weight,
    ``teacher <- decay * teacher + (1 - decay) * student``, with ONE ``tf.function`` over all weight pairs.
    :param schedule: ``post-warmup step -> decay`` (``cosine_ema_schedule(...)``), as ``TeacherEMACallback`` takes it.
    :param warmup_steps: the teacher is not updated while ``optimizer.iterations <= warmup_steps``; the schedule's index restarts at 0 after it.
    :param log_every: log the decay every that many updates (0: never).
    Reads the step from ``model.optimizer.iterations`` at ``on_train_batch_end``, where Keras has already added the batch (so the value is the
    legacy callback's 1-based step with no offset) and keeps no counter of its own: a resumed run, whose optimizer step the weights file
    restores, continues the schedule where the weights are. The legacy callback restarts its private counter at 0 and replays the warmup freeze.
    ``update(decay)`` applies one update (what ``on_train_batch_end`` calls); ``updates`` counts them and ``traces`` counts how often the
    function was traced (1 for a run). The result equals ``DINOTrainingModel.update_teacher_ema`` bit for bit (D-024).
    :raises ValueError: at ``on_train_begin`` for a model with no ``teacher`` and ``student`` of equally many weights (the callback never
        disables itself: a teacher that quietly stops moving looks like a working run), for a decay outside ``[0, 1]`` and for a negative warmup.

``CentreFile(loss, path, *, completed_epochs)``
    A ``keras.callbacks.Callback`` that keeps ``loss.center`` in the ``.npy`` file ``path``: written atomically at every ``on_epoch_end`` (a
    temporary file of the same directory, flushed, fsynced and moved over the old file, then the directory fsynced), restored in
    ``on_train_begin`` by assignment.
    :param completed_epochs: ``None`` for a fresh run, which reads no file; the number of epochs a resumed run has finished otherwise. 0 reads
        no file either (nothing was trained). A resume after one or more epochs restores the file; a missing file is refused after two or more
        (a zero centre would silently change the teacher distribution) and starts from zero with a warning after exactly one (the kill came
        between the weights backup and the first write).
    :raises ValueError: for a path that is not ``.npy``, a negative or non-``int`` epoch count, and a file whose shape is not the loss's
        ``(1, out_dim)`` or that holds a non-finite value (the loss is left untouched). :raises FileNotFoundError: see above. :raises OSError:
        a write that fails leaves the old file whole and raises (a stale centre must not be silent).
    ``write()`` and ``restore()`` are the two actions; ``restore`` returns whether a file was read.

``build_probe(data, loss)``
    :param data: a ``train.common.dino_data.DinoBatches``. :param loss: the run's ``DINOLoss`` (its centre and temperature centre and sharpen the
        teacher distribution whose entropy the probe reports).
    :return: a new ``KNNEvalCallback`` over the data's cached bank and query streams: compiled extractor (D-025), ``k`` 10 and 20, every epoch,
        no random-init control file. The ONE place the probe is configured: epoch zero, every epoch and the reload check call it.

``constant_feature_baseline(data)``
    :return: ``{dino_knn_top1_k<k>: accuracy}`` of CONSTANT features against the real labels of the data's bank and queries: what a representation
        that carries no information scores. Labels only: no model, so a resumed run (which never calls ``epoch_zero``) reads it too.

``epoch_zero(model, data, loss)``
    :return: an :class:`EpochZero`: ``probe`` (every log key of the probe on the untrained model, through ``evaluate_split``), ``baseline``
        (the k-NN accuracy of constant features against the real labels, per ``k``: what a representation that carries no information scores) and
        ``loss`` (the DINO loss of one eager forward of ``model`` on ``data.loss_batch_ds`` through a throwaway ``DINOLoss`` of
        ``loss``'s hyperparameters and a zero centre). ``model.evaluate`` is never called (it would advance the loss centre inside ``call``);
        the run's centre, weights, optimizer step and random state are untouched.
    :raises ValueError: whatever the probe raises.

``knn_of(model, ds, *, data, loss)``
    The ``(model, dataset) -> {metric: float}`` seam of ``classification_run.check_best_checkpoint``, bound to its data and loss by the caller.
    :param ds: a finite batched ``(image, label)`` stream of stated length (the test split). :return: every probe key of ``model`` on ``ds``, as
        floats, through a FRESH probe (an extractor is bound to one model, and the reload check scores two). :raises ValueError: for a stream of
        unknown or endless length.
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.losses.dino_loss import DINOLoss
from dl_techniques.utils.logger import logger
from train.common.callbacks import partial_path, promote_partial
from train.common.dino_data import DinoBatches
from train.dino.knn_eval import DEFAULT_KS, DEFAULT_KNN_TEMPERATURE, KNNEvalCallback, knn_top1_accuracy


class CompiledTeacherEMA(keras.callbacks.Callback):
    """Move the teacher toward the student after every training batch, in one ``tf.function``; see the module contract."""

    def __init__(self, schedule: Callable[[int], float], warmup_steps: int = 0, log_every: int = 0) -> None:
        super().__init__()
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        self.schedule = schedule
        self.warmup_steps = int(warmup_steps)
        self.log_every = int(log_every)
        self.updates = 0
        self.traces = 0
        self._update_fn: Optional[Any] = None

    def _bind(self) -> None:
        """Capture the teacher and student weights and build the function over them (the variables are read by reference, so every later
        ``assign`` of ``load_weights`` or a backup restore is seen; a model that REPLACES its variables is not)."""
        teacher, student = getattr(self.model, "teacher", None), getattr(self.model, "student", None)
        if teacher is None or student is None:
            raise ValueError(
                f"CompiledTeacherEMA needs a model with a teacher and a student (a DINOTrainingModel), got {type(self.model).__name__}")
        if len(teacher.weights) != len(student.weights):
            raise ValueError(
                f"the teacher has {len(teacher.weights)} weights and the student {len(student.weights)}; they are paired by position")
        teacher_weights, student_weights = list(teacher.weights), list(student.weights)
        self.traces = 0

        # DECISION plan-2026-11-26T224507-e3381378/D-024
        # TWO float32 tensor arguments, `decay` and `1 - decay`, both converted on the host from the double the legacy method forms. Do NOT
        # pass `decay` as a Python float (the graph then holds it as a constant and traces again at every new value: 16,000 traces), and do NOT
        # form `1 - decay` inside the graph from one float32 tensor: that is another float32 number (0.0040000081 against 0.0040000002 at decay
        # 0.996), so the teacher drifts 9 to 12 float32 steps from the legacy one in 50 updates and passes any 1e-6 band. The arithmetic is the
        # legacy method's term for term; the parity tests hold it at ZERO difference.
        @tf.function
        def fused(decay: Any, one_minus_decay: Any) -> None:
            self.traces += 1  # runs at trace time only: a Python statement of the traced body
            for teacher_weight, student_weight in zip(teacher_weights, student_weights):
                teacher_weight.assign(keras.ops.add(
                    keras.ops.multiply(decay, teacher_weight),
                    keras.ops.multiply(one_minus_decay, keras.ops.cast(student_weight, teacher_weight.dtype))))

        self._update_fn = fused

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        # After `BackupAndRestore` has restored the weights (it assigns into the same variables), so the binding sees the resumed ones.
        self._bind()

    def update(self, decay: float) -> None:
        """Apply one update with ``decay``; the ``decay`` of the legacy ``update_teacher_ema``."""
        decay = float(decay)
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        if self._update_fn is None:
            self._bind()
        self._update_fn(tf.constant(decay, tf.float32), tf.constant(1.0 - decay, tf.float32))
        self.updates += 1

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        # DECISION plan-2026-11-26T224507-e3381378/D-024
        # The step is `optimizer.iterations`, read HERE, where Keras has already counted this batch (measured: 1, 2, 3, ... at batch end; the
        # same as the legacy 1-based `_step`, no offset). Do NOT keep a counter of the callback's own: it restarts at 0 in every process, so
        # a resumed run would replay the warmup freeze and restart the schedule at its first decay while the weights are far along it.
        step = int(self.model.optimizer.iterations)
        if step <= self.warmup_steps:
            return
        post = step - self.warmup_steps - 1  # 0-based after the warmup, as the legacy callback indexes the schedule
        decay = float(self.schedule(post))
        self.update(decay)
        if self.log_every > 0 and self.updates % self.log_every == 0:
            logger.info(f"CompiledTeacherEMA: step={step} post={post} decay={decay:.6f}")


class CentreFile(keras.callbacks.Callback):
    """Keep the loss centre in a ``.npy`` file across a kill; see the module contract."""

    def __init__(self, loss: DINOLoss, path: Union[str, Path], *, completed_epochs: Optional[int]) -> None:
        super().__init__()
        if Path(path).suffix != ".npy":
            raise ValueError(f"the centre file must be a .npy path (its temporary file is named from the extension), got {str(path)!r}")
        if completed_epochs is not None and (
                isinstance(completed_epochs, bool) or not isinstance(completed_epochs, int) or completed_epochs < 0):
            raise ValueError(f"completed_epochs must be None (a fresh run) or a count of finished epochs, got {completed_epochs!r}")
        self.loss = loss
        self.path = str(path)
        self.completed_epochs = completed_epochs

    def write(self) -> None:
        """Write ``loss.center`` to the file: whole, or not at all."""
        # DECISION plan-2026-11-26T224507-e3381378/D-104
        # The file carries NO epoch stamp. The gap between the centre and the weights is at most one epoch (the centre is written first, D-054)
        # because a failing write of either raises and ends `fit`: no later epoch writes a newer centre over weights that were never saved
        # (`test_a_failing_weights_write_after_the_centre_write_stops_the_run...`). Do NOT add a stamp that is refused on a mismatch: it would have to
        # accept the documented one-epoch-ahead window, so it could only catch a gap the ordering already excludes, and it is a second thing to keep
        # in step.
        centre = np.asarray(keras.ops.convert_to_numpy(self.loss.center), dtype=np.float32)
        partial = partial_path(self.path)
        # DECISION plan-2026-11-26T224507-e3381378/D-027
        # The file is written to a temporary name of the SAME directory and moved over the old one (`promote_partial`: fsync, then
        # `os.replace`), and the directory is fsynced after it. Do NOT write the final name in place: a process killed (or a disk filled)
        # mid-write would leave a torn file where the last epoch's centre was, and a resume would restore garbage or refuse a run it could
        # have continued. Measured with two real SIGKILLs (plan scratch p5): the old file stays byte-identical and the next write succeeds.
        with open(partial, "wb") as handle:
            np.save(handle, centre, allow_pickle=False)
            handle.flush()
        promote_partial(self.path)
        descriptor = os.open(os.path.dirname(os.path.abspath(self.path)), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def restore(self) -> bool:
        """Assign the file's centre to the loss when this is a resume; ``True`` when a file was read."""
        # DECISION plan-2026-11-26T224507-e3381378/D-039
        # `None` is a FRESH run and reads nothing: a reused directory may hold the centre file of an earlier run. Do NOT test for the file's
        # existence instead of the epoch count: a stale file would then seed a run that did not resume. A resume that cannot restore its centre
        # is refused after two or more epochs (a zero centre trains on with a finite loss and a different teacher distribution), and starts
        # from zero with one warning after exactly one (the kill fell between the weights backup and the first write). The rule holds under
        # either order of the two epoch-end writes, so the task chooses the order and the documents say a kill between them leaves the centre
        # one epoch off.
        if self.completed_epochs is None or self.completed_epochs == 0:
            return False
        if not os.path.exists(self.path):
            if self.completed_epochs >= 2:
                raise FileNotFoundError(
                    f"{self.path} is missing, but this run has finished {self.completed_epochs} epochs: its loss centre cannot be "
                    "restored, and a zero centre would silently change the teacher distribution of the resumed run")
            logger.warning(
                f"{self.path} is missing after one finished epoch (the kill came before its first write): the run resumes "
                "from a zero centre")
            return False
        centre = np.load(self.path, allow_pickle=False)
        expected = tuple(int(size) for size in self.loss.center.shape)
        if centre.shape != expected:
            raise ValueError(
                f"{self.path} holds a centre of shape {centre.shape}, but the loss has out_dim={expected[-1]} and wants {expected}")
        if not np.isfinite(centre).all():
            raise ValueError(f"{self.path} holds a centre with a value that is not finite")
        self.loss.center.assign(centre.astype(np.float32))
        return True

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.restore()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self.write()


def build_probe(data: DinoBatches, loss: DINOLoss) -> KNNEvalCallback:
    """A new k-NN probe over the data's cached streams; see the module contract."""
    return KNNEvalCallback(
        data.probe_bank_ds, data.probe_query_ds, bank_batches=data.bank_batches, query_batches=data.query_batches,
        ks=DEFAULT_KS, temperature=DEFAULT_KNN_TEMPERATURE, every_n_epochs=1, dino_loss=loss,
        control_json_path=None, random_init_repeats=0, compiled=True)


@dataclass(frozen=True)
class EpochZero:
    """What the untrained model scores: the probe's keys, the constant-feature baseline per ``k`` and the DINO loss of one batch."""

    probe: Dict[str, float]
    baseline: Dict[str, float]
    loss: float

    def as_dict(self) -> Dict[str, Any]:
        """Plain values, for ``results_summary.json``."""
        return asdict(self)


def _labels_of(stream: "tf.data.Dataset") -> np.ndarray:
    """The labels of one pass of a finite ``(image, label)`` stream, as one 1-D array."""
    return np.concatenate([np.asarray(batch[1]).reshape(-1) for batch in stream])


def constant_feature_baseline(data: DinoBatches) -> Dict[str, float]:
    """The k-NN accuracy of constant features against the real labels of the data's bank and queries, per ``k``; see ``epoch_zero``."""
    bank_labels, query_labels = _labels_of(data.probe_bank_ds), _labels_of(data.probe_query_ds)
    return {
        f"dino_knn_top1_k{k}": float(knn_top1_accuracy(
            np.ones((len(bank_labels), 1)), bank_labels, np.ones((len(query_labels), 1)), query_labels, k=k,
            temperature=DEFAULT_KNN_TEMPERATURE, num_classes=None))
        for k in DEFAULT_KS}


def epoch_zero(model: Any, data: DinoBatches, loss: DINOLoss) -> EpochZero:
    """The :class:`EpochZero` of ``model``; see the module contract."""
    # DECISION plan-2026-11-26T224507-e3381378/D-022
    # Do NOT take any of these numbers from `model.evaluate`, `test_on_batch` or `predict`: `DINOLoss` advances its centre inside `call()`,
    # so one `evaluate` of the untrained model moves the run's centre off its zero start with a finite value and a clean exit (measured at
    # 81 percent for a four-sample validation set). The probe runs the frozen networks directly, and the loss is one eager forward through a
    # THROWAWAY loss of the same hyperparameters, never the run's. `training=False`, and no op that draws from the global seed generator:
    # a diagnostic pass must not perturb the run it reports on (a training-mode guard once moved a same-seed accuracy by 0.004).
    # DECISION plan-2026-11-26T224507-e3381378/D-040
    # The control is scored by a probe of its OWN, built by `build_probe` like every other probe of the run: one configuration, so the number
    # equals the per-epoch probe's on the same weights to the last digit (an eager extractor here moves three float columns in the 8th
    # digit). Do NOT hand it the per-epoch probe object: an extractor is bound to the first model it sees.
    probe = build_probe(data, loss)
    probe.set_model(model)
    metrics = {key: float(value) for key, value in probe.evaluate_split(data.probe_query_ds, data.query_batches).items()}

    baseline = constant_feature_baseline(data)

    throwaway = DINOLoss(
        out_dim=loss.out_dim, student_temp=loss.student_temp, teacher_temp=loss.teacher_temp, center_momentum=loss.center_momentum)
    views, labels = next(iter(data.loss_batch_ds))
    first_loss = float(keras.ops.convert_to_numpy(throwaway(labels, model(views, training=False))))
    return EpochZero(probe=metrics, baseline=baseline, loss=first_loss)


def knn_of(model: Any, ds: "tf.data.Dataset", *, data: DinoBatches, loss: DINOLoss) -> Dict[str, float]:
    """The probe keys of ``model`` on the finite stream ``ds``; see the module contract."""
    # DECISION plan-2026-11-26T224507-e3381378/D-040
    # A FRESH probe per call: `KNNEvalCallback` binds its feature model and its compiled functions to the first model it is given, and the
    # best-checkpoint reload check scores two (the in-memory best and the reloaded file). Do NOT cache the probe across calls.
    batches = int(ds.cardinality())
    if batches < 0:
        raise ValueError("knn_of scores one finite pass: the stream has no stated length (unknown or endless)")
    probe = build_probe(data, loss)
    probe.set_model(model)
    return {key: float(value) for key, value in probe.evaluate_split(ds, batches).items()}
