"""``train_nanovlm`` is nanoVLM's only training entry point, and it could not
train at all -- while reporting a clean run.

Three independent defects, in the order they masked each other (F-82):

(a) ``keras.GradientTape``. There is no such attribute in Keras 3, so every
    call to ``NanoVLMTrainer.train_step`` raised ``AttributeError: module
    'keras' has no attribute 'GradientTape'`` before a single gradient was
    computed. The repo idiom is ``tf.GradientTape`` (see
    ``models/masked_autoencoder/mae.py``).
(b) ``Metric.reset_states``. Keras 3 renamed it to ``reset_state``, so
    ``trainer.reset_metrics()`` raised at the top of every epoch.
(c) ``for step, batch in enumerate(train_dataset)``. ``keras.utils.Sequence``
    IS ``PyDataset`` in Keras 3 and defines no ``__iter__``, so this used
    Python's legacy ``__getitem__`` protocol, which walks 0, 1, 2, ... until
    ``IndexError`` and ignores ``__len__`` entirely. ``VQADataSequence``
    serves any index and never raises ``IndexError``, so epoch 0 never ended.

The reason all three survived is the per-step ``except Exception: continue``
inside the epoch loop: an ``AttributeError`` -- a programming defect, not a bad
sample -- was logged at ``error`` level and stepped over, so the run "completed"
all ten epochs and saved an untrained model.

This file drives the entry point, not just the trainer class: constructing
``NanoVLMTrainer`` and calling nothing passes with all three defects present.
``create_nanovlm`` / ``load_cauldron_sample`` / ``create_vqa_dataset`` are
patched in the trainer module's OWN namespace (they are imported there at
module scope), which is the standard rule in ``tests/test_train/``.

Out of scope on purpose: ``load_cauldron_sample`` returns placeholder paths
(``'path/to/image1.jpg'``) that no ``preprocess_image`` can resolve. That is a
separate, lower-severity data-plumbing gap; folding it in here would make the
loop/API fixes unfalsifiable.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest

import train.nano_vlm.train_nano_vlm as train_nano_vlm_module
from train.nano_vlm.train_nano_vlm import NanoVLMTrainer, train_nanovlm

BATCH = 2
SEQ = 8
FEATURES = 4
VOCAB = 16


@pytest.fixture(autouse=True)
def restore_global_dtype_policy():
    """``train_nanovlm`` calls ``configure_mixed_precision()``, which sets the
    PROCESS-global Keras dtype policy. Leaking ``mixed_float16`` into the rest
    of the session would change the numeric regime of every later test."""
    previous = keras.mixed_precision.global_policy()
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(previous)


def _tiny_model() -> keras.Model:
    """A stand-in for the real nanoVLM: same call contract
    (``inputs -> [batch, seq, vocab]`` logits), small enough for CPU.

    Not "the 222M-parameter nanoVLM", which is what this said until 2026-08-22:
    `create_nanovlm()` at the default `variant="base"` measures **305,435,904**
    parameters. 222M is a design target from `research/nanoVLM_research.md` that
    the shipped model never hit; see D-008."""
    model = keras.Sequential(
        [keras.layers.Input(shape=(SEQ, FEATURES)), keras.layers.Dense(VOCAB)],
        name="tiny_nanovlm",
    )
    model.build((None, SEQ, FEATURES))
    return model


def _batch(seed: int = 0):
    rng = np.random.default_rng(seed)
    inputs = rng.random((BATCH, SEQ, FEATURES)).astype("float32")
    # `ignore_index=0`, so 0-valued labels contribute nothing to the loss.
    labels = rng.integers(1, VOCAB, (BATCH, SEQ)).astype("int32")
    return inputs, labels


class _CountingDataset(keras.utils.Sequence):
    """Records every index it is asked for, and hard-stops past a small limit.

    ``__getitem__`` deliberately serves ANY index instead of raising
    ``IndexError`` past the end -- that is what ``VQADataSequence`` does, and it
    is exactly the shape that made the old ``enumerate`` loop non-terminating.
    The cap turns an unbounded hang into a bounded, named failure.
    """

    _LIMIT = 6

    def __init__(self, length: int = 3) -> None:
        super().__init__()
        self._inputs, self._labels = _batch(seed=1)
        self._length = length
        self.requested = []

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx):
        self.requested.append(idx)
        if len(self.requested) > self._LIMIT:
            raise RuntimeError(
                "the training loop asked for more batches than __len__ "
                f"declares ({self.requested}) -- it is iterating past the end "
                "of the dataset and would not terminate"
            )
        return self._inputs, self._labels


def _patch_entry_point(monkeypatch, dataset):
    monkeypatch.setattr(train_nano_vlm_module, "create_nanovlm",
                        lambda *a, **k: _tiny_model())
    monkeypatch.setattr(train_nano_vlm_module, "load_cauldron_sample",
                        lambda *a, **k: [])
    monkeypatch.setattr(train_nano_vlm_module, "create_vqa_dataset",
                        lambda *a, **k: dataset)


class TestNanoVLMTrainerUsesTheKeras3API:
    """Defects (a) and (b), probed directly on the trainer class."""

    def test_reset_metrics_uses_reset_state(self):
        trainer = NanoVLMTrainer(
            _tiny_model(),
            train_nano_vlm_module.NanoVLMLoss(ignore_index=0),
            use_multi_optimizer=False,
        )
        # Pre-fix: AttributeError: 'Mean' object has no attribute 'reset_states'
        assert trainer.reset_metrics() is None

    def test_train_step_computes_a_finite_loss_and_moves_the_weights(self):
        model = _tiny_model()
        trainer = NanoVLMTrainer(
            model,
            train_nano_vlm_module.NanoVLMLoss(ignore_index=0),
            use_multi_optimizer=False,
        )
        before = keras.ops.convert_to_numpy(model.trainable_variables[0]).copy()

        # Pre-fix: AttributeError: module 'keras' has no attribute 'GradientTape'
        metrics = trainer.train_step(_batch(seed=2))

        assert set(metrics) == {"loss", "accuracy"}
        loss = float(metrics["loss"])
        assert np.isfinite(loss), f"loss must be finite, got {loss}"
        after = keras.ops.convert_to_numpy(model.trainable_variables[0])
        # Exact inequality, not `allclose`: the shipped schedule warms up from
        # `warmup_start_lr=1e-8`, so the first step's update is ~1e-8 and sits
        # INSIDE `allclose`'s default tolerance. A tolerance-based assertion
        # here passes against an optimizer that applied nothing at all.
        assert not np.array_equal(before, after), (
            "the optimizer applied no update -- gradients never reached the "
            "weights"
        )


class TestTrainNanoVLMEntryPoint:
    """Defect (c) plus the swallowing ``except``, probed end to end."""

    def test_each_batch_is_consumed_exactly_once_per_epoch(
            self, monkeypatch, tmp_path
    ):
        dataset = _CountingDataset(length=3)
        _patch_entry_point(monkeypatch, dataset)
        # `results_dir` is built from the RELATIVE literal "results", so the
        # run directory must not land in the repo-root results/ tree (which is
        # gitignored, untracked and therefore unrecoverable).
        monkeypatch.chdir(tmp_path)

        assert train_nanovlm(
            epochs=1, batch_size=BATCH, use_multi_optimizer=False,
            checkpoint_frequency=0, log_frequency=1,
        ) is None

        assert dataset.requested == list(range(len(dataset))), (
            "the loop must request every index exactly once, in order, with "
            f"no repeats and no gaps; got {dataset.requested}"
        )
        saved = sorted(tmp_path.glob("results/nanovlm_*/final_model.keras"))
        assert len(saved) == 1, f"expected one saved model, got {saved}"
        assert saved[0].stat().st_size > 0

    def test_two_epochs_restart_at_index_zero(self, monkeypatch, tmp_path):
        dataset = _CountingDataset(length=2)
        _patch_entry_point(monkeypatch, dataset)
        monkeypatch.chdir(tmp_path)

        train_nanovlm(
            epochs=2, batch_size=BATCH, use_multi_optimizer=False,
            checkpoint_frequency=0, log_frequency=1,
        )

        assert dataset.requested == [0, 1, 0, 1]

    def test_a_programming_error_in_a_step_is_not_swallowed(
            self, monkeypatch, tmp_path
    ):
        """The per-step handler exists to survive a bad SAMPLE, not a broken
        training mechanism. An ``AttributeError`` is a defect in the code and
        must propagate -- swallowing it is what let (a) and (b) run for ten
        full epochs and save an untrained model."""
        dataset = _CountingDataset(length=2)
        _patch_entry_point(monkeypatch, dataset)
        monkeypatch.chdir(tmp_path)

        def _boom(self, batch_data):
            raise AttributeError("module 'keras' has no attribute 'Whatever'")

        monkeypatch.setattr(NanoVLMTrainer, "train_step", _boom)

        with pytest.raises(AttributeError):
            train_nanovlm(
                epochs=1, batch_size=BATCH, use_multi_optimizer=False,
                checkpoint_frequency=0, log_frequency=1,
            )

    def test_a_bad_sample_is_still_survivable(self, monkeypatch, tmp_path):
        """The other half of the narrowing: a shape mismatch from one corrupt
        batch is a data problem, is logged, and the epoch continues."""
        dataset = _CountingDataset(length=2)
        _patch_entry_point(monkeypatch, dataset)
        monkeypatch.chdir(tmp_path)

        calls = {"n": 0}
        original = NanoVLMTrainer.train_step

        def _one_bad_batch(self, batch_data):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("incompatible shapes in this sample")
            return original(self, batch_data)

        monkeypatch.setattr(NanoVLMTrainer, "train_step", _one_bad_batch)

        train_nanovlm(
            epochs=1, batch_size=BATCH, use_multi_optimizer=False,
            checkpoint_frequency=0, log_frequency=1,
        )
        assert dataset.requested == [0, 1]


class TestKerasSequenceHasNoIter:
    """The mechanism behind defect (c), pinned so a future reader does not
    'simplify' the indexed loop back into a ``for ... in dataset``."""

    def test_pydataset_does_not_define_iter(self):
        assert not hasattr(keras.utils.Sequence, "__iter__"), (
            "keras.utils.Sequence gained __iter__; the indexed loop in "
            "train_nanovlm is still correct, but the reason recorded in its "
            "DECISION anchor has changed"
        )

    def test_legacy_getitem_iteration_ignores_len(self):
        dataset = _CountingDataset(length=1)
        seen = 0
        for _ in dataset:
            seen += 1
            if seen > 2:
                break
        assert seen == 3, (
            "a plain `for ... in` over a PyDataset walks past __len__ -- this "
            "is why train_nanovlm indexes by range(len(...))"
        )


# ---------------------------------------------------------------------
# `steps_per_epoch == 0` and the unbuilt model (D-134)
# ---------------------------------------------------------------------


class _UnbuiltStandIn(keras.Model):
    """A stand-in that is UNBUILT on construction, like ``create_nanovlm()``.

    ``NanoVLM.build`` takes a dict of ``{'images': ..., 'text_tokens': ...}``
    shapes and materialises its sub-layers; nothing else builds it, so
    ``count_params()`` raises until someone calls it. This records the shape
    dict it was handed so the test can assert where those numbers came from.
    """

    def __init__(self) -> None:
        super().__init__(name="unbuilt_standin")
        self.dense = keras.layers.Dense(VOCAB)
        self.build_shapes = []

    def build(self, input_shape) -> None:
        self.build_shapes.append(input_shape)
        self.dense.build((None, SEQ, FEATURES))
        super().build((None, SEQ, FEATURES))

    def call(self, inputs, training=None):
        return self.dense(inputs)


class TestTrainNanoVLMRefusesZeroSteps:
    """``steps_per_epoch == 0`` at the shipped defaults, MEASURED.

    ``VQADataSequence.__len__`` is ``len(samples) // batch_size`` and
    ``load_cauldron_sample()`` returns exactly 3 placeholder rows, so the
    shipped default ``batch_size=8`` gives ``3 // 8 == 0``. Pre-fix, that made
    ``for step in range(0)`` run ZERO steps: ten epochs each logged
    ``Loss=0.0000`` from an untouched ``Mean`` metric and the run saved an
    UNTRAINED ``final_model.keras`` with no error anywhere. Not a crash -- the
    quiet outcome is the whole defect.

    The real ``load_cauldron_sample`` / ``create_vqa_dataset`` are used here on
    purpose: the arithmetic under test is theirs, and patching them would pin
    a number this test made up.
    """

    def test_the_shipped_default_batch_size_is_refused_by_name(
            self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(train_nano_vlm_module, "create_nanovlm",
                            lambda *a, **k: _tiny_model())
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            train_nanovlm(epochs=10, batch_size=8, use_multi_optimizer=False,
                          checkpoint_frequency=0, log_frequency=10)

        message = str(excinfo.value)
        assert "steps_per_epoch == 0" in message, message
        # The arithmetic must be IN the message: a bare "invalid batch size"
        # would not tell the caller that the sample loader has only 3 rows.
        assert "len(samples)=3" in message and "batch_size=8" in message, message
        # Pre-fix this directory held a saved, untrained model.
        assert list(tmp_path.glob("results/nanovlm_*/*.keras")) == []

    def test_a_batch_size_the_sample_loader_can_fill_is_not_refused(
            self, monkeypatch, tmp_path
    ):
        """The control. ``3 // 3 == 1``, so the guard must NOT fire -- it must
        let the run through to the (separate, loud) placeholder-image failure
        at ``vqa_dataset.py``'s ``np.stack``, which is a data problem and not
        this guard's business."""
        monkeypatch.setattr(train_nano_vlm_module, "create_nanovlm",
                            lambda *a, **k: _tiny_model())
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            train_nanovlm(epochs=1, batch_size=3, use_multi_optimizer=False,
                          checkpoint_frequency=0, log_frequency=1)

        assert "steps_per_epoch" not in str(excinfo.value), str(excinfo.value)
        assert "need at least one array to stack" in str(excinfo.value)


class TestTrainNanoVLMBuildsTheModelBeforeUsingIt:
    """``create_nanovlm()`` returns an UNBUILT subclassed model.

    MEASURED at HEAD, before this fix::

        ValueError: You tried to call `count_params` on layer 'nano_vlm',
        but the layer isn't built.

    raised from ``train_nano_vlm.py``'s own second statement, so the entry
    point died before it ever touched the data -- the identical defect the
    D-031 anchor fixed in ``src/train/hrm/train_hrm.py``. A second, quieter
    consequence: ``setup_different_learning_rates`` reads
    ``layer.trainable_variables``, which on an unbuilt model is empty, so the
    multi-optimizer partition would have been three EMPTY lists.
    """

    def test_the_model_is_built_with_the_processors_own_shapes(
            self, monkeypatch, tmp_path
    ):
        model = _UnbuiltStandIn()
        dataset = _CountingDataset(length=2)
        monkeypatch.setattr(train_nano_vlm_module, "create_nanovlm",
                            lambda *a, **k: model)
        monkeypatch.setattr(train_nano_vlm_module, "load_cauldron_sample",
                            lambda *a, **k: [{}, {}])
        monkeypatch.setattr(train_nano_vlm_module, "create_vqa_dataset",
                            lambda *a, **k: dataset)
        monkeypatch.chdir(tmp_path)

        # Pre-fix: ValueError from `count_params` on the unbuilt model.
        train_nanovlm(epochs=1, batch_size=BATCH, use_multi_optimizer=False,
                      checkpoint_frequency=0, log_frequency=1)

        assert model.build_shapes == [
            {"images": (None, 224, 224, 3), "text_tokens": (None, 512)}
        ], model.build_shapes

    def test_the_multi_optimizer_partition_sees_real_variables(
            self, monkeypatch, tmp_path
    ):
        """Proves the build happens BEFORE ``NanoVLMTrainer`` is constructed,
        not merely somewhere in the function."""
        model = _UnbuiltStandIn()
        dataset = _CountingDataset(length=2)
        seen = {}
        original = train_nano_vlm_module.setup_different_learning_rates

        def _recording(m):
            seen["n_trainable"] = len(m.trainable_variables)
            return original(m)

        monkeypatch.setattr(train_nano_vlm_module, "create_nanovlm",
                            lambda *a, **k: model)
        monkeypatch.setattr(train_nano_vlm_module, "load_cauldron_sample",
                            lambda *a, **k: [{}, {}])
        monkeypatch.setattr(train_nano_vlm_module, "create_vqa_dataset",
                            lambda *a, **k: dataset)
        monkeypatch.setattr(train_nano_vlm_module,
                            "setup_different_learning_rates", _recording)
        monkeypatch.chdir(tmp_path)

        train_nanovlm(epochs=1, batch_size=BATCH, use_multi_optimizer=True,
                      checkpoint_frequency=0, log_frequency=1)

        assert seen["n_trainable"] == 2, seen
