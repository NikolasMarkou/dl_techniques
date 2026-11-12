"""The child process of ``test_pretrain_resume.py``: one real ``pretrain`` fit a parent can SIGKILL.

Run as ``python _pretrain_child.py '<json>'`` on the CPU, in the marker protocol of ``_denoising_child.py`` (the parent is
``test_backup_every_steps.run_child``: it reads the ``@@`` markers off the log and SIGKILLs the RECORDED PID). The JSON holds

* ``mapping``: the request, as ``validate(PretrainConfig, mapping)`` takes it;
* ``stop_at``: ``[epoch, done]`` prints ``@@AT <epoch> <done>`` and sleeps when ``done`` batches of ``epoch`` have finished;
* ``trace``: a file the child appends JSON lines to, flushed at once, so a SIGKILL leaves exactly what ran: ``epoch_begin``
  (the epoch Keras passed, the optimizer's step count, and the scores of the weights the resume restored when
  ``probe_restored``), ``batch``, ``epoch_end``, ``train_end``, ``select`` (the row identity of every selection image of
  the data the fit will use) and ``train_batch`` (the digest of the batch and the row identities in it);
* ``label``: a name copied into every trace line; ``floor``: the tokenizer usage floor, when the run needs a lower one.

``train_batch`` lines come from a ``tf.numpy_function`` the child adds to ``train_ds``, so they are what the fit really read
and in the order it read it; ``select`` lines come from one pass of ``select_ds`` when the data is built (and once more for
the stream a resume replaces it with). The data is the identity fake CIFAR-10 of ``tests/classification_fakes``, which the
child installs itself; determinism is the golden run's (op determinism, one thread per pool). Prints ``@@DONE`` at the end.
"""

import dataclasses
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")


def main(spec: dict) -> None:
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    import keras
    import pytest

    import train.api.recipes as recipes
    from tests.classification_fakes import identities_of, row_identities, use_identity_cifar10
    from train.api.config import validate
    from train.api.tasks import register_builtin_tasks
    from train.api.tasks.pretrain import PretrainConfig, PretrainTrainer
    from train.common import pretrain_run

    use_identity_cifar10(pytest.MonkeyPatch())
    if spec.get("floor") is not None:
        pretrain_run.MIN_TOKENIZER_PERPLEXITY = spec["floor"]
    recipes.register_builtin_models()
    register_builtin_tasks()

    def write(kind: str, **fields) -> None:
        with open(spec["trace"], "a") as handle:
            handle.write(json.dumps({"label": spec["label"], "kind": kind, **fields}) + "\n")
            handle.flush()

    def steps_of(model) -> int:
        optimizer = model.optimizer
        return int(keras.ops.convert_to_numpy(getattr(optimizer, "inner_optimizer", optimizer).iterations))

    def record(image, mask):
        digest = hashlib.sha256(image.tobytes() + mask.tobytes()).hexdigest()
        write("train_batch", digest=digest, rows=row_identities(image, "cifar10", 0).tolist())
        return image

    def watched(data):
        """``data`` with its select identities written and its train stream read through ``record``."""
        write("select", rows=sorted(identities_of("cifar10", data.select_ds, 0).tolist()))

        def see(inputs, target, weight):
            image, mask = inputs
            seen = tf.numpy_function(record, [image, mask], image.dtype)
            seen.set_shape(image.shape)
            return (seen, mask), target, weight

        return dataclasses.replace(data, train_ds=data.train_ds.map(see))

    class Recorder(keras.callbacks.Callback):
        """Append what the run did to the trace; sits last in the list, after the callback that restores the backup."""

        def __init__(self) -> None:
            super().__init__()
            self.epoch, self.first = 0, True

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch
            fields = {"epoch": epoch, "iterations": steps_of(self.model)}
            if self.first and spec.get("probe_restored"):
                # The weights a resume restored, scored before any step of this process: the select masks are fixed by
                # (split seed, row), so these are the numbers the killed run logged for the epoch they came from.
                fields["restored"] = self.model.evaluate(self.data.select_ds, verbose=0, return_dict=True)
            self.first = False
            write("epoch_begin", **fields)

        def on_train_batch_end(self, batch, logs=None):
            write("batch", epoch=self.epoch, batch=batch, before=steps_of(self.model) - 1)

        def on_epoch_end(self, epoch, logs=None):
            write("epoch_end", epoch=epoch, iterations=steps_of(self.model))

        def on_train_end(self, logs=None):
            write("train_end", iterations=steps_of(self.model))

    class Stopper(keras.callbacks.Callback):
        """Block when ``done`` batches of ``epoch`` are finished (runs before the next batch)."""

        def __init__(self, epoch: int, done: int) -> None:
            super().__init__()
            self.target, self.epoch = (epoch, done), 0

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch

        def on_train_batch_begin(self, batch, logs=None):
            if self.target == (self.epoch, batch) and batch > 0:
                print(f"\n@@AT {self.epoch} {batch}", flush=True)
                time.sleep(3600)

    class Child(PretrainTrainer):
        def build_data(self, config, context):
            return watched(super().build_data(config, context))

        def resume_stream(self, config, data, completed_steps):
            replacement = super().resume_stream(config, data, completed_steps)
            # None is the task saying "keep the data": the fit then reads the stream `build_data` made.
            return None if replacement is None else watched(replacement)

        def _callbacks(self, config, data, *arguments):
            epoch_line, callbacks = super()._callbacks(config, data, *arguments)
            recorder = Recorder()
            recorder.data = data
            callbacks.append(recorder)
            if spec.get("stop_at"):
                callbacks.append(Stopper(*spec["stop_at"]))
            return epoch_line, callbacks

    Child(validate(PretrainConfig, spec["mapping"])).run()
    print("\n@@DONE", flush=True)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
