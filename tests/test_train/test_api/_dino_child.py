"""The child process of ``test_dino_resume.py``: one real ``dino`` fit a parent can SIGKILL.

Run as ``python _dino_child.py '<json>'`` on the CPU, in the marker protocol of ``_resume_protocol.py`` (the trace, the ``Recorder`` and the
``Stopper`` the stream tasks' children share, and the parent side). The JSON holds ``mapping`` (the request), ``stop_at`` (``[epoch, done]``: print
``@@AT`` and sleep when ``done`` batches of ``epoch`` have finished), ``trace`` (a file the child appends JSON lines to, flushed at once, so a
SIGKILL leaves exactly what ran) and ``label``.

The lines this child adds to the trace beyond the protocol's: ``select``, ``bank`` and ``query`` (the row identity of every image of the selection
split and of the two probe streams, written when the data is built and again for the stream a resume replaces it with), ``train_batch`` (the sha256
of the view batch, written by a ``tf.numpy_function`` added to ``train_ds``), ``restored`` (the digest of every weight, of the loss's centre and the
optimizer step, written at the start of ``fit``, after every restore), ``epoch_state`` (the same digests at each epoch end) and ``ema`` (the
teacher callback's update count at the end). The data is the identity fake CIFAR-10 of ``tests/classification_fakes`` and the backbone is 2
blocks deep; determinism is the golden run's. Prints ``@@DONE`` at the end.
"""

import dataclasses
import hashlib
import json
import sys

import _resume_protocol as protocol

protocol.child_environment()


def main(spec: dict) -> None:
    tf = protocol.start_tensorflow()

    import keras
    import numpy as np
    import pytest

    import dl_techniques.models.vision.dino.training as training
    import train.api.recipes as recipes
    from tests.classification_fakes import identities_of, use_identity_cifar10
    from train.api.tasks import register_builtin_tasks
    from train.api.tasks.dino import DinoConfig, DinoTrainer
    from train.common.dino_run import CompiledTeacherEMA

    patch = pytest.MonkeyPatch()
    use_identity_cifar10(patch)
    original = training.create_dino_training_model
    patch.setattr(training, "create_dino_training_model", lambda *args, **kwargs: original(*args, depth=2, **kwargs))
    recipes.register_builtin_models()
    register_builtin_tasks()

    def write(kind, **fields):
        with open(spec["trace"], "a") as handle:
            handle.write(json.dumps({"label": spec["label"], "kind": kind, **fields}) + "\n")
            handle.flush()

    def digest_of(arrays):
        return hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)).hexdigest()

    class Extra(keras.callbacks.Callback):
        def __init__(self, trainer, ema):
            super().__init__()
            self.trainer, self.ema = trainer, ema

        def state(self):
            return {
                "weights": digest_of([w.numpy() for w in self.model.weights]),
                "centre": digest_of([keras.ops.convert_to_numpy(self.trainer._loss.center)]),
                "iterations": protocol.steps_of(self.model),
            }

        def on_train_begin(self, logs=None):
            write("restored", **self.state())

        def on_epoch_end(self, epoch, logs=None):
            write("epoch_state", epoch=epoch, **self.state())

        def on_train_end(self, logs=None):
            write("ema", updates=self.ema.updates, traces=self.ema.traces)

    original_callbacks = DinoTrainer._callbacks

    def callbacks(self, *arguments, **keywords):
        epoch_line, built = original_callbacks(self, *arguments, **keywords)
        built.append(Extra(self, next(c for c in built if isinstance(c, CompiledTeacherEMA))))
        return epoch_line, built

    DinoTrainer._callbacks = callbacks

    def watcher(write_line):
        def record(views):
            write_line("train_batch", digest=hashlib.sha256(views.tobytes()).hexdigest())
            return views

        def watched(data, config):
            for kind, ds in (("select", data.select_ds), ("bank", data.probe_bank_ds), ("query", data.probe_query_ds)):
                write_line(kind, rows=sorted(identities_of("cifar10", ds, 0).tolist()))

            def see(views, label):
                seen = tf.numpy_function(record, [views], views.dtype)
                seen.set_shape(views.shape)
                return seen, label

            return dataclasses.replace(data, train_ds=data.train_ds.map(see))

        return watched

    protocol.run_child(spec, DinoTrainer, DinoConfig, watcher)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
