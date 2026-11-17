"""The child process of ``test_depth_resume.py``: one real ``depth_estimation`` fit a parent can SIGKILL.

Run as ``python _depth_child.py '<json>'`` on the CPU, in the marker protocol of ``_resume_protocol.py`` (which holds the
trace, the ``Recorder`` and the ``Stopper`` the stream tasks share, and the parent side). The JSON holds

* ``mapping``: the request, as ``validate(DepthEstimationConfig, mapping)`` takes it;
* ``stop_at``: ``[epoch, done]`` prints ``@@AT <epoch> <done>`` and sleeps when ``done`` batches of ``epoch`` have finished;
* ``trace``: a file the child appends JSON lines to, flushed at once, so a SIGKILL leaves exactly what ran: the protocol's
  ``epoch_begin`` (with the scores of the restored weights when ``probe_restored``), ``batch``, ``epoch_end`` and
  ``train_end``, and three of this task:

  - ``select``: one line per data object the fit could use (the one ``build_data`` made, and the one ``resume_stream``
    replaced it with): its stream seed, its fingerprint and the sha256 of its selection crops, test crops and layout map,
    and the selection and test files (relative to the root);
  - ``train_batch``: one line each time the train stream hands out a batch: the stream seed, the global index, the sha256
    of the batch (x bytes then y bytes, as the parent hashes a stream) and, per sample, every file its draw tried (a
    rejected crop included), relative to the root. Keras reads a batch in the caller for the shape before the fit reads it
    again, and the queue reads ahead of the fit, so an index can appear twice and beyond the last trained batch;
  - ``optimizer``: at the first epoch begin of the process and at every epoch end (after the backup callback has written
    the copy): the optimizer's class, its learning-rate table by unscoped variable path, the unscoped paths of the DPT
    head's trainable variables, the number of optimizer variables, how many hold a nonzero value (the step count left
    out) and the sha256 of all of them in order;
* ``label``: a name copied into every trace line.

The watched stream is a :class:`~train.common.depth_data.DepthStream` subclass that draws each sample with the stream's own
``sample`` (so the batch is the stream's; the parent holds the digests equal to the streams it builds itself). Prints
``@@DONE`` at the end. Determinism is the golden run's (op determinism, one thread per pool).
"""

import dataclasses
import hashlib
import json
import os
import sys

import _resume_protocol as protocol

protocol.child_environment()


def sha256_of(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(array.tobytes())
    return digest.hexdigest()


def unscoped(path: str) -> str:
    """A variable path without the model's own name (``depth_anything/block_0/...`` -> ``block_0/...``)."""
    return path.partition("/")[2]


def main(spec: dict) -> None:
    import numpy as np

    protocol.start_tensorflow()

    import keras

    import train.api.recipes as recipes
    from train.api.tasks import register_builtin_tasks
    from train.api.tasks.depth_estimation import DepthEstimationConfig, DepthEstimationTrainer
    from train.common.depth_data import DepthStream

    recipes.register_builtin_models()
    register_builtin_tasks()
    root = spec["mapping"]["data"]["root"]
    #: The protocol's ``write``, handed to ``watcher`` before the trainer is built.
    trace = {}

    def relative(paths):
        return [os.path.relpath(path, root) for path in paths]

    class WatchedStream(DepthStream):
        """The stream it wraps, batch for batch, with each batch written to the trace."""

        def __getitem__(self, index):
            start = int(index) * self.batch_size
            xs, ys, files = [], [], []
            for counter in range(start, start + self.batch_size):
                x, y, tried = self.sample(counter)
                xs.append(x)
                ys.append(y)
                files.append(relative([self.rgb_paths[k] for k in tried]))
            x, y = np.stack(xs), np.stack(ys)
            trace["write"]("train_batch", stream_seed=self.stream_seed, index=int(index), digest=sha256_of(x, y),
                           files=files)
            return x, y

    def watcher(write):
        trace["write"] = write

        def watched(data, config):
            """``data`` with its fixed sets written and its train stream replaced by a watched copy of it."""
            write("select", stream_seed=data.stream_seed, fingerprint=data.fingerprint,
                  select=sha256_of(data.x_select, data.y_select), test=sha256_of(data.x_test, data.y_test),
                  layout=sha256_of(data.layout), select_files=relative(data.select_files),
                  test_files=relative(data.test_files))
            inner = data.train_ds
            stream = WatchedStream(inner.rgb_paths, inner.depth_paths, stream_seed=inner.stream_seed,
                                   batch_size=inner.batch_size, image_size=inner.image_size,
                                   min_valid_ratio=inner.min_valid_ratio, augment=inner.augment, workers=data.workers)
            return dataclasses.replace(data, train_ds=stream)

        return watched

    class OptimizerProbe(keras.callbacks.Callback):
        """Write what the optimizer holds; sits after the task's callbacks, so after the copy is written or restored."""

        def __init__(self):
            super().__init__()
            self.first = True

        def describe(self, at, epoch):
            optimizer = self.model.optimizer
            values = [keras.ops.convert_to_numpy(v) for v in optimizer.variables]
            steps = id(optimizer.iterations)
            nonzero = sum(bool(np.any(value)) for v, value in zip(optimizer.variables, values) if id(v) != steps)
            trace["write"](
                "optimizer", at=at, epoch=epoch, cls=type(optimizer).__name__,
                table={unscoped(k): float(m) for k, m in dict(getattr(optimizer, "lr_multipliers", {}) or {}).items()},
                table_paths_are_the_heads=sorted(getattr(optimizer, "lr_multipliers", {}) or {}) == sorted(
                    v.path for v in self.model.decoder.trainable_variables),
                head=sorted(unscoped(v.path) for v in self.model.decoder.trainable_variables),
                variables=len(values), nonzero=nonzero, iterations=int(keras.ops.convert_to_numpy(optimizer.iterations)),
                digest=sha256_of(*values))

        def on_epoch_begin(self, epoch, logs=None):
            if self.first:
                self.describe("begin", epoch)
            self.first = False

        def on_epoch_end(self, epoch, logs=None):
            self.describe("end", epoch)

    class Probed(DepthEstimationTrainer):
        def _callbacks(self, config, data, *arguments):
            epoch_line, callbacks = super()._callbacks(config, data, *arguments)
            callbacks.append(OptimizerProbe())
            return epoch_line, callbacks

    protocol.run_child(spec, Probed, DepthEstimationConfig, watcher)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
