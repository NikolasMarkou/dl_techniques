"""The child process of ``test_forecast_resume.py``: one real ``time_series_forecasting`` fit a parent can SIGKILL.

Run as ``python _forecast_child.py '<json>'`` on the CPU, in the marker protocol of ``_pretrain_child.py`` (the parent is
``test_backup_every_steps.run_child``: it reads the ``@@`` markers off the log and SIGKILLs the RECORDED PID). The JSON holds

* ``mapping``: the request, as ``validate(TimeSeriesForecastingConfig, mapping)`` takes it;
* ``stop_at``: ``[epoch, done]`` prints ``@@AT <epoch> <done>`` and sleeps when ``done`` batches of ``epoch`` have finished;
* ``trace``: a file the child appends JSON lines to, flushed at once, so a SIGKILL leaves exactly what ran: ``epoch_begin``
  (the epoch Keras passed, the optimizer's step count, and the scores of the weights the resume restored when
  ``probe_restored``), ``batch``, ``epoch_end``, ``train_end``, ``select`` (the sha256 of the bank, the segments and the
  ``(series, start)`` of every selection window of the data the fit will use) and ``train_batch`` (the digest of the batch and
  the ``(series, start)`` of every window in it);
* ``label``: a name copied into every trace line.

A ``(series, start)`` is found by looking the standardized window up in a table of EVERY window of the bank the process
built first (before any resume), so it is what the fit really read and does not depend on the keys the data module used to
draw it: a replacement stream built on another bank finds nothing (``distance`` far from zero). ``train_batch`` lines come
from a ``tf.numpy_function`` the child adds AFTER the batching (a sequential map, so they come in the order the fit reads).
Prints ``@@DONE`` at the end. Determinism is the golden run's (op determinism, one thread per pool).
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
    import numpy as np
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    import keras

    import train.api.recipes as recipes
    from train.api.config import validate
    from train.api.tasks import register_builtin_tasks
    from train.api.tasks.time_series_forecasting import TimeSeriesForecastingConfig, TimeSeriesForecastingTrainer
    from train.common.forecast_data import standardize_np

    recipes.register_builtin_models()
    register_builtin_tasks()
    #: Filled by the first ``build_data`` of the process: ``(table, positions)`` of every window of its bank.
    reference = {}

    def write(kind: str, **fields) -> None:
        with open(spec["trace"], "a") as handle:
            handle.write(json.dumps({"label": spec["label"], "kind": kind, **fields}) + "\n")
            handle.flush()

    def steps_of(model) -> int:
        optimizer = model.optimizer
        return int(keras.ops.convert_to_numpy(getattr(optimizer, "inner_optimizer", optimizer).iterations))

    def table_of(data, floor) -> None:
        """Every window ``(series, start)`` of the bank, standardized: context then target, one row each."""
        width = data.lookback + data.horizon
        rows, positions = [], []
        for series in range(data.bank.shape[0]):
            for start in range(data.bank.shape[1] - width + 1):
                x, y = standardize_np(data.bank[series, start:start + width], data.lookback, floor)
                rows.append(np.concatenate([x.reshape(-1), y.reshape(-1)]))
                positions.append((series, start))
        reference["table"], reference["positions"] = np.stack(rows), positions

    def locate(x, y):
        """``([(series, start)], largest distance to the nearest window of the table)`` of a batch."""
        found = np.concatenate([x.reshape(len(x), -1), y.reshape(len(y), -1)], axis=1)
        nearest = np.abs(found[:, None, :] - reference["table"][None]).max(axis=-1)
        index = nearest.argmin(axis=1)
        return [list(reference["positions"][i]) for i in index], float(nearest.min(axis=1).max())

    def record(x, y):
        where, distance = locate(x, y)
        write("train_batch", digest=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(), windows=where, distance=distance)
        return x

    def watched(data, floor):
        """``data`` with its select windows and bank written and its train stream read through ``record``."""
        if not reference:
            table_of(data, floor)
        windows, distance = [], 0.0
        for x, y in data.select_ds:
            where, far = locate(x.numpy(), y.numpy())
            windows += where
            distance = max(distance, far)
        write("select", bank=hashlib.sha256(data.bank.tobytes()).hexdigest(), segments=data.segments,
              windows=sorted(windows), distance=distance, stream_seed=data.stream_seed)

        def see(x, y):
            seen = tf.numpy_function(record, [x, y], x.dtype)
            seen.set_shape(x.shape)
            return seen, y

        return dataclasses.replace(data, train_ds=data.train_ds.map(see))

    class Recorder(keras.callbacks.Callback):
        """Append what the run did to the trace; sits last in the list, after the callback that restores the backup."""

        def __init__(self, data) -> None:
            super().__init__()
            self.data, self.epoch, self.first = data, 0, True

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch
            fields = {"epoch": epoch, "iterations": steps_of(self.model)}
            if self.first and spec.get("probe_restored"):
                # The weights a resume restored, scored before any step of this process: the select windows are fixed by
                # (split seed, salt, index), so these are the numbers the killed run logged for the epoch they came from.
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

    class Child(TimeSeriesForecastingTrainer):
        def build_data(self, config, context):
            return watched(super().build_data(config, context), config.transform_args.scale_floor)

        def resume_stream(self, config, data, completed_steps):
            replacement = super().resume_stream(config, data, completed_steps)
            # None is the task saying "keep the data": the fit then reads the stream `build_data` made.
            return None if replacement is None else watched(replacement, config.transform_args.scale_floor)

        def _callbacks(self, config, data, *arguments):
            epoch_line, callbacks = super()._callbacks(config, data, *arguments)
            callbacks.append(Recorder(data))
            if spec.get("stop_at"):
                callbacks.append(Stopper(*spec["stop_at"]))
            return epoch_line, callbacks

    Child(validate(TimeSeriesForecastingConfig, spec["mapping"])).run()
    print("\n@@DONE", flush=True)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
