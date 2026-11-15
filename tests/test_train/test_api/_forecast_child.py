"""The child process of ``test_forecast_resume.py``: one real ``time_series_forecasting`` fit a parent can SIGKILL.

Run as ``python _forecast_child.py '<json>'`` on the CPU, in the marker protocol of ``_resume_protocol.py`` (which holds the
trace, the ``Recorder`` and the ``Stopper`` the pretrain and forecasting children share, and the parent side). The JSON holds

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
import sys

import _resume_protocol as protocol

protocol.child_environment()


def main(spec: dict) -> None:
    import numpy as np

    tf = protocol.start_tensorflow()

    import train.api.recipes as recipes
    from _forecast_fixtures import standardize_np
    from train.api.tasks import register_builtin_tasks
    from train.api.tasks.time_series_forecasting import TimeSeriesForecastingConfig, TimeSeriesForecastingTrainer

    recipes.register_builtin_models()
    register_builtin_tasks()
    #: Filled by the first ``build_data`` of the process: ``(table, positions)`` of every window of its bank.
    reference = {}

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

    def watcher(write):
        def record(x, y):
            where, distance = locate(x, y)
            write("train_batch", digest=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(), windows=where, distance=distance)
            return x

        def watched(data, config):
            """``data`` with its select windows and bank written and its train stream read through ``record``."""
            if not reference:
                table_of(data, config.transform_args.scale_floor)
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

        return watched

    protocol.run_child(spec, TimeSeriesForecastingTrainer, TimeSeriesForecastingConfig, watcher)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
