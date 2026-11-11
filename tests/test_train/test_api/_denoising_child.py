"""The child process of ``test_denoising_resume.py``: one real ``image_denoising`` fit a parent can SIGKILL.

Run as ``python _denoising_child.py '<json>'`` on the CPU. It speaks the protocol of ``_backup_child.py`` (the parent
side is ``test_backup_every_steps.run_child``: it reads the ``@@`` markers off the log and SIGKILLs the RECORDED PID),
so the kill is a real one at a real batch. The JSON holds

* ``mapping``: the request, as ``validate(ImageDenoisingConfig, mapping)`` takes it (the parent builds it, so the family,
  the noise kind and the folders are the task tests' own);
* ``stop_at``: ``[epoch, done]`` prints ``@@AT <epoch> <done>`` and sleeps when ``done`` batches of ``epoch`` have
  finished (``done`` above 0), so a SIGKILL lands between two batches of an epoch the curriculum is running;
* ``trace``: a file the child appends JSON lines to, flushed at once, so a SIGKILL leaves exactly what ran:
  ``epoch_begin`` (the epoch Keras passed, the curriculum's ``sigma_max`` and the optimizer's step count, read AFTER the
  curriculum callback ran, and the validation scores of the weights the resume restored when ``probe_restored``),
  ``batch`` (epoch, batch, the sigma the batch drew from, the step count before it), ``epoch_end`` and ``train_end``;
* ``label``: a name copied into every trace line.

The data is the folders of the mapping; determinism is the golden run's (op determinism, one thread per pool).
Prints ``@@DONE`` when ``run()`` returned.
"""

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

    import train.api.recipes as recipes
    from train.api.config import validate
    from train.api.tasks import register_builtin_tasks
    from train.api.tasks.image_denoising import ImageDenoisingConfig, ImageDenoisingTrainer

    recipes.register_builtin_models()
    register_builtin_tasks()

    def number(variable) -> float:
        return float(keras.ops.convert_to_numpy(variable))

    def steps_of(model) -> int:
        optimizer = model.optimizer
        return int(keras.ops.convert_to_numpy(getattr(optimizer, "inner_optimizer", optimizer).iterations))

    class Recorder(keras.callbacks.Callback):
        """Append what the run did to the trace; sits after the curriculum callback in the list."""

        def __init__(self, data) -> None:
            super().__init__()
            self.data, self.epoch, self.first = data, 0, True

        def write(self, kind: str, **fields) -> None:
            with open(spec["trace"], "a") as handle:
                handle.write(json.dumps({"label": spec["label"], "kind": kind, **fields}) + "\n")
                handle.flush()

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch
            fields = {"epoch": epoch, "sigma": number(self.data.sigma_max_var), "iterations": steps_of(self.model)}
            if self.first and spec.get("probe_restored"):
                # The weights a resume restored, scored before any step of this process: the validation set is fixed
                # and its noise stateless, so these are the numbers the killed run's last finished epoch logged.
                fields["restored"] = self.model.evaluate(self.data.select_ds, verbose=0, return_dict=True)
            self.first = False
            self.write("epoch_begin", **fields)

        def on_train_batch_end(self, batch, logs=None):
            # The step count AFTER the batch (the optimizer has applied it) minus one is the count before it.
            self.write("batch", epoch=self.epoch, batch=batch, sigma=number(self.data.sigma_max_var),
                       before=steps_of(self.model) - 1)

        def on_epoch_end(self, epoch, logs=None):
            self.write("epoch_end", epoch=epoch, iterations=steps_of(self.model),
                       val_psnr=float((logs or {}).get("val_psnr", float("nan"))))

        def on_train_end(self, logs=None):
            self.write("train_end", iterations=steps_of(self.model))

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

    class Child(ImageDenoisingTrainer):
        def _callbacks(self, config, data, *arguments):
            epoch_line, callbacks = super()._callbacks(config, data, *arguments)
            callbacks.append(Recorder(data))
            if spec.get("stop_at"):
                callbacks.append(Stopper(*spec["stop_at"]))
            return epoch_line, callbacks

    Child(validate(ImageDenoisingConfig, spec["mapping"])).run()
    print("\n@@DONE", flush=True)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
