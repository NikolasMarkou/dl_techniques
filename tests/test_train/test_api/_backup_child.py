"""The child process of ``test_backup_every_steps.py``: one real API fit that a parent can SIGKILL.

Run as ``python _backup_child.py '<json>'`` on the CPU. The JSON holds

* ``out``: the run directory;
* ``epochs``, ``seed`` (default 7), ``every`` (``checkpoint_args.backup_every_steps``, ``null``
  for the stock callback) and ``resume`` (``resume_interrupted``);
* ``stop_at``: ``[epoch, done]`` makes the child print ``@@AT <epoch> <done>`` and sleep
  when ``done`` batches of ``epoch`` have finished and every callback of the last one
  (the backup included) has run, so a SIGKILL from the parent lands between two batches;
* ``stop_at_meta_write``: ``k`` makes it print ``@@AT_META <k>`` and sleep at the ``k``-th
  write of ``training_metadata.json`` of this process, which is AFTER ``save_weights`` of that
  save returned and BEFORE the metadata is written: the window between the two writes.

The data is the fake CIFAR-10 of ``tests/classification_fakes``; determinism is the golden
run's (op determinism, one thread per pool). Prints ``@@DONE`` when the fit returned.
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
    import train.common.classification_data as classification_data
    from tests.classification_fakes import tiny_raw
    from train.api.config import validate
    from train.api.tasks import register_builtin_tasks
    from train.api.tasks.image_classification import (
        ImageClassificationConfig,
        ImageClassificationTrainer,
    )

    classification_data._load_raw = tiny_raw(n_train=spec.get("n_train", 80), n_test=8)
    recipes.register_builtin_models()
    register_builtin_tasks()

    def stop(marker: str) -> None:
        print(f"\n@@{marker}", flush=True)
        time.sleep(3600)

    metadata_writes = {"count": 0}
    # The module the callback calls ``File`` through: keras.callbacks.BackupAndRestore's own
    # module global, reached by the class's ``__module__`` (keras has no public file_utils).
    file_utils = sys.modules[keras.callbacks.BackupAndRestore.__module__].file_utils
    real_file = file_utils.File
    if spec.get("stop_at_meta_write"):

        def gated_file(path, mode="r", *arguments, **keywords):
            if str(path).endswith("training_metadata.json") and mode == "w":
                metadata_writes["count"] += 1
                if metadata_writes["count"] == spec["stop_at_meta_write"]:
                    stop(f"AT_META {metadata_writes['count']}")
            return real_file(path, mode, *arguments, **keywords)

        file_utils.File = gated_file

    class Stopper(keras.callbacks.Callback):
        """Block when ``done`` batches of ``epoch`` are finished (runs before the next batch)."""

        def __init__(self, epoch: int, done: int) -> None:
            super().__init__()
            self.target = (epoch, done)
            self.epoch = 0

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch
            if self.target == (epoch, 0):
                stop(f"AT {epoch} 0")

        def on_train_batch_begin(self, batch, logs=None):
            if self.target == (self.epoch, batch) and batch > 0:
                stop(f"AT {self.epoch} {batch}")

    class Child(ImageClassificationTrainer):
        def _callbacks(self, *arguments):
            epoch_line, callbacks = super()._callbacks(*arguments)
            if spec.get("stop_at"):
                callbacks.append(Stopper(*spec["stop_at"]))
            return epoch_line, callbacks

    mapping = {
        "out": spec["out"],
        "model": "vit/vit_pico",
        "data": {"dataset": "cifar10"},
        "batch_size": 4,
        "epochs": spec["epochs"],
        "seed": spec.get("seed", 7),
        "resume_interrupted": bool(spec.get("resume")),
        "analysis_args": {"model_analysis": False, "figures": False, "epoch_analysis_every": 0},
        "metric_args": {"success_threshold": 0.0},
        "checkpoint_args": {
            "early_stopping_patience": 50,
            "backup_every_steps": spec.get("every"),
        },
    }
    Child(validate(ImageClassificationConfig, mapping)).run()
    print("\n@@DONE", flush=True)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
