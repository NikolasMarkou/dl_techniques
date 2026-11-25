"""The child process of the text tasks' resume tests: one real ``language_model_pretraining`` or ``text_classification`` fit a parent
can SIGKILL (``test_language_model_pretraining_resume.py``, ``test_text_classification_resume.py``).

Run as ``python _language_child.py '<json>'`` on the CPU, in the marker protocol of ``_resume_protocol.py`` (which holds the trace,
the ``Recorder`` and the ``Stopper`` the stream tasks' children share, and the parent side). The JSON holds ``mapping`` (the
request, as ``validate(<the task's config class>, mapping)`` takes it), ``task`` (``text_classification``; absent means the language
task), ``stop_at`` (``[epoch, done]``: print ``@@AT`` and
sleep when ``done`` batches of ``epoch`` have finished), ``trace`` (a file the child appends JSON lines to, flushed at once, so a
SIGKILL leaves exactly what ran), ``label`` and ``probe_restored``.

The two lines this child adds to the trace: ``select`` (the sha256 of the selection windows of the data the fit will use, written
when the data is built and once more for the stream a resume replaces it with) and ``train_batch`` (the sha256 of the ``(x, y)``
batch a ``tf.numpy_function`` added to ``train_ds`` saw, so it is what the fit really read, in the order it read it). The bank is
offline and the stream is a function of ``(seed, step)``, so no determinism switch beyond the protocol's is needed. Prints
``@@DONE`` at the end.
"""

import dataclasses
import hashlib
import importlib
import json
import sys

import _resume_protocol as protocol

protocol.child_environment()

#: ``task: (module, trainer class, config class)``.
TASKS = {
    "language_model_pretraining": (
        "train.api.tasks.language_model_pretraining", "LanguageModelPretrainingTrainer", "LanguageModelPretrainingConfig"),
    "text_classification": ("train.api.tasks.text_classification", "TextClassificationTrainer", "TextClassificationConfig"),
}


def main(spec: dict) -> None:
    tf = protocol.start_tensorflow()

    import train.api.recipes as recipes
    from train.api.tasks import register_builtin_tasks

    recipes.register_builtin_models()
    register_builtin_tasks()
    module, trainer_name, config_name = TASKS[spec.get("task", "language_model_pretraining")]
    module = importlib.import_module(module)

    def watcher(write):
        def record(x, y):
            write("train_batch", digest=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest())
            return x

        def watched(data, config):
            """``data`` with its selection windows written and its train stream read through ``record``."""
            x, y = data.split.select
            write("select", digest=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest())

            def see(x, y):
                seen = tf.numpy_function(record, [x, y], x.dtype)
                seen.set_shape(x.shape)
                return seen, y

            return dataclasses.replace(data, train_ds=data.train_ds.map(see))

        return watched

    protocol.run_child(spec, getattr(module, trainer_name), getattr(module, config_name), watcher)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
