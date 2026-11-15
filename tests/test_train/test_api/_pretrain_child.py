"""The child process of ``test_pretrain_resume.py``: one real ``pretrain`` fit a parent can SIGKILL.

Run as ``python _pretrain_child.py '<json>'`` on the CPU, in the marker protocol of ``_resume_protocol.py`` (which holds the
trace, the ``Recorder`` and the ``Stopper`` the pretrain and forecasting children share, and the parent side). The JSON holds

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
import sys

import _resume_protocol as protocol

protocol.child_environment()


def main(spec: dict) -> None:
    tf = protocol.start_tensorflow()

    import pytest

    import train.api.recipes as recipes
    from tests.classification_fakes import identities_of, row_identities, use_identity_cifar10
    from train.api.tasks import register_builtin_tasks
    from train.api.tasks.pretrain import PretrainConfig, PretrainTrainer
    from train.common import pretrain_run

    use_identity_cifar10(pytest.MonkeyPatch())
    if spec.get("floor") is not None:
        pretrain_run.MIN_TOKENIZER_PERPLEXITY = spec["floor"]
    recipes.register_builtin_models()
    register_builtin_tasks()

    def watcher(write):
        def record(image, mask):
            digest = hashlib.sha256(image.tobytes() + mask.tobytes()).hexdigest()
            write("train_batch", digest=digest, rows=row_identities(image, "cifar10", 0).tolist())
            return image

        def watched(data, config):
            """``data`` with its select identities written and its train stream read through ``record``."""
            write("select", rows=sorted(identities_of("cifar10", data.select_ds, 0).tolist()))

            def see(inputs, target, weight):
                image, mask = inputs
                seen = tf.numpy_function(record, [image, mask], image.dtype)
                seen.set_shape(image.shape)
                return (seen, mask), target, weight

            return dataclasses.replace(data, train_ds=data.train_ds.map(see))

        return watched

    protocol.run_child(spec, PretrainTrainer, PretrainConfig, watcher)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
