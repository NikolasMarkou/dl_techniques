"""The child process of ``test_beit_tokenizer_resume.py``: one real ``beit_tokenizer`` fit a parent can SIGKILL.

Run as ``python _tokenizer_child.py '<json>'`` on the CPU, in the marker protocol of ``_resume_protocol.py`` (which holds the trace, the
``Recorder`` and the ``Stopper`` the stream tasks' children share, and the parent side). The JSON holds ``mapping`` (the request, as
``validate(BeitTokenizerConfig, mapping)`` takes it), ``stop_at`` (``[epoch, done]``: print ``@@AT`` and sleep when ``done`` batches of
``epoch`` have finished), ``trace`` (a file the child appends JSON lines to, flushed at once, so a SIGKILL leaves exactly what ran),
``label`` and ``probe_restored``.

The two lines this child adds to the trace: ``select`` (the row identity of every selection image of the data the fit will use, written
when the data is built and once more for the stream a resume replaces it with) and ``train_batch`` (the sha256 of the image batch and the
row identities in it, written by a ``tf.numpy_function`` added to ``train_ds``, so it is what the fit really read, in the order it read
it). The data is the identity fake CIFAR-10 of ``tests/classification_fakes``, which the child installs itself; determinism is the golden
run's (op determinism, one thread per pool). Prints ``@@DONE`` at the end.
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
    from train.api.tasks.beit_tokenizer import BeitTokenizerConfig, BeitTokenizerTrainer

    use_identity_cifar10(pytest.MonkeyPatch())
    recipes.register_builtin_models()
    register_builtin_tasks()

    def watcher(write):
        def record(image):
            write("train_batch", digest=hashlib.sha256(image.tobytes()).hexdigest(), rows=row_identities(image, "cifar10", 0).tolist())
            return image

        def watched(data, config):
            """``data`` with its select identities written and its train stream read through ``record``."""
            write("select", rows=sorted(identities_of("cifar10", data.select_ds, 0).tolist()))

            def see(image, label):
                seen = tf.numpy_function(record, [image], image.dtype)
                seen.set_shape(image.shape)
                return seen, label

            return dataclasses.replace(data, train_ds=data.train_ds.map(see))

        return watched

    protocol.run_child(spec, BeitTokenizerTrainer, BeitTokenizerConfig, watcher)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
