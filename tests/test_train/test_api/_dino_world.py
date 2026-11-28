"""The shared pieces of the ``dino`` tests; no ``test_`` prefix, so pytest does not collect it.

The ``dino`` task is tested on tiny worlds that a CPU builds in seconds. This module serves the data layer and the run callbacks
(``train.common.dino_data``, ``train.common.dino_run``) and also the task, wiring, resume-helper, init and predictor tests. The data world
is fake CIFAR-10 rows (noise, so every row is its own identity; ``tests.classification_fakes``), a ``DINOTrainingModel`` whose backbone is
2 blocks deep (the callbacks and the probe do not care how deep the backbone is, and 12 blocks cost about a minute of tracing per fit) and a
``DINOLoss`` of 64 outputs; the task world and the real tiny run (their entries come last) are for the task-level tests.

Interface contract
------------------
``IMAGE`` / ``PATCH`` / ``N_LOCAL`` / ``OUT_DIM`` / ``BATCH`` / ``DEPTH`` / ``N_TRAIN`` / ``N_TEST`` / ``BANK_BATCHES`` / ``QUERY_BATCHES``
    The sizes of the world. 220 fake train rows give 22 selection rows (3 batches: 8, 8 and 6) and 198 fit rows (24 steps of 8).
``N_VIEWS`` / ``SEED`` / ``CLASSES``
    ``2 + N_LOCAL``, the number of views the model takes; the seed of the world (42); the CIFAR-10 class count.
``fake_rows`` (fixture, module scope)
    Replaces the raw CIFAR-10 arrays with ``N_TRAIN`` / ``N_TEST`` rows of noise; nothing reads the Keras cache. Every test that builds
    data asks for it.
``build(**changes)``
    :return: a ``DinoBatches`` at the sizes above (``changes`` override any keyword of ``load_dino_batches``).
``raw_rows()``
    :return: ``(x_train, y_train, x_test, y_test)``, the uint8 arrays the fake serves (read from the same fake, never from the loader under test).
``labels_of(ds, batches=None)``
    The labels of the first ``batches`` batches of ``ds`` (the whole finite pass when ``None``), as one array. ``images_of`` and ``standardized``
    are the ones of ``test_common_tokenizer_data`` (the same function, imported where used: D-116).
``nearest_rows(images, oracle)``
    :return: ``(index, distance)`` arrays: for each image the index of the nearest oracle row and its euclidean distance. Two fake rows
        are noise, so two different rows are hundreds apart and one row against its own standardization is below 1e-3.
``make_model(*, depth=DEPTH, seed=42)``
    :return: a built ``DINOTrainingModel`` (tiny variant, ``depth`` blocks) at ``IMAGE`` px, patch ``PATCH``, ``N_LOCAL`` local views.
``make_loss(**changes)``
    :return: a ``DINOLoss`` of ``OUT_DIM`` outputs.
``compile_model(model, loss)``
    Compiles as the run sequence does: AdamW, the loss, no metric. :return: ``model``.
``views_and_labels(n_batches, seed=3)``
    :return: ``(x, y)`` numpy arrays ``(n_batches * BATCH, 2 + N_LOCAL, IMAGE, IMAGE, 3)`` of noise views and zero labels, for a ``fit``
        with ``shuffle=False`` and no data pipeline in the way.
``weights_of(model)``
    :return: ``[numpy copy of every weight]``.
``same_arrays(a, b)``
    ``True`` when two lists of arrays are equal bit for bit.
``CLAIM_LEAVES`` / ``TASK_MODEL`` / ``TASK_BATCH`` / ``TASK_EPOCHS`` / ``TASK_MAX_SAMPLES``
    What the claim of a run directory writes and so all a run refused after the claim leaves behind (README, ``test_dino_init.py``); the model,
    batch size, epochs and pool rows of the tiny real run.
``task_request(out, **changes)``
    :return: the mapping of a tiny run of ``dino_v1/tiny`` (``dino_out_dim`` 64, 2 local crops, patch 8, 1,000 pool rows, 32 per batch) that a
        dotted key adjusts.
``shallow_backbone(monkeypatch, depth=DEPTH)``
    Makes the factory build a ``depth``-block backbone (12 blocks cost about a minute of tracing per fit on a CPU; nothing a run tests depends
    on the depth).
``flatten_request(asked)``
    :return: ``key=value`` tokens for ``cli.main`` (groups dotted, ``model_args`` one JSON value).
``csv_column(run_dir, name)``
    :return: a column of ``training_log.csv`` as floats.
``world`` / ``identity_world`` (fixtures, module scope)
    An empty registry holding every built-in family and task; the second also serves the identity fake of CIFAR-10 (no real data is read).
``train_real_run(out)``
    One finished 1-epoch run of the real 12-block model through ``cli.main``, with the cifar10 margin patched to what a toy reaches (for the exit
    code only); :return: the exit code. ``load_model`` refuses a model whose depth is not its variant's, so a test of the file a run writes
    (predict, export, init) needs these 12 blocks and not ``shallow_backbone``'s 2.
"""

from typing import Any, List, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.losses.dino_loss import DINOLoss
from dl_techniques.models.vision.dino.training import create_dino_training_model
from tests.classification_fakes import tiny_raw
from train.common import classification_data, dino_data

IMAGE = 32
PATCH = 8
N_LOCAL = 2
N_VIEWS = 2 + N_LOCAL
OUT_DIM = 64
BATCH = 8
DEPTH = 2
SEED = 42
N_TRAIN, N_TEST = 220, 40
BANK_BATCHES, QUERY_BATCHES = 6, 2
CLASSES = 10


@pytest.fixture(scope="module")
def fake_rows():
    """Noise rows in place of the Keras cache: ``N_TRAIN`` train rows and ``N_TEST`` test rows, the same on every read."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(classification_data, "_load_raw", tiny_raw(n_train=N_TRAIN, n_test=N_TEST))
        yield


def raw_rows() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The fake's arrays ``(x_train, y_train, x_test, y_test)`` (uint8 images, int labels), read through the patched seam."""
    (x_train, y_train), (x_test, y_test) = classification_data._load_raw("cifar10")
    return x_train, y_train, x_test, y_test


def build(**changes: Any) -> dino_data.DinoBatches:
    """``load_dino_batches`` at the sizes of the world."""
    settings = dict(
        dataset="cifar10", image_size=IMAGE, n_local_crops=N_LOCAL, batch_size=BATCH, seed=SEED, stream_seed=None,
        validation_split=None, max_samples=None, knn_bank_batches=BANK_BATCHES, knn_query_batches=QUERY_BATCHES)
    settings.update(changes)
    return dino_data.load_dino_batches(**settings)


def labels_of(ds: Any, batches: Any = None) -> np.ndarray:
    taken = ds if batches is None else ds.take(batches)
    return np.concatenate([np.asarray(batch[1]).reshape(-1) for batch in taken])


def nearest_rows(images: np.ndarray, oracle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat, reference = images.reshape(len(images), -1).astype(np.float64), oracle.reshape(len(oracle), -1).astype(np.float64)
    distance = np.sqrt(((flat ** 2).sum(1)[:, None] - 2.0 * flat @ reference.T + (reference ** 2).sum(1)[None, :]).clip(min=0.0))
    return distance.argmin(axis=1), distance.min(axis=1)


def make_model(*, depth: int = DEPTH, seed: int = SEED) -> Any:
    """A built ``DINOTrainingModel`` of the world's size; the same ``seed`` gives the same weights."""
    keras.utils.set_random_seed(seed)
    model = create_dino_training_model(
        "tiny", image_size=IMAGE, patch_size=PATCH, n_local_views=N_LOCAL, dino_out_dim=OUT_DIM, depth=depth)
    model.build((None, N_VIEWS, IMAGE, IMAGE, 3))
    return model


def make_loss(**changes: Any) -> DINOLoss:
    return DINOLoss(**{"out_dim": OUT_DIM, **changes})


def compile_model(model: Any, loss: Any) -> Any:
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3), loss=loss, metrics=[])
    return model


def views_and_labels(n_batches: int, seed: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.random((BATCH * n_batches, N_VIEWS, IMAGE, IMAGE, 3)).astype(np.float32)
    return x, np.zeros((len(x),), np.float32)


def weights_of(model: Any) -> List[np.ndarray]:
    return [np.array(w.numpy(), copy=True) for w in model.weights]


def same_arrays(a: List[np.ndarray], b: List[np.ndarray]) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


# ---------------------------------------------------------------------
# The task world (step 6): a tiny REAL run of the ``dino`` task on the real CIFAR-10 rows
# ---------------------------------------------------------------------

#: What the claim of a run directory writes, and so all a run refused after the claim leaves behind (README, ``test_dino_init.py``).
CLAIM_LEAVES = ("config.json", "config_requested.json", "run.lock", "run.log")
TASK_MODEL = "dino_v1/tiny"
TASK_BATCH, TASK_EPOCHS, TASK_MAX_SAMPLES = 32, 2, 1000


def task_request(out: Any, **changes: Any) -> dict:
    """The mapping of a tiny real run; see the block comment above."""
    from ._resume_protocol import merge_changes

    asked = {
        "out": str(out), "model": TASK_MODEL,
        "data": {"dataset": "cifar10", "max_samples": TASK_MAX_SAMPLES, "patch_size": PATCH, "n_local_crops": N_LOCAL},
        "epochs": TASK_EPOCHS, "batch_size": TASK_BATCH, "model_args": {"dino_out_dim": OUT_DIM},
        "optim_args": {"lr": 1e-3, "warmup_epochs": 0},
        "analysis_args": {"model_analysis": False, "figures": False},
    }
    return merge_changes(asked, changes)


def shallow_backbone(monkeypatch: Any, depth: int = DEPTH) -> None:
    """Make ``create_dino_model`` build a ``depth``-block backbone (the library factory takes ``depth``; the adapter does not pass one)."""
    import dl_techniques.models.vision.dino.training as training

    original = training.create_dino_training_model
    monkeypatch.setattr(training, "create_dino_training_model", lambda *args, **kwargs: original(*args, depth=depth, **kwargs))


def flatten_request(asked: dict) -> list:
    """``key=value`` tokens as the command line takes them: groups are dotted, ``model_args`` is one JSON value."""
    import json

    tokens = []
    for key, value in asked.items():
        if isinstance(value, dict) and key != "model_args":
            tokens += [f"{key}.{inner}={str(v).lower() if isinstance(v, bool) else v}" for inner, v in value.items()]
        else:
            tokens.append(f"{key}={json.dumps(value, separators=(',', ':')) if isinstance(value, dict) else value}")
    return tokens


def csv_column(run_dir: Any, name: str) -> List[float]:
    import csv
    from pathlib import Path

    with open(Path(run_dir) / "training_log.csv", newline="") as handle:
        return [float(row[name]) for row in csv.DictReader(handle)]


@pytest.fixture(scope="module")
def world():
    """An empty registry holding every built-in family and task."""
    from train.api import recipes, registry
    from train.api.tasks import register_builtin_tasks

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


# ---------------------------------------------------------------------
# The real tiny run (steps 7 and 8): the FULL 12-block ``dino_v1/tiny`` on the identity fake of CIFAR-10
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def identity_world():
    from tests.classification_fakes import use_identity_cifar10
    from train.api import recipes, registry
    from train.api.tasks import register_builtin_tasks

    with pytest.MonkeyPatch.context() as patch:
        use_identity_cifar10(patch)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


def train_real_run(out: Any) -> int:
    from train.api import cli
    from train.api.tasks import dino as task

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(task.MARGIN, "cifar10", -0.5)
        # A warmup of one epoch is as long as a 1-epoch run, which the task refuses (D-100): half an epoch trains the teacher for the second half.
        return cli.main(["dino", *flatten_request(task_request(
            out, epochs=1, batch_size=16, model_args={"dino_out_dim": OUT_DIM, "ema_warmup_epochs": 0.5}))])
