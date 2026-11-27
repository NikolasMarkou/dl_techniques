"""The shared pieces of the ``dino`` tests; no ``test_`` prefix, so pytest does not collect it.

The data layer and the run callbacks of the ``dino`` task (``train.common.dino_data``, ``train.common.dino_run``) are tested on a tiny
world that a CPU builds in seconds: fake CIFAR-10 rows (noise, so every row is its own identity; ``tests.classification_fakes``), a
``DINOTrainingModel`` whose backbone is 2 blocks deep (the callbacks and the probe do not care how deep the backbone is, and 12 blocks
cost about a minute of tracing per fit), and a ``DINOLoss`` of 64 outputs.

Interface contract
------------------
``IMAGE`` / ``PATCH`` / ``N_LOCAL`` / ``OUT_DIM`` / ``BATCH`` / ``DEPTH`` / ``N_TRAIN`` / ``N_TEST`` / ``BANK_BATCHES`` / ``QUERY_BATCHES``
    The sizes of the world. 220 fake train rows give 22 selection rows (3 batches: 8, 8 and 6) and 198 fit rows (24 steps of 8).
``fake_rows`` (fixture, module scope)
    Replaces the raw CIFAR-10 arrays with ``N_TRAIN`` / ``N_TEST`` rows of noise; nothing reads the Keras cache. Every test that builds
    data asks for it.
``build(**changes)``
    :return: a ``DinoBatches`` at the sizes above (``changes`` override any keyword of ``load_dino_batches``).
``raw_rows()``
    :return: ``(x_train, y_train, x_test, y_test)``, the uint8 arrays the fake serves (read from the same fake, never from the loader under test).
``standardized(rows)``
    uint8 rows to the streams' input in float32, ``(rows / 255 - mean) / std`` per channel (the loader's own statistics).
``images_of(ds, batches=None)`` / ``labels_of(ds, batches=None)``
    The images (labels) of the first ``batches`` batches of ``ds`` (the whole finite pass when ``None``), as one array.
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
"""

from typing import Any, List, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.losses.dino_loss import DINOLoss
from dl_techniques.models.vision.dino.training import create_dino_training_model
from tests.classification_fakes import tiny_raw
from train.common import classification_data, dino_data
from train.common.image_streams import stream_statistics

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


def standardized(rows: np.ndarray) -> np.ndarray:
    mean, std = stream_statistics("cifar10")
    return (rows.astype(np.float32) / 255.0 - mean) / std


def build(**changes: Any) -> dino_data.DinoBatches:
    """``load_dino_batches`` at the sizes of the world."""
    settings = dict(
        dataset="cifar10", image_size=IMAGE, n_local_crops=N_LOCAL, batch_size=BATCH, seed=SEED, stream_seed=None,
        validation_split=None, max_samples=None, knn_bank_batches=BANK_BATCHES, knn_query_batches=QUERY_BATCHES)
    settings.update(changes)
    return dino_data.load_dino_batches(**settings)


def images_of(ds: Any, batches: Any = None) -> np.ndarray:
    taken = ds if batches is None else ds.take(batches)
    return np.concatenate([np.asarray(batch[0]) for batch in taken])


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
