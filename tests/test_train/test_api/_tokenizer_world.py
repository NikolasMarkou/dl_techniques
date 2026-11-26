"""The shared pieces of the ``beit_tokenizer`` tests; no ``test_`` prefix, so pytest does not collect it.

The task is tested on a tiny ``vq_vae_rotation/small`` (64 codes of 8 numbers, 16 channels wide, one residual block: a few thousand
weights) over the REAL CIFAR-10 rows capped by ``data.max_samples``, which a CPU trains in seconds. This module holds what the task, data
and resume tests have in common: where the real rows are read from and the loud skip when they are missing, the request of a tiny run, the
world (an empty registry holding every built-in family and task), what a resolved trainer looks like before its data is built, and the
bring-your-own constant decoder that is the control of every bar (a model that reconstructs one constant image and nothing else).

Interface contract
------------------
``CIFAR_CACHE`` / ``NO_CIFAR``
    The Keras cache the real rows are read from, and the message of the skip a test makes without it. A test that reads real rows never
    downloads anything; its skip says it is NOT a pass.

``request(out, **changes)``
    :return: the mapping of a tiny run (a dotted key, ``"data.max_samples"``, sets a field of a group). ``analysis_args`` is off.

``world`` (fixture, module scope)
    An empty registry holding every built-in family and task, restored afterwards. It reads no data.

``prepared(out, **changes)``
    :return: ``(trainer, resolved config, context)``: what the sequence hands the hooks, resolved the way ``run`` resolves, nothing
        claimed and no data loaded.

``ConstantImage(value)`` / ``constant_model(image, *, size, patch, dim, codes)``
    A layer whose output is one constant image whatever its input, and a ``VQVAERotationTrick`` with it as the decoder: its
    reconstruction loss is the constant image's, which is what ``constant_image_baseline`` computes. Compiled as the run sequence compiles
    it (no loss, no metric).

``positional_model(*, size, patch, dim)``
    A ``VQVAERotationTrick`` whose code is the CELL INDEX alone: the encoder ignores the image (a fixed table, one vector per cell), the codebook
    holds exactly those vectors, so every image gets the same code grid and its pooled perplexity is the number of cells; the decoder is the
    library's own and trains as usual. The control of the per-cell floor (review pass 1, W3). Compiled as the run sequence compiles it.

``read_json(path)`` / ``csv_column(run_dir, name)``
    The parsed JSON file, and a column of ``training_log.csv`` as floats.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.vq_vae_rotation.model import VQVAERotationTrick
from train.api import recipes, registry
from train.api.config import validate
from train.api.tasks import register_builtin_tasks

from ._resume_protocol import merge_changes

CIFAR_CACHE = Path.home() / ".keras" / "datasets" / "cifar-10-batches-py-target"
NO_CIFAR = (
    f"CIFAR-10 is not in {CIFAR_CACHE} on this machine. This file reads the REAL rows and never downloads anything. "
    "A SKIP here is NOT a pass: the guard did not run."
)

MODEL = "vq_vae_rotation/small"
#: The ``small`` variant is 128 wide with 2 residual blocks; the tests shrink it (``model_args`` win over the variant's table). 64 codes is
#: the floor of the bar, so this codebook passes the check that refuses a codebook the bar could never be met with.
TINY_MODEL_ARGS = {"num_embeddings": 64, "embedding_dim": 8, "hidden_channels": 16, "num_res_blocks": 1}
BATCH, EPOCHS = 32, 2
#: Rows of the train pool a tiny run reads: 1,440 fit rows (45 steps of an epoch) and 160 selection rows (5 batches), and the whole test split.
MAX_SAMPLES = 1600
#: The perplexity floor a run that is meant to PASS the bar uses: a tiny model that trains for two epochs does not reach 64 effective codes.
LOW_FLOOR = 2.0


def request(out: Any, **changes: Any) -> Dict[str, Any]:
    """The mapping of a tiny run of ``vq_vae_rotation/small`` on CIFAR-10; see the module contract."""
    asked: Dict[str, Any] = {
        "out": str(out),
        "model": MODEL,
        "data": {"dataset": "cifar10", "max_samples": MAX_SAMPLES},
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "model_args": dict(TINY_MODEL_ARGS),
        # a tiny model over ninety steps wants a rate and no warmup a real run would not take
        "optim_args": {"lr": 3e-3, "warmup_epochs": 0},
        "transform_args": {"augment": False},
        "analysis_args": {"model_analysis": False, "figures": False},
    }
    return merge_changes(asked, changes)


@pytest.fixture(scope="module")
def world():
    """An empty registry holding every built-in family and task."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


def prepared(out: Any, **changes: Any) -> Tuple[Any, Any, Any]:
    """``(trainer, resolved config, context)`` of a tiny request; see the module contract."""
    from train.api.tasks.beit_tokenizer import BeitTokenizerConfig, BeitTokenizerTrainer  # not at module scope: the data tests import this module

    config = validate(BeitTokenizerConfig, request(out, **changes))
    trainer = BeitTokenizerTrainer(config)
    family, variant = registry.parse_model_name(config.model, task=trainer.task)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, variant)
    return trainer, resolved, context


@keras.saving.register_keras_serializable(package="beit_tokenizer_tests")
class ConstantImage(keras.layers.Layer):
    """A layer whose output is one constant image for every input, the batch read off the input (a bring-your-own decoder's output).

    Serializable (a run saves its model): the image is in the layer's config as nested lists of float32 values, which read back exactly.
    """

    def __init__(self, value: np.ndarray, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = np.asarray(value, np.float32)

    def call(self, inputs: Any) -> Any:
        batch = keras.ops.shape(inputs)[0]
        return keras.ops.broadcast_to(keras.ops.convert_to_tensor(self.value), (batch,) + self.value.shape)

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "value": self.value.tolist()}


def constant_model(image: np.ndarray, *, size: int = 32, patch: int = 4, dim: int = 8, codes: int = 16) -> VQVAERotationTrick:
    """A ``VQVAERotationTrick`` with a bring-your-own constant decoder, compiled as the run sequence compiles it (no loss, no metric)."""
    grid = size // patch
    encoder_input = keras.Input((size, size, 3))
    encoder = keras.Model(encoder_input, keras.layers.Conv2D(dim, patch, strides=patch, padding="same")(encoder_input))
    decoder_input = keras.Input((grid, grid, dim))
    value = np.broadcast_to(np.asarray(image, np.float32), (size, size, 3)).copy()
    decoder = keras.Model(decoder_input, ConstantImage(value)(decoder_input))
    model = VQVAERotationTrick(num_embeddings=codes, embedding_dim=dim, encoder=encoder, decoder=decoder)
    model.build((None, size, size, 3))
    model.compile(optimizer="sgd", loss=None, metrics=[])
    return model


def positional_model(*, size: int = 32, patch: int = 4, dim: int = 8, hidden: int = 16) -> VQVAERotationTrick:
    """A ``VQVAERotationTrick`` whose code is the cell index alone, whatever the image; see the module contract."""
    grid = size // patch
    table = (np.random.default_rng(0).normal(size=(grid * grid, dim)) * 3.0).astype(np.float32)
    encoder_input = keras.Input((size, size, 3))
    encoder = keras.Model(encoder_input, ConstantImage(table.reshape(grid, grid, dim))(encoder_input))
    # The decoder of an auto-built model of the same geometry: the library's own, trainable (the auto model itself is dropped).
    auto = VQVAERotationTrick(num_embeddings=grid * grid, embedding_dim=dim, input_shape=(size, size, 3), hidden_channels=hidden,
                              downsample_factor=patch, num_res_blocks=1)
    model = VQVAERotationTrick(num_embeddings=grid * grid, embedding_dim=dim, encoder=encoder, decoder=auto.decoder)
    model.build((None, size, size, 3))
    book = np.zeros(model.quantizer.embeddings.shape, np.float32)
    book[0] = table  # the codebook IS the table: the nearest code of cell c is c, at distance 0
    model.quantizer.embeddings.assign(book)
    model.compile(optimizer="sgd", loss=None, metrics=[])
    return model


def read_json(path: Any) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def csv_column(run_dir: Any, name: str) -> List[float]:
    with open(Path(run_dir) / "training_log.csv", newline="") as handle:
        return [float(row[name]) for row in csv.DictReader(handle)]
