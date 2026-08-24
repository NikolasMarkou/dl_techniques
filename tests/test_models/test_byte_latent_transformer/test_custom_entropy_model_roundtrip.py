"""Pins BLT serialization with a CUSTOM entropy model (F-84, decisions.md D-030).

`get_config()` used to store the live `EntropyModel` layer object and
`from_config()` was a bare `cls(**config)`. `model.save()` succeeded -- Keras'
config encoder serialized the layer on the way out -- but on reload the value
arrived as a plain dict, `__init__` assigned it straight to
`self.entropy_model`, and the first `build()` died.

MEASURED RED at commit ae2e2aa0a with the fixture below:
``AttributeError: 'TrackedDict' object has no attribute 'built'``.

The default path (`entropy_model=None`) was never affected, which is why the
shipped round-trip test did not see this: no test built a custom entropy model.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.blt_blocks import EntropyModel
from dl_techniques.models.byte_latent_transformer.model import ByteLatentTransformer


VOCAB = 64
SEQ = 16


def _custom_entropy_model() -> EntropyModel:
    return EntropyModel(
        vocab_size=VOCAB,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        max_seq_len=32,
        name="custom_entropy",
    )


def _blt(entropy_model=None) -> ByteLatentTransformer:
    return ByteLatentTransformer(
        vocab_size=VOCAB,
        local_dim=32,
        global_dim=32,
        num_local_layers=1,
        num_global_layers=1,
        num_heads_local=2,
        num_heads_global=2,
        max_sequence_length=32,
        max_patches=8,
        dropout_rate=0.0,
        entropy_model=entropy_model,
    )


def _tokens(batch: int = 2) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, VOCAB, size=(batch, SEQ)).astype("int32")


class TestCustomEntropyModelSerialization:

    def test_get_config_stores_a_serialized_dict_not_a_layer(self):
        """The live-object storage is the mechanism; pin it directly."""
        model = _blt(entropy_model=_custom_entropy_model())
        entry = model.get_config()["entropy_model"]
        assert not isinstance(entry, keras.layers.Layer)
        assert isinstance(entry, dict)
        assert entry["class_name"].endswith("EntropyModel")

    def test_default_entropy_model_still_serializes_as_none(self):
        """The unaffected path must stay unaffected."""
        model = _blt()
        assert model.get_config()["entropy_model"] is None

    def test_from_config_materializes_the_entropy_model(self):
        model = _blt(entropy_model=_custom_entropy_model())
        rebuilt = ByteLatentTransformer.from_config(model.get_config())
        assert isinstance(rebuilt.entropy_model, EntropyModel)
        assert rebuilt.entropy_model.hidden_dim == 32

    def test_save_load_round_trip_with_custom_entropy_model(self, tmp_path):
        """RED at HEAD: `'TrackedDict' object has no attribute 'built'`."""
        model = _blt(entropy_model=_custom_entropy_model())
        x = _tokens()
        before = model(x, training=False)
        path = os.path.join(str(tmp_path), "blt_custom_entropy.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        assert isinstance(loaded.entropy_model, EntropyModel)
        after = loaded(x, training=False)
        before_np = ops.convert_to_numpy(before)
        after_np = ops.convert_to_numpy(after)
        assert before_np.shape == after_np.shape
        np.testing.assert_allclose(before_np, after_np, atol=1e-5, rtol=1e-5)
