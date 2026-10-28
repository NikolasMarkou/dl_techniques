"""Each down-step of the auto-built encoder is one ``ConvStem`` named ``enc_down_{i}``.

``_build_auto_encoder`` wrote every stride-2 step as three loose layers (``enc_down_{i}``,
``enc_norm_{i}``, ``enc_act_{i}``). It is now one ``ConvStem`` per step. The numeric identity
with the three layers, for several norm types and both steps, is proved in
``tests/test_models/test_the_conv_stem_replicates_every_hand_rolled_stem.py``; this file pins what
that file cannot see: the graph the model builds from its constructor arguments, the weight
counts, the ``.keras`` round trip through the nested layer, and the constructor's checks.
"""

import math
import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.models.vision.vq_vae_rotation.model import VQVAERotationTrick

IMAGE_SHAPE = (32, 32, 3)
#: Weights and parameters of the roster model (32 codes, 16 dims, 32 channels, factor 4, one
#: residual block), counted at the revision before the migration with the three loose layers.
WEIGHTS = {"layer_norm": 35, "rms_norm": 29, "batch_norm": 47}
PARAMS = {"layer_norm": 90_643, "rms_norm": 90_451, "batch_norm": 91_027}


def _model(norm_type: str = "layer_norm", **changes) -> VQVAERotationTrick:
    arguments = dict(
        num_embeddings=32,
        embedding_dim=16,
        input_shape=IMAGE_SHAPE,
        hidden_channels=32,
        downsample_factor=4,
        num_res_blocks=1,
        norm_type=norm_type,
    )
    arguments.update(changes)
    return VQVAERotationTrick(**arguments)


def _images(batch: int = 2) -> np.ndarray:
    return np.random.default_rng(0).uniform(size=(batch, *IMAGE_SHAPE)).astype("float32")


class TestTheEncoderGraph:

    @pytest.mark.parametrize("factor, steps", [(2, 1), (4, 2), (8, 3)])
    def test_one_stem_per_halving_numbered_from_zero(self, factor, steps) -> None:
        encoder = _model(downsample_factor=factor).encoder
        stems = [layer for layer in encoder.layers if isinstance(layer, ConvStem)]
        assert [layer.name for layer in stems] == [f"enc_down_{i}" for i in range(steps)]
        assert steps == (int(math.log2(factor)))

    def test_none_of_the_loose_layers_remains(self) -> None:
        names = {layer.name for layer in _model().encoder.layers}
        loose = {f"enc_norm_{i}" for i in range(3)} | {f"enc_act_{i}" for i in range(3)}
        assert not names & loose, sorted(names & loose)

    @pytest.mark.parametrize("norm_type", ["layer_norm", "rms_norm", "batch_norm"])
    def test_the_norm_type_reaches_the_stem(self, norm_type) -> None:
        stem = _model(norm_type).encoder.get_layer("enc_down_0")
        assert stem.normalization_type == norm_type
        assert stem.norm1 is not None and stem.activation_type == "relu"
        assert (stem.filters, stem.kernel_size, stem.strides, stem.padding) == (32, 4, 2, "same")
        assert stem.use_bias is True and stem.num_convs == 1 and stem.pool is None

    @pytest.mark.parametrize("norm_type", ["layer_norm", "rms_norm", "batch_norm"])
    def test_the_weights_are_the_ones_the_loose_layers_held(self, norm_type) -> None:
        model = _model(norm_type)
        model(_images(), training=False)
        stem_weights = model.encoder.get_layer("enc_down_0").weights
        assert len(stem_weights) == 2 + {"layer_norm": 2, "rms_norm": 1, "batch_norm": 4}[norm_type]
        assert len(model.weights) == WEIGHTS[norm_type]
        assert model.count_params() == PARAMS[norm_type]


class TestTheRoundTrip:

    @pytest.mark.parametrize("norm_type", ["layer_norm", "batch_norm"])
    def test_a_saved_model_reloads_with_equal_outputs(self, norm_type) -> None:
        keras.utils.set_random_seed(0)
        model = _model(norm_type)
        x = _images()
        want = keras.ops.convert_to_numpy(model(x, training=False))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "vq.keras")
            model.save(path)
            restored = keras.models.load_model(path)
        got = keras.ops.convert_to_numpy(restored(x, training=False))
        assert np.array_equal(want, got)
        assert isinstance(restored.encoder.get_layer("enc_down_1"), ConvStem)

    def test_the_config_round_trip_builds_the_same_stems(self) -> None:
        model = _model()
        clone = VQVAERotationTrick.from_config(model.get_config())
        assert [layer.name for layer in clone.encoder.layers] == [
            layer.name for layer in model.encoder.layers]


class TestTheConstructor:

    @pytest.mark.parametrize(
        "changes, message",
        [
            ({"num_embeddings": 0}, "num_embeddings"),
            ({"num_embeddings": True}, "num_embeddings"),
            ({"num_embeddings": 3.0}, "num_embeddings"),
            ({"embedding_dim": -1}, "embedding_dim"),
            ({"hidden_channels": 0}, "hidden_channels"),
            ({"downsample_factor": 0}, "downsample_factor"),
            ({"downsample_factor": 3}, "power of 2"),
            ({"reconstruction_loss_weight": 0}, "reconstruction_loss_weight"),
            ({"reconstruction_loss_weight": float("nan")}, "reconstruction_loss_weight"),
        ],
    )
    def test_a_bad_value_is_refused_naming_the_argument(self, changes, message) -> None:
        with pytest.raises(ValueError, match=message):
            _model(**changes)

    def test_a_numpy_count_is_stored_as_a_python_int(self) -> None:
        model = _model(num_embeddings=np.int64(32), embedding_dim=np.int64(16), hidden_channels=np.int64(32))
        assert type(model.num_embeddings) is int and type(model.embedding_dim) is int
        assert type(model.hidden_channels) is int
        assert model.get_config()["num_embeddings"] == 32
