"""SAM1's mask-prompt embedder: one two-convolution ``ConvStem`` and the 1x1 projection.

``PromptEncoder.mask_downscaling`` was a ``keras.Sequential`` of seven loose layers. It is now a
``Sequential`` of a ``ConvStem`` named ``stem`` (the first six) and ``conv3``. The numeric identity
of the stem with the six layers, for several norms and activations, is proved in
``tests/test_models/test_the_conv_stem_replicates_every_hand_rolled_stem.py``. This file pins what
that file cannot see: every activation name the encoder has always accepted still computes the same
function (a hand-built copy of the old stack, same weights, ``atol=0``), the stem runs once, the
constructor's checks, and the config round trip.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.models.vision_language.sam.sam1.prompt_encoder import (
    PositionEmbeddingRandom,
    PromptEncoder,
)

EMBED_DIM = 32
GRID = (8, 8)
MASK_IN_CHANS = 8

#: Names Keras resolves and the old stack accepted. ``sparsemax`` is the one whose registry layer is
#: another function than Keras' own; the others are listed because they were the cases at risk.
ACTIVATIONS = ["gelu", "relu", "silu", "swish", "tanh", "relu6", "leaky_relu", "elu", "mish", "sparsemax"]


def _encoder(**changes) -> PromptEncoder:
    arguments = dict(
        embed_dim=EMBED_DIM,
        image_embedding_size=GRID,
        input_image_size=(128, 128),
        mask_in_chans=MASK_IN_CHANS,
    )
    arguments.update(changes)
    encoder = PromptEncoder(**arguments)
    encoder.build()
    return encoder


def _masks(batch: int = 2) -> np.ndarray:
    side = 4 * GRID[0]
    return np.random.default_rng(0).uniform(size=(batch, side, side, 1)).astype("float32")


def _old_stack(norm_type: str, activation: str) -> keras.Sequential:
    """The stack as it was written before the migration, built here from layers."""
    return keras.Sequential(
        [
            keras.layers.Conv2D(MASK_IN_CHANS // 4, kernel_size=2, strides=2, name="conv1"),
            create_normalization_layer(norm_type, name="norm1"),
            keras.layers.Activation(activation, name="act1"),
            keras.layers.Conv2D(MASK_IN_CHANS, kernel_size=2, strides=2, name="conv2"),
            create_normalization_layer(norm_type, name="norm2"),
            keras.layers.Activation(activation, name="act2"),
            keras.layers.Conv2D(EMBED_DIM, kernel_size=1, name="conv3"),
        ],
        name="old_mask_downscaling",
    )


def _copy_into_old_stack(encoder: PromptEncoder, old: keras.Sequential) -> None:
    """Copy the stem's and the projection's weights into the old stack, layer for layer."""
    stem = encoder.mask_downscaling.get_layer("stem")
    pairs = [
        (stem.conv1, old.get_layer("conv1")), (stem.norm1, old.get_layer("norm1")),
        (stem.conv2, old.get_layer("conv2")), (stem.norm2, old.get_layer("norm2")),
        (encoder.mask_downscaling.get_layer("conv3"), old.get_layer("conv3")),
    ]
    for new_layer, old_layer in pairs:
        old_layer.set_weights(new_layer.get_weights())


class TestTheStack:

    def test_the_stack_is_the_stem_then_the_projection(self) -> None:
        stack = _encoder().mask_downscaling
        assert [layer.name for layer in stack.layers] == ["stem", "conv3"]
        stem, projection = stack.layers
        assert isinstance(stem, ConvStem)
        assert (stem.filters, stem.conv2_filters, stem.num_convs) == (MASK_IN_CHANS // 4, MASK_IN_CHANS, 2)
        assert (stem.kernel_size, stem.strides, stem.padding) == (2, 2, "valid")
        assert stem.use_bias is True and stem.final_activation is True and stem.pool is None
        assert projection.filters == EMBED_DIM and projection.kernel_size == (1, 1)

    def test_the_stack_keeps_its_ten_trainable_variables(self) -> None:
        stack = _encoder().mask_downscaling
        assert len(stack.trainable_variables) == 10
        assert stack.built is True

    @pytest.mark.parametrize("norm_type", ["layer_norm", "rms_norm", "batch_norm"])
    def test_the_norm_type_reaches_both_norms(self, norm_type) -> None:
        stem = _encoder(normalization_type=norm_type).mask_downscaling.get_layer("stem")
        assert stem.normalization_type == norm_type
        assert stem.norm1 is not None and stem.norm2 is not None

    def test_the_old_layer_names_are_gone(self) -> None:
        names = {layer.name for layer in _encoder().mask_downscaling._flatten_layers(include_self=False)}
        assert not names & {"conv1", "norm1", "act1", "conv2", "norm2", "act2"}, sorted(names)


class TestTheActivationIsKerasOwn:

    @pytest.mark.parametrize("activation", ACTIVATIONS)
    def test_the_stack_equals_the_old_stack_with_the_same_weights(self, activation) -> None:
        keras.utils.set_random_seed(0)
        encoder = _encoder(activation=activation)
        old = _old_stack("layer_norm", activation)
        old.build((None, None, None, 1))
        _copy_into_old_stack(encoder, old)
        x = _masks()
        want = keras.ops.convert_to_numpy(old(x, training=False))
        got = keras.ops.convert_to_numpy(encoder.mask_downscaling(x, training=False))
        assert np.array_equal(want, got), (
            f"{activation}: max|delta| = {float(np.max(np.abs(want - got))):.3e}")

    def test_the_comparison_is_not_vacuous(self) -> None:
        """A different activation gives a different output, so the equality above could fail."""
        keras.utils.set_random_seed(0)
        encoder = _encoder(activation="gelu")
        old = _old_stack("layer_norm", "relu")
        old.build((None, None, None, 1))
        _copy_into_old_stack(encoder, old)
        x = _masks()
        assert float(np.max(np.abs(
            keras.ops.convert_to_numpy(old(x, training=False))
            - keras.ops.convert_to_numpy(encoder.mask_downscaling(x, training=False))))) > 1e-3

    def test_a_name_keras_does_not_know_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError):
            _encoder(activation="not_an_activation")

    def test_the_config_round_trip_keeps_the_activation_and_the_outputs(self) -> None:
        keras.utils.set_random_seed(0)
        encoder = _encoder(activation="sparsemax", normalization_type="rms_norm")
        clone = PromptEncoder.from_config(encoder.get_config())
        clone.build()
        clone.mask_downscaling.set_weights(encoder.mask_downscaling.get_weights())
        x = _masks()
        assert np.array_equal(
            keras.ops.convert_to_numpy(encoder.mask_downscaling(x, training=False)),
            keras.ops.convert_to_numpy(clone.mask_downscaling(x, training=False)))


class TestTheStemRunsOnce:

    def test_a_mask_prompt_runs_the_stem_and_each_of_its_layers_once(self) -> None:
        encoder = _encoder()
        masks = np.transpose(_masks(), (0, 3, 1, 2))  # the encoder takes (B, 1, H, W)
        encoder(masks=masks, training=False)
        stem = encoder.mask_downscaling.get_layer("stem")
        layers = {"stem": stem, "conv1": stem.conv1, "norm1": stem.norm1, "act1": stem.act1,
                  "conv2": stem.conv2, "norm2": stem.norm2, "act2": stem.act2}
        counts = {label: 0 for label in layers}
        originals = {}
        for label, layer in layers.items():
            originals[label] = layer.call

            def counted(*args, _label=label, **kwargs):
                counts[_label] += 1
                return originals[_label](*args, **kwargs)

            layer.call = counted
        try:
            encoder(masks=masks, training=False)
        finally:
            for layer in layers.values():
                del layer.call
        assert counts == {label: 1 for label in layers}


class TestTheConstructor:

    @pytest.mark.parametrize("keyword, value", [("input_shape", (32, 32, 1)), ("input_dim", 3)])
    def test_a_shape_keyword_is_refused_naming_it(self, keyword, value) -> None:
        with pytest.raises(ValueError, match=keyword):
            _encoder(**{keyword: value})
        with pytest.raises(ValueError, match=keyword):
            PositionEmbeddingRandom(**{keyword: value})

    @pytest.mark.parametrize(
        "changes, message",
        [
            ({"embed_dim": 0}, "embed_dim"),
            ({"embed_dim": 2.0}, "embed_dim"),
            ({"mask_in_chans": 0}, "mask_in_chans"),
            ({"mask_in_chans": True}, "mask_in_chans"),
            ({"mask_in_chans": 3}, "at least 4"),
        ],
    )
    def test_a_bad_count_is_refused_naming_the_argument(self, changes, message) -> None:
        with pytest.raises(ValueError, match=message):
            _encoder(**changes)

    def test_a_numpy_count_is_stored_as_a_python_int(self) -> None:
        encoder = _encoder(embed_dim=np.int64(EMBED_DIM), mask_in_chans=np.int64(MASK_IN_CHANS))
        assert type(encoder.embed_dim) is int and type(encoder.mask_in_chans) is int
        assert encoder.get_config()["mask_in_chans"] == MASK_IN_CHANS
