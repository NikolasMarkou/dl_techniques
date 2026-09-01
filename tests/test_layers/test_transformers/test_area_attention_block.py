"""Test suite for ``AreaAttentionBlock`` (``layers/transformers/area_attention_block.py``).

The first four classes' worth of behaviour is the suite that used to live as
``TestAttentionBlock`` in ``tests/test_layers/test_yolo12.py``, moved verbatim with the
class it tests and repointed at the D-006 rename (``AttentionBlock`` ->
``AreaAttentionBlock``).

Everything below ``TestNormalizationKwargsThreading`` is NEW surface that the moved suite
never covered, because the pre-move class had none of it: ``normalization_kwargs`` and
``use_bias`` pass-through, and the two architectural DECLINES (no norm/layer-scale/drop-path
on the residual stream; the MLP pair keeps its intermediate BatchNorm) stated as claims that
can go red rather than as prose in a docstring.

Numerical equivalence against the pre-move class is NOT tested here — that is
``tests/test_layers/test_the_yolo12_relocation_is_equivalent.py``, which pins a real
pre-move copy from git rather than a re-implementation.
"""

import os
import tempfile
from typing import Any, Dict

import keras
import numpy as np
import pytest

from dl_techniques.layers.transformers.area_attention_block import AreaAttentionBlock

#: The yolo12 D-067 pair, quoted here as a literal ON PURPOSE. This test file is a
#: consumer-side check that a caller's dict arrives intact at the leaf normalization
#: layers; importing `YOLO12_NORM_KWARGS` would make the assertion tautological with
#: respect to the value, and would also make a `layers/transformers/` test depend on
#: `layers/yolo12_blocks.py`. Any two non-default values would do.
CALLER_NORM_KWARGS = {"epsilon": 1e-3, "momentum": 0.97}

#: `create_normalization_layer`'s own default, i.e. what a caller gets when it passes
#: nothing. Deliberately NOT Keras' `BatchNormalization` default of 1e-3.
FACTORY_DEFAULT_EPSILON = 1e-6


def _batch_norms(layer: keras.layers.Layer):
    """Return every ``BatchNormalization`` in ``layer``'s flattened sub-layer tree."""
    return [
        sub
        for sub in layer._flatten_layers(include_self=True)
        if isinstance(sub, keras.layers.BatchNormalization)
    ]


class TestAreaAttentionBlock:
    """The suite relocated from ``tests/test_layers/test_yolo12.py::TestAttentionBlock``."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample 4D input tensor for testing."""
        return keras.random.normal([2, 16, 16, 256])

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'dim': 256,
            'num_heads': 8,
            'mlp_ratio': 1.2,
            'area': 1
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = AreaAttentionBlock(dim=256)

        assert layer.dim == 256
        assert layer.num_heads == 8
        assert layer.mlp_ratio == 1.2
        assert layer.area == 1
        assert layer.mlp_hidden_dim == int(256 * 1.2)

        # Check sub-layers are created
        assert layer.attn is not None
        assert layer.mlp1 is not None
        assert layer.mlp2 is not None

        # The two relocation knobs default to the pre-move behaviour.
        assert layer.use_bias is False
        assert layer.normalization_kwargs is None

    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="dim must be positive"):
            AreaAttentionBlock(dim=0)

        with pytest.raises(ValueError, match="num_heads must be positive"):
            AreaAttentionBlock(dim=256, num_heads=0)

        with pytest.raises(ValueError, match="mlp_ratio must be positive"):
            AreaAttentionBlock(dim=256, mlp_ratio=0)

        with pytest.raises(ValueError, match="area must be positive"):
            AreaAttentionBlock(dim=256, area=0)

    def test_head_divisibility_is_delegated_to_the_attention_sublayer(self):
        """`dim % num_heads` is checked once, by the attention sub-layer, and still raises.

        The block deliberately does not re-check it (that would give one condition two
        messages), so this asserts the delegation actually reaches the caller.
        """
        with pytest.raises(ValueError):
            AreaAttentionBlock(dim=100, num_heads=8)

    def test_forward_pass_residual_connections(self, sample_input, layer_config):
        """Test that residual connections work properly."""
        layer = AreaAttentionBlock(**layer_config)

        # Ensure input matches expected dimension
        if sample_input.shape[-1] != layer_config['dim']:
            sample_input = keras.random.normal([2, 16, 16, layer_config['dim']])

        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

        # Output should be different from input due to transformations
        diff = keras.ops.mean(keras.ops.abs(output - sample_input))
        assert keras.ops.convert_to_numpy(diff) > 1e-6

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Ensure input matches expected dimension
        if sample_input.shape[-1] != layer_config['dim']:
            sample_input = keras.random.normal([2, 16, 16, layer_config['dim']])

        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = AreaAttentionBlock(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_prediction = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-5, atol=1e-5,
                err_msg="Predictions differ after serialization cycle"
            )

    def test_serialization_carries_the_relocation_knobs(self):
        """`use_bias` and `normalization_kwargs` survive a config round trip.

        A pass-through parameter that `get_config()` forgets is a silent
        reconstruct-with-different-numerics bug, and the round-trip test above cannot see
        it: it serializes a layer built at the defaults, where the forgotten key and its
        default agree.
        """
        original = AreaAttentionBlock(
            dim=64, num_heads=4, area=2,
            use_bias=True,
            normalization_kwargs=dict(CALLER_NORM_KWARGS),
        )
        config = original.get_config()
        assert config["use_bias"] is True
        assert config["normalization_kwargs"] == CALLER_NORM_KWARGS

        restored = AreaAttentionBlock.from_config(config)
        restored.build((None, 8, 8, 64))
        assert restored.use_bias is True
        assert {norm.epsilon for norm in _batch_norms(restored)} == {
            CALLER_NORM_KWARGS["epsilon"]
        }


class TestNormalizationKwargsThreading:
    """`normalization_kwargs` must reach EVERY leaf normalization layer, not just some.

    This is invariant I4's consumer side: the D-067 epsilon/momentum pair has exactly one
    home in `yolo12_blocks.py` and travels to this block as data. A thread that drops the
    dict at one of the three sub-layers is invisible to a shape/round-trip test and
    changes inference.
    """

    def _built_block(self, **kwargs) -> AreaAttentionBlock:
        block = AreaAttentionBlock(dim=32, num_heads=4, area=2, **kwargs)
        block.build((None, 8, 8, 32))
        return block

    def test_the_caller_dict_reaches_every_batch_norm(self):
        block = self._built_block(normalization_kwargs=dict(CALLER_NORM_KWARGS))
        norms = _batch_norms(block)

        # 4 in the attention sub-layer (qk, v, pe, proj) + mlp1 + mlp2.
        assert len(norms) == 6, [n.name for n in norms]
        assert {n.epsilon for n in norms} == {CALLER_NORM_KWARGS["epsilon"]}
        assert {n.momentum for n in norms} == {CALLER_NORM_KWARGS["momentum"]}

    def test_omitting_it_gives_the_factory_default_not_the_caller_value(self):
        """The discriminator's other arm: without the dict the epsilon is the factory's.

        Without this arm the test above passes for a block that hardcodes 1e-3 — which is
        exactly what the pre-move class did and exactly what D-005 forbids.
        """
        block = self._built_block()
        norms = _batch_norms(block)

        assert len(norms) == 6
        assert {n.epsilon for n in norms} == {FACTORY_DEFAULT_EPSILON}

    def test_the_epsilon_is_load_bearing_on_the_output(self):
        """A different epsilon must MOVE the output, on transferred weights.

        Otherwise the two assertions above are pinning an attribute nobody reads.
        """
        keras.utils.set_random_seed(11)
        threaded = self._built_block(normalization_kwargs=dict(CALLER_NORM_KWARGS))
        default = self._built_block()

        assert [w.shape for w in threaded.weights] == [w.shape for w in default.weights]
        default.set_weights([keras.ops.convert_to_numpy(w) for w in threaded.weights])

        x = np.asarray(
            np.random.RandomState(0).normal(size=(2, 8, 8, 32)), dtype="float32"
        )
        delta = float(
            np.max(
                np.abs(
                    keras.ops.convert_to_numpy(threaded(x, training=False))
                    - keras.ops.convert_to_numpy(default(x, training=False))
                )
            )
        )
        assert delta > 1e-4, (
            f"epsilon 1e-3 vs 1e-6 moved the output by only {delta!r}; the threading "
            "assertions above would then be pinning a dead attribute"
        )


class TestUseBiasThreading:
    """`use_bias` must reach all six convolutions, and default to the pre-move `False`."""

    def _built_block(self, **kwargs) -> AreaAttentionBlock:
        block = AreaAttentionBlock(dim=32, num_heads=4, area=2, **kwargs)
        block.build((None, 8, 8, 32))
        return block

    def test_default_is_bias_free(self):
        block = self._built_block()
        convs = [
            sub
            for sub in block._flatten_layers(include_self=True)
            if isinstance(sub, keras.layers.Conv2D)
        ]
        assert len(convs) == 6
        assert [c.use_bias for c in convs] == [False] * 6
        assert all(c.bias is None for c in convs)

    def test_use_bias_true_adds_one_bias_per_convolution(self):
        block = self._built_block(use_bias=True)
        convs = [
            sub
            for sub in block._flatten_layers(include_self=True)
            if isinstance(sub, keras.layers.Conv2D)
        ]
        assert [c.use_bias for c in convs] == [True] * 6
        assert all(c.bias is not None for c in convs)

        # And the flag must reach the WEIGHT TREE, not only the sub-layer attribute.
        bias_free = self._built_block()
        assert len(block.weights) - len(bias_free.weights) == 6


class TestTheArchitecturalDeclinesAreReal:
    """D-007's two declines, asserted rather than described.

    Both are claims about what the block does NOT contain, so both are written as
    membership assertions over the built sub-layer tree. A future "bring it up to the
    house shape" edit reddens these deliberately.
    """

    def _built_block(self) -> AreaAttentionBlock:
        block = AreaAttentionBlock(dim=32, num_heads=4, area=2)
        block.build((None, 8, 8, 32))
        return block

    def test_the_residual_stream_carries_no_norm_layerscale_or_droppath(self):
        """S-2 declined: the block's own sub-layers are exactly attn, mlp1, mlp2."""
        from dl_techniques.layers.layer_scale import LayerScale
        from dl_techniques.layers.stochastic_depth import StochasticDepth

        block = self._built_block()
        direct = list(block._layers) if hasattr(block, "_layers") else []
        direct = [sub for sub in direct if isinstance(sub, keras.layers.Layer)]

        assert [sub.name for sub in direct] == ["attn", "mlp1", "mlp2"], (
            f"the block gained a sub-layer on its residual stream: "
            f"{[s.name for s in direct]}"
        )

        tree = list(block._flatten_layers(include_self=True))
        assert not any(isinstance(sub, LayerScale) for sub in tree)
        assert not any(isinstance(sub, StochasticDepth) for sub in tree)

        # Every normalization in the tree belongs to a ConvBlock (conv -> norm -> act),
        # i.e. none of them sits on the residual stream as a Pre/Post-Norm.
        norm_owners = {
            sub.name for sub in tree if isinstance(sub, keras.layers.BatchNormalization)
        }
        assert norm_owners == {
            "qk_norm", "v_norm", "pe_norm", "proj_norm", "mlp1_norm", "mlp2_norm"
        }, norm_owners

    def test_the_mlp_pair_keeps_its_intermediate_batch_norm(self):
        """S-3 declined: substituting an `ffn/` type would drop `mlp1`'s normalization."""
        block = self._built_block()

        assert isinstance(block.mlp1.norm, keras.layers.BatchNormalization)
        assert isinstance(block.mlp2.norm, keras.layers.BatchNormalization)

        # And it is live, not a decorative attribute: its gamma/beta are in the tree and
        # perturbing gamma moves the block's output.
        gamma = block.mlp1.norm.gamma
        x = np.asarray(
            np.random.RandomState(3).normal(size=(2, 8, 8, 32)), dtype="float32"
        )
        before = keras.ops.convert_to_numpy(block(x, training=False))
        gamma.assign(keras.ops.multiply(gamma, 3.0))
        after = keras.ops.convert_to_numpy(block(x, training=False))

        assert float(np.max(np.abs(after - before))) > 1e-4, (
            "mlp1's BatchNorm gamma does not affect the output -- the intermediate "
            "normalization this block declines to drop is not actually in the graph"
        )

    def test_the_mlp_activation_pair_is_silu_then_linear(self):
        """The expand stage is non-linear and the project stage is an exact identity."""
        block = self._built_block()
        assert block.mlp1.activation_type == "silu"
        assert block.mlp2.activation_type == "linear"


class TestSubLayerCreationOrder:
    """Weight order is a contract, not an accident.

    The relocation's equivalence harness transfers weights by ordered `set_weights`, and
    the yolo12 census counts norms positionally. A reordering of the three `__init__`
    statements is a silent weight-permutation bug that no shape assertion catches, because
    the shape SEQUENCE is what moves.
    """

    def test_weights_are_ordered_attn_then_mlp1_then_mlp2(self):
        block = AreaAttentionBlock(dim=32, num_heads=4, area=2)
        block.build((None, 8, 8, 32))

        prefixes = []
        for weight in block.weights:
            owner = weight.path.split("/")[1]
            if not prefixes or prefixes[-1] != owner:
                prefixes.append(owner)

        assert prefixes == ["attn", "mlp1", "mlp2"], prefixes


class TestAttentionMaskThreading:
    """`call(attention_mask=...)` is new surface; it must reach the attention sub-layer."""

    def test_a_keep_mask_changes_the_output(self):
        keras.utils.set_random_seed(5)
        block = AreaAttentionBlock(dim=32, num_heads=4, area=1)
        x = np.asarray(
            np.random.RandomState(7).normal(size=(2, 8, 8, 32)), dtype="float32"
        )

        unmasked = keras.ops.convert_to_numpy(block(x, training=False))

        # Keep the first four rows, drop the last four.
        mask = np.ones((2, 8, 8), dtype="float32")
        mask[:, 4:, :] = 0.0
        masked = keras.ops.convert_to_numpy(
            block(x, attention_mask=mask, training=False)
        )

        assert float(np.max(np.abs(masked - unmasked))) > 1e-4, (
            "the attention_mask argument does not reach the attention sub-layer"
        )
