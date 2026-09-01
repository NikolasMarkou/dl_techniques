"""
Test suite for the relocated ``AreaAttention`` layer.

``AreaAttention`` moved out of ``dl_techniques.layers.yolo12_blocks`` into
``dl_techniques.layers.attention.area_attention`` and, on the move, was brought
to ``layers/attention/GUIDE.md`` compliance. This module carries two things:

- the ``TestAreaAttention`` suite relocated verbatim from
  ``tests/test_layers/test_yolo12.py`` (construction, validation, both attention
  branches, output shape, ``.keras`` round trip), retargeted at the NEW class;
- ``TestAreaAttentionGuideSurface``, covering the surface the relocation ADDED
  and which the relocated suite therefore cannot see: the factory path
  (``create_attention_layer('area', ...)``), ``attention_mask``, ``dropout_rate``
  and ``qk_norm_type``. New capability with no test is how this package's
  recorded defects got in, so each of those four has its own guard.

Tests follow dl-techniques Keras 3 conventions: class-based, pytest fixtures,
fixed-seed inputs, headless / GPU-agnostic.
"""

import os
import tempfile
from typing import Any, Dict

import pytest
import numpy as np
import keras

from dl_techniques.layers.attention.area_attention import AreaAttention
from dl_techniques.layers.attention.factory import (
    ATTENTION_REGISTRY,
    create_attention_layer,
)


class TestAreaAttention:
    """Comprehensive test suite for AreaAttention layer."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample 4D input tensor for testing."""
        return keras.random.normal([2, 16, 16, 256])  # batch, height, width, channels

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard layer configuration for testing."""
        return {
            'dim': 256,
            'num_heads': 8,
            'area': 1
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom layer configuration."""
        return {
            'dim': 512,
            'num_heads': 16,
            'area': 4,
            'kernel_initializer': 'glorot_uniform'
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = AreaAttention(dim=256)

        assert layer.dim == 256
        assert layer.num_heads == 8
        assert layer.area == 1
        assert layer.head_dim == 32  # 256 // 8
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)

        # Check sub-layers are created
        assert layer.qk_conv is not None
        assert layer.v_conv is not None
        assert layer.pe_conv is not None
        assert layer.proj_conv is not None

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid dim
        with pytest.raises(ValueError, match="dim must be positive"):
            AreaAttention(dim=0)

        # Test invalid num_heads
        with pytest.raises(ValueError, match="num_heads must be positive"):
            AreaAttention(dim=256, num_heads=0)

        # Test dim not divisible by num_heads
        with pytest.raises(ValueError, match="dim .* must be divisible by num_heads"):
            AreaAttention(dim=256, num_heads=7)

        # Test invalid area
        with pytest.raises(ValueError, match="area must be positive"):
            AreaAttention(dim=256, area=0)

    def test_forward_pass_global_attention(self, sample_input):
        """Test forward pass with global attention (area=1)."""
        layer = AreaAttention(dim=256, num_heads=8, area=1)
        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_forward_pass_area_attention(self):
        """Test forward pass with area-based attention."""
        # Use input size that's divisible by area
        sample_input = keras.random.normal([2, 8, 8, 256])  # 64 total positions
        layer = AreaAttention(dim=256, num_heads=8, area=4)  # 16 positions per area

        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_compute_output_shape(self):
        """Test output shape computation."""
        layer = AreaAttention(dim=128, num_heads=4)

        input_shape = (None, 32, 32, 256)
        expected_shape = (None, 32, 32, 128)

        computed_shape = layer.compute_output_shape(input_shape)
        assert computed_shape == expected_shape

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = AreaAttention(**layer_config)(inputs)
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


class TestAreaAttentionGuideSurface:
    """The surface the relocation ADDED — factory reachability, mask, dropout, qk-norm.

    None of these four is exercised by the relocated suite above, because none of
    them existed before the move. A registry entry that is never CONSTRUCTED and a
    knob that is never EXERCISED are the same defect shape this repository has
    recorded before: an advertised branch dead under a green suite.
    """

    @pytest.fixture
    def spatial_input(self) -> keras.KerasTensor:
        """A small 4D input whose H*W (64) is divisible by the probed `area` values."""
        return keras.random.normal([2, 8, 8, 32], seed=17)

    # -- factory path ----------------------------------------------------

    def test_the_factory_builds_and_runs_the_layer(self, spatial_input):
        """`create_attention_layer('area', ...)` must BUILD and run a forward pass.

        Registering a key without ever constructing through it leaves the branch
        dead while every registry-shape assertion stays green.
        """
        layer = create_attention_layer(
            'area', dim=32, num_heads=4, area=4, name='area_via_factory'
        )

        assert isinstance(layer, AreaAttention)
        assert layer.name == 'area_via_factory'

        output = layer(spatial_input)
        assert tuple(output.shape) == (2, 8, 8, 32)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(output)))

    def test_every_declared_optional_param_is_accepted_by_the_constructor(self):
        """The registry entry must match the class's ACTUAL signature, both ways.

        The factory dispatcher is STRICT: a declared name the constructor does not
        accept raises at build time, and a constructor parameter the registry omits
        is unreachable through the factory. Both halves are asserted, so neither a
        typo nor an omission can pass.
        """
        import inspect

        entry = ATTENTION_REGISTRY['area']
        declared = set(entry['required_params']) | set(entry['optional_params'])
        accepted = {
            name
            for name, p in inspect.signature(AreaAttention.__init__).parameters.items()
            if name not in ('self', 'kwargs')
        }

        assert declared == accepted, (
            "ATTENTION_REGISTRY['area'] drifted from AreaAttention.__init__: "
            f"declared-not-accepted={sorted(declared - accepted)}, "
            f"accepted-not-declared={sorted(accepted - declared)}"
        )

        # And the declared defaults actually construct.
        layer = create_attention_layer(
            'area', dim=32, **dict(entry['optional_params'])
        )
        assert isinstance(layer, AreaAttention)

    # -- attention_mask --------------------------------------------------

    def test_a_spatial_keep_mask_changes_the_output(self, spatial_input):
        """`attention_mask` must actually reach the score matrix.

        The oracle is a DIFFERENCE against the unmasked forward on IDENTICAL
        weights, not a shape or finiteness check: a mask argument that is accepted
        and then dropped passes every shape assertion.
        """
        layer = AreaAttention(dim=32, num_heads=4, area=1)
        unmasked = keras.ops.convert_to_numpy(layer(spatial_input))

        # Keep only the first half of the 64 spatial positions.
        keep = np.zeros((2, 8, 8), dtype="float32")
        keep[:, :4, :] = 1.0
        masked = keras.ops.convert_to_numpy(
            layer(spatial_input, attention_mask=keras.ops.convert_to_tensor(keep))
        )

        assert np.all(np.isfinite(masked))
        assert not np.allclose(unmasked, masked, atol=1e-6), (
            "attention_mask had NO effect on the output — it is being accepted "
            "and dropped"
        )

    def test_a_full_keep_mask_is_an_exact_no_op(self):
        """A full-keep mask is an EXACT no-op, and a full-mask row is rescued.

        Two halves of the mask contract that need no numerical tolerance:

        * ``keep`` all-ones must reproduce ``attention_mask=None`` bit-for-bit —
          a nonzero bias applied at kept positions fails it;
        * ``keep`` all-zeros must stay finite and equal the unmasked forward, which
          is ``apply_attention_mask``'s documented rescue convention (a slice that
          keeps nothing is treated as keeping everything).

        Note what this test deliberately does NOT claim: it is BLIND to an inverted
        polarity, because the rescue turns an all-zero keep back into an all-keep.
        Polarity is pinned by
        ``test_a_masked_out_key_cannot_reach_a_kept_query`` below, which is the
        test that fails when ``keep`` is negated.
        """
        layer = AreaAttention(dim=16, num_heads=2, area=1)
        base = keras.random.normal([1, 4, 4, 16], seed=3)
        _ = layer(base)  # force build so every call shares weights

        keep_all = keras.ops.convert_to_tensor(np.ones((1, 4, 4), dtype="float32"))
        keep_none = keras.ops.convert_to_tensor(np.zeros((1, 4, 4), dtype="float32"))

        without = keras.ops.convert_to_numpy(layer(base))
        with_all = keras.ops.convert_to_numpy(layer(base, attention_mask=keep_all))
        np.testing.assert_allclose(
            without, with_all, rtol=0, atol=0,
            err_msg="an all-KEEP mask is not a no-op: a bias is being applied at "
                    "kept positions"
        )

        with_none = keras.ops.convert_to_numpy(layer(base, attention_mask=keep_none))
        assert np.all(np.isfinite(with_none)), (
            "an all-MASKED row produced non-finite output — the degenerate-slice "
            "rescue is not reaching this layer"
        )
        np.testing.assert_allclose(
            without, with_none, rtol=1e-6, atol=1e-6,
            err_msg="the rescue convention (a slice that keeps nothing keeps "
                    "everything) did not hold"
        )

    def test_a_masked_out_key_cannot_reach_a_kept_query(self):
        """The POLARITY oracle: `1 = keep`, proven by an influence measurement.

        Perturb the input ONLY at masked-out positions and require the kept
        queries' outputs to be unchanged. Under an inverted polarity those very
        positions are the ones being attended to, so the outputs move.

        The geometry is load-bearing and is why this cannot be run on a small map.
        Three of the four sub-layers are 1x1 convolutions and are position-local,
        but ``pe_conv`` is a 5x5 depthwise convolution, i.e. a radius-2 spill that
        has nothing to do with the score matrix. Rows 6:8 are perturbed and rows
        0:3 are examined; row 3's positional-encoding receptive field reaches row 5
        at most, so no non-attention path connects the two regions. An earlier
        draft of this test used a 4x4 map and read the pe_conv spill as a mask
        failure -- an instrument defect, not a finding.
        """
        layer = AreaAttention(dim=16, num_heads=2, area=1)
        base = keras.random.normal([1, 8, 8, 16], seed=11)
        _ = layer(base)  # force build so both calls share weights

        keep = np.ones((1, 8, 8), dtype="float32")
        keep[:, 6:, :] = 0.0  # mask the last two rows
        keep_t = keras.ops.convert_to_tensor(keep)

        perturbed = keras.ops.convert_to_numpy(base).copy()
        perturbed[:, 6:, :, :] += 25.0
        perturbed_t = keras.ops.convert_to_tensor(perturbed)

        out_a = keras.ops.convert_to_numpy(layer(base, attention_mask=keep_t))
        out_b = keras.ops.convert_to_numpy(layer(perturbed_t, attention_mask=keep_t))

        np.testing.assert_allclose(
            out_a[:, :3, :, :], out_b[:, :3, :, :], rtol=1e-4, atol=1e-4,
            err_msg="a masked-out key influenced a kept query's output: the keep "
                    "predicate is inverted, or the mask never reaches the scores"
        )

    # -- dropout_rate ----------------------------------------------------

    def test_dropout_rate_is_active_in_training_and_inert_at_inference(self):
        """`dropout_rate > 0` must be STOCHASTIC in training and inert at inference.

        The oracle is deliberately NOT "training differs from inference": every
        `ConvBlock` sub-layer carries a BatchNorm, so those two regimes differ by
        7.49 with `dropout_rate=0.0` — a difference wholly attributable to batch
        statistics. That confounded oracle passes with the dropout layer deleted.
        What only dropout can produce is a training forward that differs from
        ANOTHER training forward on the same input and weights.
        """
        layer = AreaAttention(dim=32, num_heads=4, area=1, dropout_rate=0.9)
        x = keras.random.normal([2, 8, 8, 32], seed=5)
        _ = layer(x)

        infer_a = keras.ops.convert_to_numpy(layer(x, training=False))
        infer_b = keras.ops.convert_to_numpy(layer(x, training=False))
        np.testing.assert_allclose(
            infer_a, infer_b, rtol=0, atol=0,
            err_msg="inference forward is non-deterministic: `training=` is not "
                    "reaching the attention dropout"
        )

        train_a = keras.ops.convert_to_numpy(layer(x, training=True))
        train_b = keras.ops.convert_to_numpy(layer(x, training=True))
        assert np.all(np.isfinite(train_a))
        assert not np.allclose(train_a, train_b, atol=1e-6), (
            "two training forwards on identical input and weights are identical "
            "with dropout_rate=0.9 — the dropout layer is never applied"
        )

        # And the default is an exact no-op: with dropout_rate=0.0 the same pair of
        # training forwards must agree to the bit.
        plain = AreaAttention(dim=32, num_heads=4, area=1)
        _ = plain(x)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(plain(x, training=True)),
            keras.ops.convert_to_numpy(plain(x, training=True)),
            rtol=0, atol=0,
            err_msg="dropout_rate=0.0 is not an exact no-op"
        )

    # -- qk_norm_type ----------------------------------------------------

    def test_qk_norm_type_builds_weights_and_changes_the_output(self, spatial_input):
        """`qk_norm_type` must create the normalizers AND be applied in `call`.

        The weight-count half catches "the knob is stored but no sub-layer is
        built"; the numerical half catches "the sub-layers are built but never
        called" — the shape this repository recorded as an anchor shipping the
        wrong mechanism.
        """
        plain = AreaAttention(dim=32, num_heads=4, area=1)
        normed = AreaAttention(dim=32, num_heads=4, area=1, qk_norm_type='rms_norm')

        plain_out = keras.ops.convert_to_numpy(plain(spatial_input))
        normed_out = keras.ops.convert_to_numpy(normed(spatial_input))

        assert len(normed.weights) > len(plain.weights), (
            "qk_norm_type='rms_norm' added no weights — the normalizers are not "
            "being built"
        )

        # Copy the shared sub-layer weights across so the ONLY difference between
        # the two forwards is the QK normalization itself.
        by_path = {w.path.split('/', 1)[-1]: w for w in normed.weights}
        for w in plain.weights:
            key = w.path.split('/', 1)[-1]
            assert key in by_path, key
            by_path[key].assign(w)

        normed_out = keras.ops.convert_to_numpy(normed(spatial_input))
        assert np.all(np.isfinite(normed_out))
        assert not np.allclose(plain_out, normed_out, atol=1e-6), (
            "qk_norm_type='rms_norm' left the output bit-identical on identical "
            "weights — the normalizers are built but never applied"
        )

        assert normed.get_config()['qk_norm_type'] == 'rms_norm'
