"""
Comprehensive test suite for PRISMModel.

Tests cover instantiation, forward pass, serialization, configuration,
shape inference, and integration with training pipeline.
"""

import os
import keras
import logging
import pytest
import tempfile
import numpy as np
from typing import Dict, Any, Tuple

from dl_techniques.models.time_series.prism.model import PRISMModel, create_prism_model

from ..knob_sensitivity_oracle import (
    as_array,
    assert_structural_knob_changes_weights,
    build_seeded,
)


class TestPRISMModelInstantiation:
    """Test model instantiation and configuration validation."""

    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Default model configuration."""
        return {
            "context_len": 96,
            "forecast_len": 24,
            "num_features": 7,
            "hidden_dim": 64,
            "num_layers": 2,
            "tree_depth": 2,
            "overlap_ratio": 0.25,
            "num_wavelet_levels": 3,
            "router_hidden_dim": 64,
            "router_temperature": 1.0,
            "dropout_rate": 0.1,
            "ffn_expansion": 4,
            "kernel_initializer": "glorot_uniform",
        }

    def test_valid_instantiation(self, model_config: Dict[str, Any]) -> None:
        """Test model can be instantiated with valid config."""
        model = PRISMModel(**model_config)

        assert model.context_len == model_config["context_len"]
        assert model.forecast_len == model_config["forecast_len"]
        assert model.num_features == model_config["num_features"]
        assert model.hidden_dim == model_config["hidden_dim"]
        assert model.num_layers == model_config["num_layers"]
        assert len(model.prism_layers) == model_config["num_layers"]

    def test_hidden_dim_defaults_to_num_features(self) -> None:
        """Test hidden_dim defaults to num_features when not specified."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=None,
        )

        assert model.hidden_dim == 7

    def test_invalid_context_len(self) -> None:
        """Test model rejects invalid context_len."""
        with pytest.raises(ValueError, match="context_len must be > 0"):
            PRISMModel(
                context_len=0,
                forecast_len=24,
                num_features=7,
            )

        with pytest.raises(ValueError, match="context_len must be > 0"):
            PRISMModel(
                context_len=-10,
                forecast_len=24,
                num_features=7,
            )

    def test_invalid_forecast_len(self) -> None:
        """Test model rejects invalid forecast_len."""
        with pytest.raises(ValueError, match="forecast_len must be > 0"):
            PRISMModel(
                context_len=96,
                forecast_len=0,
                num_features=7,
            )

    def test_invalid_num_features(self) -> None:
        """Test model rejects invalid num_features."""
        with pytest.raises(ValueError, match="num_features must be > 0"):
            PRISMModel(
                context_len=96,
                forecast_len=24,
                num_features=-5,
            )

    def test_out_of_range_overlap_ratio_raises_rather_than_hangs(self) -> None:
        """A negative ``overlap_ratio`` must RAISE -- it used to HANG.

        RED-proof (measured before the fix, plan-2026-08-18T073231-52a93f8c
        completion fix A): this exact call did not return within 60s and was
        killed by ``timeout`` (exit 124). The remedy search inside the
        ``min_band_len`` refusal was an unbounded ``while True`` over
        ``context_len``, and a negative ``overlap_ratio`` makes the segment
        length negative at EVERY ``context_len`` (-18 at 96, -18750 at
        100000), so the loop could never break. The contract ``[0, 0.5)`` is
        ``PRISMTimeTree.__init__``'s, which is constructed only AFTER this
        arithmetic runs -- hence the duplicate guard.
        """
        with pytest.raises(ValueError, match=r"overlap_ratio must be in \[0, 0.5\)"):
            PRISMModel(
                context_len=96,
                forecast_len=24,
                num_features=7,
                overlap_ratio=-20.0,
            )

    @pytest.mark.parametrize("bad_overlap", [-0.1, 0.5, 0.9, 1.0])
    def test_overlap_ratio_outside_half_open_interval_raises(
        self, bad_overlap: float
    ) -> None:
        """The interval is half-open: 0.5 itself is refused, 0.0 is not."""
        with pytest.raises(ValueError, match="overlap_ratio"):
            PRISMModel(
                context_len=96,
                forecast_len=24,
                num_features=7,
                overlap_ratio=bad_overlap,
            )

    def test_zero_overlap_ratio_is_accepted(self) -> None:
        """The closed end of ``[0, 0.5)`` must still construct."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            overlap_ratio=0.0,
        )
        assert model.overlap_ratio == 0.0

    def test_negative_tree_depth_raises(self) -> None:
        """Negative ``tree_depth`` must RAISE, not silently skip validation.

        The band-validation block used to sit behind
        ``if tree_depth >= 0 and num_wavelet_levels >= 0:``, so a negative
        value bypassed the whole ``min_band_len`` check instead of being
        rejected.
        """
        with pytest.raises(ValueError, match="tree_depth must be >= 0"):
            PRISMModel(
                context_len=96,
                forecast_len=24,
                num_features=7,
                tree_depth=-1,
            )

    def test_negative_num_wavelet_levels_raises(self) -> None:
        """Negative ``num_wavelet_levels`` must RAISE (same skipped-gate bug)."""
        with pytest.raises(ValueError, match="num_wavelet_levels must be >= 0"):
            PRISMModel(
                context_len=96,
                forecast_len=24,
                num_features=7,
                num_wavelet_levels=-3,
            )

    def test_unsupportable_band_config_raises_value_error(self) -> None:
        """A configuration whose deepest frequency band has length 0 is REFUSED.

        The governing quantity is ``min_band_len``, not ``tree_depth``::

            deepest_leaf_seg = PRISMTimeTree._segment_len(
                context_len, overlap_ratio, 2 ** tree_depth)[2]
            min_band_len     = deepest_leaf_seg // 2 ** num_wavelet_levels

        Measured (plan-2026-08-18T073231-52a93f8c step 1, 36-cell grid): at
        ``context_len=96`` the deepest-leaf segment is 54 / 25 / 12 / 6 for
        ``tree_depth`` 1-4, so ``96/4/3`` gives ``6 >> 3 == 0``. Without this
        raise the model builds happily and, since the degenerate-band guard
        landed, returns finite ALL-ZERO band statistics -- silent garbage.
        """
        with pytest.raises(ValueError, match="min_band_len"):
            PRISMModel(
                context_len=96,
                forecast_len=24,
                num_features=7,
                tree_depth=4,
                num_wavelet_levels=3,
            )

    def test_unsupportable_band_message_names_all_four_knobs(self) -> None:
        """The refusal must name every knob the caller can turn to fix it."""
        with pytest.raises(ValueError) as excinfo:
            PRISMModel(
                context_len=96,
                forecast_len=24,
                num_features=7,
                tree_depth=3,
                num_wavelet_levels=4,
            )
        message = str(excinfo.value)
        for knob in (
            "context_len",
            "tree_depth",
            "num_wavelet_levels",
            "overlap_ratio",
        ):
            assert knob in message, f"{knob!r} missing from: {message}"
        assert "min_band_len=0" in message
        assert "deepest_leaf_seg=12" in message

    def test_min_band_len_one_config_is_still_supported(self) -> None:
        """The neighbour of the refused config must NOT be refused.

        ``context_len=96, tree_depth=3, num_wavelet_levels=3`` gives
        ``12 >> 3 == 1``. Threshold = 1 was CONFIRMED by measurement (all six
        ``min_band_len == 1`` grid cells forward finite once the degenerate-band
        guard landed), so this must construct AND forward finite. This is the
        discriminating half of the pair: over-broad validation fails here.
        """
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            tree_depth=3,
            num_wavelet_levels=3,
        )
        inputs = np.random.randn(4, 96, 7).astype(np.float32)
        output = as_array(model(inputs, training=False))
        assert output.shape == (4, 24, 7)
        assert np.all(np.isfinite(output))

    @pytest.mark.parametrize("variant", ["tiny", "small", "base", "large"])
    def test_shipped_variants_survive_band_validation(self, variant: str) -> None:
        """Every shipped preset must remain constructible (invariant I-2).

        If one of these starts raising, the validation rule is wrong -- do not
        loosen the rule to accommodate it without re-measuring.
        """
        model = PRISMModel.from_variant(
            variant,
            context_len=96,
            forecast_len=24,
            num_features=7,
        )
        assert model.context_len == 96


class TestPRISMDegenerateBandWarning:
    """``min_band_len == 1`` is supported, but it must not be silent."""

    def test_warning_fires_at_min_band_len_one(self, caplog) -> None:
        """``context_len=96, tree_depth=3, num_wavelet_levels=3`` -> band 1.

        Measured (36-cell grid, D-002): deepest_leaf_seg 12, ``12 >> 3 == 1``.
        Threshold = 1 ALLOWS this, so nothing raises -- but the deepest bands
        carry a single timestep, where ``mean == min == max`` and both
        first-difference features are a fabricated exact ``0.0``. The only
        record of that used to be README prose a ``from_variant`` caller never
        sees.
        """
        with caplog.at_level(logging.WARNING, logger="dl"):
            PRISMModel(
                context_len=96,
                forecast_len=24,
                num_features=7,
                tree_depth=3,
                num_wavelet_levels=3,
                num_layers=1,
            )
        messages = [r.getMessage() for r in caplog.records]
        degenerate = [m for m in messages if "min_band_len=1" in m]
        assert degenerate, f"no degenerate-boundary warning in: {messages}"
        message = degenerate[0]
        for knob in (
            "context_len",
            "tree_depth",
            "num_wavelet_levels",
            "overlap_ratio",
        ):
            assert knob in message, f"{knob!r} missing from: {message}"
        assert "SINGLE timestep" in message

    def test_warning_does_not_fire_at_min_band_len_two_or_more(
        self, caplog
    ) -> None:
        """The default config (``min_band_len == 3``) must stay quiet.

        ``context_len=96, tree_depth=2, num_wavelet_levels=3``: deepest_leaf_seg
        25, ``25 >> 3 == 3``.
        """
        with caplog.at_level(logging.WARNING, logger="dl"):
            PRISMModel(
                context_len=96,
                forecast_len=24,
                num_features=7,
                tree_depth=2,
                num_wavelet_levels=3,
                num_layers=1,
            )
        messages = [r.getMessage() for r in caplog.records]
        assert not [m for m in messages if "min_band_len=1" in m], (
            f"warning fired at min_band_len=3: {messages}"
        )


class TestPRISMModelTimeAxisContract:
    """The time axis is pinned statically, and what that pin does NOT cover."""

    def test_wrong_static_context_len_is_refused(self) -> None:
        """A static time axis other than ``context_len`` raises at ``__call__``.

        ``context_len`` is a required constructor argument and the whole tree
        geometry is derived from it, so any other static length is a caller
        error rather than a supported mode.
        """
        model = PRISMModel(
            context_len=96, forecast_len=24, num_features=7, num_layers=1
        )
        x = np.random.randn(2, 128, 7).astype("float32")
        with pytest.raises(ValueError, match="axis 1"):
            model(x, training=False)

    def test_dynamic_time_axis_is_an_UNCLOSED_hole(self) -> None:
        """MEASURED limitation, pinned so it cannot rot silently.

        The degenerate-band guard in ``FrequencyBandStatistics.call``
        (decisions.md D-004) branches on the STATIC band length and
        deliberately falls through when it is ``None``. Under a dynamic time
        axis the guard is therefore vacuous and the original all-NaN defect is
        fully present: ``nan_frac == 1.0`` where the same model gives ``0.0``
        eager.

        ``PRISMModel.input_spec`` does NOT close this. Keras'
        ``assert_input_compatibility`` tests ``shape[axis] not in {value,
        None}`` (``keras/src/layers/input_spec.py:223-226``), so an unknown
        dimension is explicitly accepted by an ``axes`` constraint. Measured
        both before and after adding the pin: ``nan_frac == 1.0`` either way.

        This test asserts the CURRENT, BROKEN behaviour on purpose. If a future
        change closes the hole this test goes RED -- that is the intended
        notification, and the fix is to replace the assertion with
        ``nan_frac == 0.0`` (or a raise) rather than to delete the test.
        """
        import tensorflow as tf

        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            tree_depth=3,
            num_layers=1,
        )
        x = np.random.randn(2, 96, 7).astype("float32")

        eager = np.asarray(model(x, training=False))
        assert float(np.mean(np.isnan(eager))) == 0.0, (
            "control: the static-shape path must be finite"
        )

        @tf.function(input_signature=[tf.TensorSpec([None, None, 7], tf.float32)])
        def traced(t):
            return model(t, training=False)

        dynamic = np.asarray(traced(tf.constant(x)))
        assert float(np.mean(np.isnan(dynamic))) == 1.0, (
            "the dynamic-time-axis hole is documented as OPEN; if this is now "
            "finite the limitation has been closed -- update D-004/D-012 and "
            "this assertion, do not delete the test"
        )


class TestPRISMModelForwardPass:
    """Test forward pass and output shapes."""

    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Default model configuration."""
        return {
            "context_len": 96,
            "forecast_len": 24,
            "num_features": 7,
            "hidden_dim": 64,
            "num_layers": 2,
        }

    @pytest.fixture
    def sample_input(self, model_config: Dict[str, Any]) -> np.ndarray:
        """Generate sample input tensor."""
        batch_size = 8
        return np.random.randn(
            batch_size,
            model_config["context_len"],
            model_config["num_features"]
        ).astype(np.float32)

    def test_forward_pass_output_shape(
            self,
            model_config: Dict[str, Any],
            sample_input: np.ndarray
    ) -> None:
        """Test forward pass produces correct output shape."""
        model = PRISMModel(**model_config)
        output = model(sample_input)

        expected_shape = (
            sample_input.shape[0],
            model_config["forecast_len"],
            model_config["num_features"]
        )
        assert output.shape == expected_shape

    def test_forward_pass_dtype(
            self,
            model_config: Dict[str, Any],
            sample_input: np.ndarray
    ) -> None:
        """Test output has correct dtype."""
        model = PRISMModel(**model_config)
        output = model(sample_input)

        assert output.dtype == sample_input.dtype

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
    def test_variable_batch_size(
            self,
            model_config: Dict[str, Any],
            batch_size: int
    ) -> None:
        """Test model handles various batch sizes."""
        model = PRISMModel(**model_config)

        inputs = np.random.randn(
            batch_size,
            model_config["context_len"],
            model_config["num_features"]
        ).astype(np.float32)

        output = model(inputs)

        assert output.shape[0] == batch_size
        assert output.shape[1] == model_config["forecast_len"]
        assert output.shape[2] == model_config["num_features"]

    def test_training_vs_inference_mode(
            self,
            model_config: Dict[str, Any],
            sample_input: np.ndarray
    ) -> None:
        """Test model behaves consistently in training vs inference mode."""
        model = PRISMModel(**model_config)

        # Build model first
        _ = model(sample_input, training=False)

        # Get outputs in both modes
        train_output = model(sample_input, training=True)
        infer_output = model(sample_input, training=False)

        # Shapes should match
        assert train_output.shape == infer_output.shape

        # Note: Outputs may differ due to dropout, so we only check shapes
        # If dropout_rate=0, outputs should be identical

    def test_forward_pass_no_dropout(
            self,
            model_config: Dict[str, Any],
            sample_input: np.ndarray
    ) -> None:
        """Test training vs inference outputs match when dropout=0."""
        config = model_config.copy()
        config["dropout_rate"] = 0.0

        model = PRISMModel(**config)

        train_output = model(sample_input, training=True)
        infer_output = model(sample_input, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(train_output),
            keras.ops.convert_to_numpy(infer_output),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Outputs should match when dropout=0"
        )


class TestPRISMModelShapeInference:
    """Test compute_output_shape functionality."""

    def test_compute_output_shape(self) -> None:
        """Test compute_output_shape returns correct shape."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
        )

        input_shape = (None, 96, 7)
        computed_shape = model.compute_output_shape(input_shape)

        assert computed_shape == (None, 24, 7)

    def test_compute_output_shape_before_build(self) -> None:
        """Test compute_output_shape works before model is built."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
        )

        # Should work even without calling model
        input_shape = (None, 96, 7)
        computed_shape = model.compute_output_shape(input_shape)

        assert computed_shape == (None, 24, 7)

    def test_compute_output_shape_matches_actual(self) -> None:
        """Test compute_output_shape matches actual forward pass output."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
        )

        inputs = np.random.randn(8, 96, 7).astype(np.float32)

        computed_shape = model.compute_output_shape(inputs.shape)
        actual_output = model(inputs)

        assert computed_shape == actual_output.shape

    @pytest.mark.parametrize(
        "context_len,forecast_len,num_features",
        [(48, 12, 3), (96, 24, 7), (192, 48, 12), (336, 96, 21)]
    )
    def test_compute_output_shape_various_configs(
            self,
            context_len: int,
            forecast_len: int,
            num_features: int
    ) -> None:
        """Test compute_output_shape with various configurations."""
        model = PRISMModel(
            context_len=context_len,
            forecast_len=forecast_len,
            num_features=num_features,
        )

        batch_size = 16
        input_shape = (batch_size, context_len, num_features)
        computed_shape = model.compute_output_shape(input_shape)

        assert computed_shape == (batch_size, forecast_len, num_features)


class TestPRISMModelSerialization:
    """Test model serialization and deserialization."""

    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Default model configuration."""
        return {
            "context_len": 96,
            "forecast_len": 24,
            "num_features": 7,
            "hidden_dim": 64,
            "num_layers": 2,
            "tree_depth": 2,
            "overlap_ratio": 0.25,
            "num_wavelet_levels": 3,
            "router_hidden_dim": 64,
            "router_temperature": 1.0,
            "dropout_rate": 0.1,
            "ffn_expansion": 4,
        }

    @pytest.fixture
    def sample_input(self, model_config: Dict[str, Any]) -> np.ndarray:
        """Generate sample input tensor."""
        return np.random.randn(
            8,
            model_config["context_len"],
            model_config["num_features"]
        ).astype(np.float32)

    def test_get_config_complete(self, model_config: Dict[str, Any]) -> None:
        """Test get_config returns all constructor arguments."""
        model = PRISMModel(**model_config)
        config = model.get_config()

        # Check all required parameters are present
        required_keys = [
            "context_len",
            "forecast_len",
            "num_features",
            "hidden_dim",
            "num_layers",
            "tree_depth",
            "overlap_ratio",
            "num_wavelet_levels",
            "router_hidden_dim",
            "router_temperature",
            "dropout_rate",
            "ffn_expansion",
            "kernel_initializer",
            "kernel_regularizer",
        ]

        for key in required_keys:
            assert key in config, f"Missing key in config: {key}"

    def test_from_config_reconstruction(
            self,
            model_config: Dict[str, Any],
            sample_input: np.ndarray
    ) -> None:
        """Test model can be reconstructed from config."""
        original = PRISMModel(**model_config)
        _ = original(sample_input)  # Build model

        config = original.get_config()
        reconstructed = PRISMModel.from_config(config)

        assert reconstructed.context_len == original.context_len
        assert reconstructed.forecast_len == original.forecast_len
        assert reconstructed.num_features == original.num_features
        assert reconstructed.hidden_dim == original.hidden_dim
        assert reconstructed.num_layers == original.num_layers
        assert reconstructed.tree_depth == original.tree_depth

    def test_serialization_cycle(
            self,
            model_config: Dict[str, Any],
            sample_input: np.ndarray
    ) -> None:
        """Test full save/load cycle preserves functionality."""
        model = PRISMModel(**model_config)

        # Get original output
        original_output = model(sample_input, training=False)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_prism_model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        # Get loaded output
        loaded_output = loaded_model(sample_input, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Outputs should match after serialization"
        )

    def test_serialization_with_custom_initializer(
            self,
            sample_input: np.ndarray
    ) -> None:
        """Test serialization with custom kernel initializer."""
        from keras import initializers

        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            kernel_initializer=initializers.HeNormal(),
        )

        _ = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        # Verify initializer was preserved
        config = loaded_model.get_config()
        assert "kernel_initializer" in config

    def test_serialization_with_regularizer(
            self,
            sample_input: np.ndarray
    ) -> None:
        """Test serialization with kernel regularizer."""
        from keras import regularizers

        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            kernel_regularizer=regularizers.L2(0.01),
        )

        _ = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        # Verify regularizer was preserved
        config = loaded_model.get_config()
        assert "kernel_regularizer" in config


class TestPRISMModelPresets:
    """Test model variant configurations."""

    @pytest.mark.parametrize(
        "variant",
        ["tiny", "small", "base", "large"]
    )
    def test_variant_creation(self, variant: str) -> None:
        """Test model can be created from variants."""
        model = PRISMModel.from_variant(
            variant=variant,
            context_len=96,
            forecast_len=24,
            num_features=7,
        )

        assert model.context_len == 96
        assert model.forecast_len == 24
        assert model.num_features == 7

        # Check variant-specific configurations
        variant_config = PRISMModel.MODEL_VARIANTS[variant]
        assert model.hidden_dim == variant_config["hidden_dim"]
        assert model.num_layers == variant_config["num_layers"]
        assert model.tree_depth == variant_config["tree_depth"]

    def test_variant_with_override(self) -> None:
        """Test variant parameters can be overridden."""
        model = PRISMModel.from_variant(
            variant="small",
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=128,  # Override variant value
            num_layers=3,  # Override variant value
        )

        assert model.hidden_dim == 128
        assert model.num_layers == 3

    def test_invalid_variant_raises_error(self) -> None:
        """Test invalid variant name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown variant"):
            PRISMModel.from_variant(
                variant="nonexistent",
                context_len=96,
                forecast_len=24,
                num_features=7,
            )

    def test_all_variants_functional(self) -> None:
        """Test all variants create functional models."""
        inputs = np.random.randn(4, 96, 7).astype(np.float32)

        for variant in PRISMModel.MODEL_VARIANTS.keys():
            model = PRISMModel.from_variant(
                variant=variant,
                context_len=96,
                forecast_len=24,
                num_features=7,
            )

            output = model(inputs)
            assert output.shape == (4, 24, 7)


class TestPRISMModelFactory:
    """Test the create_prism_model module-level factory."""

    def test_factory_builds_point_model(self) -> None:
        """Factory returns a built point-forecast model with valid output."""
        model = create_prism_model(
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=32,
            num_layers=1,
        )

        # Dummy forward inside the factory must have built the model.
        assert model.built
        assert len(model.weights) > 0

        inputs = np.random.randn(4, 96, 7).astype(np.float32)
        output = model(inputs)
        assert output.shape == (4, 24, 7)

    def test_factory_builds_quantile_model(self) -> None:
        """Factory respects quantile-head configuration."""
        model = create_prism_model(
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=32,
            num_layers=1,
            use_quantile_head=True,
            num_quantiles=3,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        assert model.built
        inputs = np.random.randn(4, 96, 7).astype(np.float32)
        output = model(inputs)
        assert output.shape == (4, 24, 7, 3)


class TestPRISMModelBuild:
    """Test model building and weight creation."""

    def test_model_builds_correctly(self) -> None:
        """Test model.build() creates all expected layers."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=64,
            num_layers=2,
        )

        input_shape = (None, 96, 7)
        model.build(input_shape)

        assert model.built
        assert model.input_projection.built

        for layer in model.prism_layers:
            assert layer.built

        assert model.temporal_projector.built
        assert model.head_dropout.built
        assert model.forecast_head.built

    def test_weights_created_after_build(self) -> None:
        """Test weights are created after build."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
        )

        inputs = np.random.randn(4, 96, 7).astype(np.float32)
        _ = model(inputs)  # Triggers build

        weights = model.get_weights()
        assert len(weights) > 0

    def test_model_summary_works(self) -> None:
        """Test model.summary() works after building."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=32,
            num_layers=1,
        )

        inputs = np.random.randn(4, 96, 7).astype(np.float32)
        _ = model(inputs)

        # Should not raise an error
        model.summary()


class TestPRISMModelIntegration:
    """Integration tests with training pipeline."""

    @pytest.fixture
    def training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data."""
        context_len = 96
        forecast_len = 24
        num_features = 7
        num_samples = 100

        x_train = np.random.randn(
            num_samples, context_len, num_features
        ).astype(np.float32)

        y_train = np.random.randn(
            num_samples, forecast_len, num_features
        ).astype(np.float32)

        return x_train, y_train

    def test_compile_and_train(
            self,
            training_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test model can be compiled and trained."""
        x_train, y_train = training_data

        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=32,
            num_layers=1,
        )

        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )

        history = model.fit(
            x_train,
            y_train,
            epochs=2,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )

        assert "loss" in history.history
        assert "mae" in history.history
        assert len(history.history["loss"]) == 2

    def test_save_trained_model(
            self,
            training_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test trained model can be saved and loaded."""
        x_train, y_train = training_data

        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=32,
            num_layers=1,
        )

        model.compile(optimizer="adam", loss="mse")
        model.fit(x_train, y_train, epochs=1, batch_size=16, verbose=0)

        # Get predictions before saving
        test_input = x_train[:5]
        pred_before = model.predict(test_input, verbose=0)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "trained_model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        # Get predictions after loading
        pred_after = loaded_model.predict(test_input, verbose=0)

        np.testing.assert_allclose(
            pred_before,
            pred_after,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Predictions should match after save/load"
        )

    def test_evaluate_on_test_data(
            self,
            training_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test model.evaluate() works correctly."""
        x_train, y_train = training_data

        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            hidden_dim=32,
            num_layers=1,
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        model.fit(x_train, y_train, epochs=1, batch_size=16, verbose=0)

        # Evaluate
        test_loss, test_mae = model.evaluate(
            x_train[:20],
            y_train[:20],
            verbose=0
        )

        assert isinstance(test_loss, float)
        assert isinstance(test_mae, float)
        assert test_loss >= 0
        assert test_mae >= 0


class TestPRISMModelDifferentConfigurations:
    """Test model with various configurations."""

    @pytest.mark.parametrize(
        "context_len,forecast_len",
        [(48, 12), (96, 24), (192, 48), (336, 96)]
    )
    def test_different_sequence_lengths(
            self,
            context_len: int,
            forecast_len: int
    ) -> None:
        """Test model with different sequence lengths."""
        model = PRISMModel(
            context_len=context_len,
            forecast_len=forecast_len,
            num_features=7,
        )

        inputs = np.random.randn(4, context_len, 7).astype(np.float32)
        output = model(inputs)

        assert output.shape == (4, forecast_len, 7)

    @pytest.mark.parametrize("num_features", [1, 3, 7, 12, 21])
    def test_different_feature_dimensions(self, num_features: int) -> None:
        """Test model with different feature dimensions."""
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=num_features,
        )

        inputs = np.random.randn(4, 96, num_features).astype(np.float32)
        output = model(inputs)

        assert output.shape == (4, 24, num_features)

    def test_different_num_layers(self) -> None:
        """``num_layers`` must build the requested stack.

        Restructured out of ``@parametrize``: one model per invocation gave the
        sweep nowhere to compare, so it asserted ``(4, 24, 7)`` -- true at every
        depth -- plus ``len(model.prism_layers) == num_layers``, a STRUCTURAL
        ECHO of the list Python just built. `num_layers` is a parameterisation
        knob, so the weight-shape signature is the discriminating fact.
        """
        layer_counts = [1, 2, 3, 4]
        inputs = np.random.randn(4, 96, 7).astype(np.float32)

        def _build(num_layers):
            model = PRISMModel(
                context_len=96,
                forecast_len=24,
                num_features=7,
                num_layers=num_layers,
            )
            model(inputs[:1])
            return model

        builders = {n: (lambda n=n: _build(n)) for n in layer_counts}
        sigs = assert_structural_knob_changes_weights(builders, knob="num_layers")
        n_weights = [len(sigs[n]) for n in layer_counts]
        assert n_weights == sorted(n_weights) and n_weights[0] < n_weights[-1], (
            f"num_layers did not add weight tensors: {dict(zip(layer_counts, n_weights))}"
        )

        for num_layers in layer_counts:
            model = _build(num_layers)
            assert len(model.prism_layers) == num_layers
            assert model(inputs).shape == (4, 24, 7)

    def test_different_tree_depths(self) -> None:
        """``tree_depth`` must build the requested frequency tree.

        Swept over 1, 2 and 3. Depth 3 was previously EXCLUDED because the
        depth-3 forward was all-NaN; that defect is fixed (see
        :meth:`test_tree_depth_3_forward_is_finite`), so the sweep is whole
        again. Depth 4 is NOT swept here, and not because it is broken:
        at ``context_len=96`` it drives ``min_band_len`` to 0 and is REFUSED by
        ``__init__`` with a ``ValueError`` at the default
        ``num_wavelet_levels=3`` -- that path is covered by
        :meth:`TestPRISMModelInstantiation.test_unsupportable_band_config_raises_value_error`.

        Weight counts below are MEASURED (seed 42, this sweep's own builder),
        not extrapolated: 48 weights / 5,989 params at depth 1, 96 / 10,189 at
        depth 2, 192 / 18,589 at depth 3. The tensor count doubles per level
        (one extra band tree level doubles the node count); the PARAMETER count
        does not -- its increments do (4,200 then 8,400), because the shared
        head is paid once.
        """
        depths = [1, 2, 3]
        inputs = np.random.randn(4, 96, 7).astype(np.float32)

        def _build(tree_depth):
            model = PRISMModel(
                context_len=96,
                forecast_len=24,
                num_features=7,
                tree_depth=tree_depth,
            )
            model(inputs[:1])
            return model

        builders = {d: (lambda d=d: _build(d)) for d in depths}
        sigs = assert_structural_knob_changes_weights(builders, knob="tree_depth")
        # Each extra level doubles the number of tree nodes, hence the tensor
        # count: measured 48 / 96 / 192 weight tensors at depths 1 / 2 / 3.
        for shallow, deep in zip(depths, depths[1:]):
            assert len(sigs[deep]) == 2 * len(sigs[shallow]), (
                f"tree_depth={deep} held {len(sigs[deep])} weight tensors, not "
                f"twice depth {shallow}'s {len(sigs[shallow])}"
            )

        for tree_depth in depths:
            model = _build(tree_depth)
            output = model(inputs)
            assert output.shape == (4, 24, 7)
            assert np.all(np.isfinite(as_array(output)))

    # DECISION plan-2026-08-17T183311-79c63e38/D-037
    # HISTORY (do not delete): D-037 added this test under a strict xfail. The
    # defect it pinned was real and MEASURED: PRISMModel(tree_depth=3) at
    # context_len=96 returned nan_frac 1.0 on seeds 1234 and 7, because at
    # num_wavelet_levels=3 the deepest leaf segment (12) decimates to a band of
    # length 1, whose first-difference tensor is EMPTY -- ops.mean/ops.var over
    # an empty axis return NaN silently, and the router's joint softmax then
    # spread that NaN across every band. The old parametrized sweep asserted
    # only the output SHAPE, so it stayed green throughout.
    # FIXED by plan-2026-08-18T073231-52a93f8c: D-004 (static-length guard in
    # FrequencyBandStatistics.call emitting exact-zero diff features at length
    # 1 and all-zero statistics at length 0) and D-005 (PRISMModel.__init__
    # refuses min_band_len == 0 with a ValueError). The strict xfail did its
    # job -- it turned XPASS the moment D-004 landed -- and was removed in the
    # same commit that restored depth 3 to the sweep above.
    # WHAT THIS TEST GUARDS NOW: that the depth-3 forward stays FINITE. Do NOT
    # relax it to a shape check or a nan-tolerant check, and do not delete it
    # as redundant with the sweep above -- the sweep's oracle is the weight
    # signature, this one's is finiteness of the actual output. Note the
    # governing quantity is min_band_len, NOT tree_depth: a depth-2 config
    # (96/2/4) was equally broken and a depth-4 one (256/4/3) was always fine.
    def test_tree_depth_3_forward_is_finite(self) -> None:
        """The depth-3 forward must be finite. Measured nan_frac 0.0 on seeds 1234 and 7."""
        model = build_seeded(lambda: PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            tree_depth=3,
        ))
        inputs = np.random.randn(4, 96, 7).astype(np.float32)
        output = as_array(model(inputs, training=False))
        assert np.all(np.isfinite(output)), (
            f"{float(np.mean(~np.isfinite(output))) * 100:.0f}% of the depth-3 "
            "forward output is non-finite"
        )

    def test_different_dropout_rates(self) -> None:
        """``dropout_rate`` must reach the training-mode forward.

        An output-difference sweep is the WRONG instrument here: dropout is
        inactive at inference, so two rates give identical inference outputs
        whether or not the kwarg was honoured. The discriminating form is
        training-mode NON-DETERMINISM -- two training-mode calls on one model
        must be bit-identical at rate 0.0 and must differ at rate > 0.

        Measured with identical seeds: rate 0.0 gives exactly 0.0 between two
        training calls, rate 0.3 gives 1.924. The bar is set from that signal.
        """
        inputs = np.random.randn(4, 96, 7).astype(np.float32)

        def _train_mode_spread(dropout_rate):
            model = build_seeded(lambda: PRISMModel(
                context_len=96,
                forecast_len=24,
                num_features=7,
                dropout_rate=dropout_rate,
            ))
            first = as_array(model(inputs, training=True))
            second = as_array(model(inputs, training=True))
            assert first.shape == (4, 24, 7)
            return float(np.max(np.abs(first - second)))

        assert _train_mode_spread(0.0) == 0.0, (
            "dropout_rate=0.0 must make the training-mode forward deterministic"
        )
        for dropout_rate in (0.1, 0.3, 0.5):
            spread = _train_mode_spread(dropout_rate)
            assert spread > 1e-5, (
                f"dropout_rate={dropout_rate} is a no-op: two training-mode "
                f"forwards agreed to {spread:.3e}"
            )

# ---------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])