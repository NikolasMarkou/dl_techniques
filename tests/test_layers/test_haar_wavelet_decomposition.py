"""
Tests for HaarWaveletDecomposition layer.

This module provides comprehensive tests for the multi-dimensional
Haar wavelet decomposition layer.
"""

import tempfile
import os

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.layers.haar_wavelet_decomposition import HaarWaveletDecomposition


class TestHaarWaveletDecompositionInit:
    """Tests for layer initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        layer = HaarWaveletDecomposition()
        assert layer.num_levels == 3

    def test_custom_num_levels(self) -> None:
        """Test initialization with custom num_levels."""
        layer = HaarWaveletDecomposition(num_levels=5)
        assert layer.num_levels == 5

    def test_invalid_num_levels_zero(self) -> None:
        """Test that num_levels=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_levels must be >= 1"):
            HaarWaveletDecomposition(num_levels=0)

    def test_invalid_num_levels_negative(self) -> None:
        """Test that negative num_levels raises ValueError."""
        with pytest.raises(ValueError, match="num_levels must be >= 1"):
            HaarWaveletDecomposition(num_levels=-2)

    def test_layer_name(self) -> None:
        """Test custom layer name."""
        layer = HaarWaveletDecomposition(num_levels=2, name="my_haar")
        assert layer.name == "my_haar"


class TestHaarWaveletDecomposition1D:
    """Tests for 1D (timeseries) decomposition."""

    def test_basic_1d_decomposition(self) -> None:
        """Test basic 1D decomposition output structure."""
        layer = HaarWaveletDecomposition(num_levels=2)
        # Use symbolic input to check for KerasTensor output
        x = keras.Input(shape=(64, 8), batch_size=2)
        outputs = layer(x)

        # Should return [approx, detail_2, detail_1]
        assert len(outputs) == 3
        assert all(isinstance(o, keras.KerasTensor) for o in outputs)

    def test_1d_output_shapes(self) -> None:
        """Test 1D output shapes are correct."""
        layer = HaarWaveletDecomposition(num_levels=3)
        x = keras.random.normal((4, 128, 16), seed=42)
        outputs = layer(x)

        # approx: 128 -> 64 -> 32 -> 16
        assert outputs[0].shape == (4, 16, 16)
        # detail_3 (coarsest): same as approx
        assert outputs[1].shape == (4, 16, 16)
        # detail_2: 32
        assert outputs[2].shape == (4, 32, 16)
        # detail_1 (finest): 64
        assert outputs[3].shape == (4, 64, 16)

    def test_1d_odd_length_handling(self) -> None:
        """Test that odd sequence lengths are handled by truncation."""
        layer = HaarWaveletDecomposition(num_levels=1)
        x = keras.random.normal((2, 65, 4), seed=42)
        outputs = layer(x)

        # 65 -> truncated to 64 -> 32
        assert outputs[0].shape == (2, 32, 4)
        assert outputs[1].shape == (2, 32, 4)

    def test_1d_energy_preservation(self) -> None:
        """Test approximate energy preservation in 1D decomposition."""
        layer = HaarWaveletDecomposition(num_levels=2)
        x = keras.random.normal((1, 64, 1), seed=42)
        outputs = layer(x)

        # Compute energy of input (truncated to even length used)
        input_energy = ops.sum(ops.square(x[:, :64, :]))

        # Compute energy of all outputs
        output_energy = sum(ops.sum(ops.square(o)) for o in outputs)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(input_energy),
            keras.ops.convert_to_numpy(output_energy),
            rtol=1e-5, atol=1e-5,
            err_msg="Energy should be approximately preserved"
        )

    def test_1d_perfect_reconstruction_property(self) -> None:
        """Test Haar wavelet coefficients satisfy reconstruction property."""
        layer = HaarWaveletDecomposition(num_levels=1)

        # Simple test: constant signal should have zero detail
        x = ops.ones((1, 8, 1))
        outputs = layer(x)

        approx = outputs[0]
        detail = outputs[1]

        # For constant input, detail should be zero
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(detail),
            np.zeros_like(keras.ops.convert_to_numpy(detail)),
            rtol=1e-6, atol=1e-6,
            err_msg="Detail coefficients of constant signal should be zero"
        )

        # Approx should be constant * sqrt(2)^levels
        expected_approx = np.sqrt(2) * np.ones((1, 4, 1))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(approx),
            expected_approx,
            rtol=1e-6, atol=1e-6,
            err_msg="Approximation of constant signal should be scaled constant"
        )


class TestHaarWaveletDecomposition2D:
    """Tests for 2D (image) decomposition."""

    def test_basic_2d_decomposition(self) -> None:
        """Test basic 2D decomposition output structure."""
        layer = HaarWaveletDecomposition(num_levels=2)
        # Use symbolic input to check for KerasTensor output
        x = keras.Input(shape=(64, 64, 3), batch_size=2)
        outputs = layer(x)

        # Should return [approx, (LH2, HL2, HH2), (LH1, HL1, HH1)]
        assert len(outputs) == 3
        assert isinstance(outputs[0], keras.KerasTensor)
        assert isinstance(outputs[1], tuple) and len(outputs[1]) == 3
        assert isinstance(outputs[2], tuple) and len(outputs[2]) == 3

    def test_2d_output_shapes(self) -> None:
        """Test 2D output shapes are correct."""
        layer = HaarWaveletDecomposition(num_levels=2)
        x = keras.random.normal((2, 64, 64, 3), seed=42)
        outputs = layer(x)

        # approx: 64x64 -> 32x32 -> 16x16
        assert outputs[0].shape == (2, 16, 16, 3)

        # detail level 2 (coarsest): 16x16
        for detail in outputs[1]:
            assert detail.shape == (2, 16, 16, 3)

        # detail level 1 (finest): 32x32
        for detail in outputs[2]:
            assert detail.shape == (2, 32, 32, 3)

    def test_2d_non_square_input(self) -> None:
        """Test 2D decomposition with non-square input."""
        layer = HaarWaveletDecomposition(num_levels=2)
        x = keras.random.normal((2, 64, 128, 4), seed=42)
        outputs = layer(x)

        # approx: 64x128 -> 32x64 -> 16x32
        assert outputs[0].shape == (2, 16, 32, 4)

    def test_2d_energy_preservation(self) -> None:
        """Test approximate energy preservation in 2D decomposition."""
        layer = HaarWaveletDecomposition(num_levels=2)
        x = keras.random.normal((1, 32, 32, 1), seed=42)
        outputs = layer(x)

        input_energy = ops.sum(ops.square(x))

        # Sum energy of approx and all detail subbands
        output_energy = ops.sum(ops.square(outputs[0]))
        for level_details in outputs[1:]:
            for detail in level_details:
                output_energy = output_energy + ops.sum(ops.square(detail))

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(input_energy),
            keras.ops.convert_to_numpy(output_energy),
            rtol=1e-5, atol=1e-5,
            err_msg="Energy should be approximately preserved in 2D"
        )

    def test_2d_constant_input(self) -> None:
        """Test that constant 2D input has zero detail coefficients."""
        layer = HaarWaveletDecomposition(num_levels=1)
        x = ops.ones((1, 8, 8, 1))
        outputs = layer(x)

        # All detail subbands should be zero for constant input
        for detail in outputs[1]:
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(detail),
                np.zeros_like(keras.ops.convert_to_numpy(detail)),
                rtol=1e-6, atol=1e-6,
                err_msg="Detail coefficients of constant 2D signal should be zero"
            )


class TestHaarWaveletDecomposition3D:
    """Tests for 3D (voxel) decomposition."""

    def test_basic_3d_decomposition(self) -> None:
        """Test basic 3D decomposition output structure."""
        layer = HaarWaveletDecomposition(num_levels=2)
        # Use symbolic input to check for KerasTensor output
        x = keras.Input(shape=(16, 16, 16, 2), batch_size=2)
        outputs = layer(x)

        # Should return [approx, (7 details level 2), (7 details level 1)]
        assert len(outputs) == 3
        assert isinstance(outputs[0], keras.KerasTensor)
        assert isinstance(outputs[1], tuple) and len(outputs[1]) == 7
        assert isinstance(outputs[2], tuple) and len(outputs[2]) == 7

    def test_3d_output_shapes(self) -> None:
        """Test 3D output shapes are correct."""
        layer = HaarWaveletDecomposition(num_levels=2)
        x = keras.random.normal((2, 32, 32, 32, 1), seed=42)
        outputs = layer(x)

        # approx: 32x32x32 -> 16x16x16 -> 8x8x8
        assert outputs[0].shape == (2, 8, 8, 8, 1)

        # detail level 2 (coarsest): 8x8x8
        for detail in outputs[1]:
            assert detail.shape == (2, 8, 8, 8, 1)

        # detail level 1: 16x16x16
        for detail in outputs[2]:
            assert detail.shape == (2, 16, 16, 16, 1)

    def test_3d_non_cubic_input(self) -> None:
        """Test 3D decomposition with non-cubic input."""
        layer = HaarWaveletDecomposition(num_levels=1)
        x = keras.random.normal((1, 8, 16, 32, 2), seed=42)
        outputs = layer(x)

        # approx: 8x16x32 -> 4x8x16
        assert outputs[0].shape == (1, 4, 8, 16, 2)

    def test_3d_energy_preservation(self) -> None:
        """Test approximate energy preservation in 3D decomposition."""
        layer = HaarWaveletDecomposition(num_levels=1)
        x = keras.random.normal((1, 8, 8, 8, 1), seed=42)
        outputs = layer(x)

        input_energy = ops.sum(ops.square(x))

        output_energy = ops.sum(ops.square(outputs[0]))
        for detail in outputs[1]:
            output_energy = output_energy + ops.sum(ops.square(detail))

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(input_energy),
            keras.ops.convert_to_numpy(output_energy),
            rtol=1e-5, atol=1e-5,
            err_msg="Energy should be approximately preserved in 3D"
        )


class TestHaarWaveletDecompositionBuild:
    """Tests for layer build behavior."""

    def test_build_1d(self) -> None:
        """Test build correctly identifies 1D input."""
        layer = HaarWaveletDecomposition(num_levels=2)
        layer.build((None, 64, 8))
        assert layer._ndim == 1

    def test_build_2d(self) -> None:
        """Test build correctly identifies 2D input."""
        layer = HaarWaveletDecomposition(num_levels=2)
        layer.build((None, 64, 64, 3))
        assert layer._ndim == 2

    def test_build_3d(self) -> None:
        """Test build correctly identifies 3D input."""
        layer = HaarWaveletDecomposition(num_levels=2)
        layer.build((None, 16, 16, 16, 1))
        assert layer._ndim == 3

    def test_build_invalid_rank_2(self) -> None:
        """Test build raises error for rank 2 input."""
        layer = HaarWaveletDecomposition(num_levels=2)
        with pytest.raises(ValueError, match="Input must have rank 3"):
            layer.build((None, 64))

    def test_build_invalid_rank_6(self) -> None:
        """Test build raises error for rank 6 input."""
        layer = HaarWaveletDecomposition(num_levels=2)
        with pytest.raises(ValueError, match="Input must have rank 3"):
            layer.build((None, 8, 8, 8, 8, 8))


class TestHaarWaveletDecompositionComputeOutputShape:
    """Tests for compute_output_shape method."""

    def test_compute_output_shape_1d(self) -> None:
        """Test compute_output_shape for 1D input."""
        layer = HaarWaveletDecomposition(num_levels=3)
        shapes = layer.compute_output_shape((None, 128, 16))

        assert len(shapes) == 4  # approx + 3 details
        assert shapes[0] == (None, 16, 16)  # approx
        assert shapes[1] == (None, 16, 16)  # detail_3
        assert shapes[2] == (None, 32, 16)  # detail_2
        assert shapes[3] == (None, 64, 16)  # detail_1

    def test_compute_output_shape_2d(self) -> None:
        """Test compute_output_shape for 2D input."""
        layer = HaarWaveletDecomposition(num_levels=2)
        shapes = layer.compute_output_shape((None, 64, 64, 3))

        assert len(shapes) == 3  # approx + 2 detail levels
        assert shapes[0] == (None, 16, 16, 3)  # approx

        # Each detail level is tuple of 3 shapes
        assert len(shapes[1]) == 3
        assert all(s == (None, 16, 16, 3) for s in shapes[1])

        assert len(shapes[2]) == 3
        assert all(s == (None, 32, 32, 3) for s in shapes[2])

    def test_compute_output_shape_3d(self) -> None:
        """Test compute_output_shape for 3D input."""
        layer = HaarWaveletDecomposition(num_levels=2)
        shapes = layer.compute_output_shape((None, 32, 32, 32, 1))

        assert len(shapes) == 3  # approx + 2 detail levels
        assert shapes[0] == (None, 8, 8, 8, 1)  # approx

        # Each detail level is tuple of 7 shapes
        assert len(shapes[1]) == 7
        assert all(s == (None, 8, 8, 8, 1) for s in shapes[1])

        assert len(shapes[2]) == 7
        assert all(s == (None, 16, 16, 16, 1) for s in shapes[2])

    def test_compute_output_shape_matches_actual_1d(self) -> None:
        """Test compute_output_shape matches actual output for 1D."""
        layer = HaarWaveletDecomposition(num_levels=2)
        input_shape = (4, 64, 8)

        computed_shapes = layer.compute_output_shape(input_shape)

        x = keras.random.normal(input_shape, seed=42)
        outputs = layer(x)

        for computed, actual in zip(computed_shapes, outputs):
            assert tuple(actual.shape) == computed

    def test_compute_output_shape_matches_actual_2d(self) -> None:
        """Test compute_output_shape matches actual output for 2D."""
        layer = HaarWaveletDecomposition(num_levels=2)
        input_shape = (2, 32, 32, 3)

        computed_shapes = layer.compute_output_shape(input_shape)

        x = keras.random.normal(input_shape, seed=42)
        outputs = layer(x)

        # Check approx
        assert tuple(outputs[0].shape) == computed_shapes[0]

        # Check detail levels
        for level_idx in range(1, len(outputs)):
            for detail_idx, detail in enumerate(outputs[level_idx]):
                assert tuple(detail.shape) == computed_shapes[level_idx][detail_idx]

    def test_compute_output_shape_invalid_rank(self) -> None:
        """Test compute_output_shape raises error for invalid rank."""
        layer = HaarWaveletDecomposition(num_levels=2)
        with pytest.raises(ValueError, match="Input must have rank 3"):
            layer.compute_output_shape((None, 64))


class TestHaarWaveletDecompositionSerialization:
    """Tests for layer serialization and deserialization."""

    def test_get_config(self) -> None:
        """Test get_config returns complete configuration."""
        layer = HaarWaveletDecomposition(num_levels=4, name="test_haar")
        config = layer.get_config()

        assert config["num_levels"] == 4
        assert config["name"] == "test_haar"

    def test_from_config(self) -> None:
        """Test from_config creates equivalent layer."""
        original = HaarWaveletDecomposition(num_levels=5, name="original")
        config = original.get_config()

        restored = HaarWaveletDecomposition.from_config(config)

        assert restored.num_levels == original.num_levels
        assert restored.name == original.name

    def test_save_load_model_1d(self) -> None:
        """Test model save/load cycle with 1D input."""
        layer = HaarWaveletDecomposition(num_levels=2)

        inputs = keras.Input(shape=(64, 8))
        outputs = layer(inputs)
        # Wrap in list-to-tuple conversion for model output
        model = keras.Model(inputs=inputs, outputs=outputs)

        x = keras.random.normal((2, 64, 8), seed=42)
        original_outputs = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_outputs = loaded_model(x)

        for orig, loaded in zip(original_outputs, loaded_outputs):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(orig),
                keras.ops.convert_to_numpy(loaded),
                rtol=1e-6, atol=1e-6,
                err_msg="Loaded model outputs should match original"
            )

    def test_save_load_model_2d(self) -> None:
        """Test model save/load cycle with 2D input."""
        layer = HaarWaveletDecomposition(num_levels=2)

        inputs = keras.Input(shape=(32, 32, 3))
        outputs = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        x = keras.random.normal((2, 32, 32, 3), seed=42)
        original_outputs = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model_2d.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_outputs = loaded_model(x)

        # Check approx
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_outputs[0]),
            keras.ops.convert_to_numpy(loaded_outputs[0]),
            rtol=1e-6, atol=1e-6,
            err_msg="Loaded model approx should match original"
        )

        # Check detail levels
        for level_idx in range(1, len(original_outputs)):
            for detail_idx in range(len(original_outputs[level_idx])):
                np.testing.assert_allclose(
                    keras.ops.convert_to_numpy(original_outputs[level_idx][detail_idx]),
                    keras.ops.convert_to_numpy(loaded_outputs[level_idx][detail_idx]),
                    rtol=1e-6, atol=1e-6,
                    err_msg=f"Loaded model detail[{level_idx}][{detail_idx}] should match"
                )


class TestHaarWaveletDecompositionDtypes:
    """Tests for different data types."""

    def test_float32(self) -> None:
        """Test with float32 input."""
        layer = HaarWaveletDecomposition(num_levels=2)
        x = keras.random.normal((2, 32, 8), seed=42, dtype="float32")
        outputs = layer(x)

        assert outputs[0].dtype == "float32"

    def test_float64(self) -> None:
        """Test with float64 input."""
        # Must explicitly set layer dtype to prevent casting to float32
        layer = HaarWaveletDecomposition(num_levels=2, dtype="float64")
        x = ops.cast(keras.random.normal((2, 32, 8), seed=42), "float64")
        outputs = layer(x)

        assert outputs[0].dtype == "float64"


class TestHaarWaveletDecompositionEdgeCases:
    """Tests for edge cases."""

    def test_single_level(self) -> None:
        """Test with single decomposition level."""
        layer = HaarWaveletDecomposition(num_levels=1)
        x = keras.random.normal((2, 16, 4), seed=42)
        outputs = layer(x)

        assert len(outputs) == 2  # approx + 1 detail
        assert outputs[0].shape == (2, 8, 4)
        assert outputs[1].shape == (2, 8, 4)

    def test_many_levels(self) -> None:
        """Test with many decomposition levels."""
        layer = HaarWaveletDecomposition(num_levels=6)
        x = keras.random.normal((1, 256, 1), seed=42)
        outputs = layer(x)

        assert len(outputs) == 7  # approx + 6 details
        # 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
        assert outputs[0].shape == (1, 4, 1)

    def test_minimum_valid_input_size(self) -> None:
        """Test with minimum valid input size for given levels."""
        layer = HaarWaveletDecomposition(num_levels=2)
        x = keras.random.normal((1, 4, 1), seed=42)  # 4 -> 2 -> 1
        outputs = layer(x)

        assert outputs[0].shape == (1, 1, 1)

    def test_batch_size_1(self) -> None:
        """Test with batch size of 1."""
        layer = HaarWaveletDecomposition(num_levels=2)
        x = keras.random.normal((1, 32, 8), seed=42)
        outputs = layer(x)

        assert outputs[0].shape[0] == 1

    def test_large_batch(self) -> None:
        """Test with large batch size."""
        layer = HaarWaveletDecomposition(num_levels=2)
        x = keras.random.normal((64, 32, 8), seed=42)
        outputs = layer(x)

        assert outputs[0].shape[0] == 64

    def test_single_channel(self) -> None:
        """Test with single channel."""
        layer = HaarWaveletDecomposition(num_levels=2)
        x = keras.random.normal((2, 32, 1), seed=42)
        outputs = layer(x)

        assert outputs[0].shape[-1] == 1

    def test_many_channels(self) -> None:
        """Test with many channels."""
        layer = HaarWaveletDecomposition(num_levels=2)
        x = keras.random.normal((2, 32, 256), seed=42)
        outputs = layer(x)

        assert outputs[0].shape[-1] == 256


class TestHaarWaveletDecompositionTrainingMode:
    """Tests for training mode behavior."""

    def test_training_flag_does_not_affect_output(self) -> None:
        """Test that training flag doesn't affect output (layer has no dropout etc)."""
        layer = HaarWaveletDecomposition(num_levels=2)
        x = keras.random.normal((2, 32, 8), seed=42)

        outputs_train = layer(x, training=True)
        outputs_eval = layer(x, training=False)

        for train, eval_ in zip(outputs_train, outputs_eval):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(train),
                keras.ops.convert_to_numpy(eval_),
                rtol=1e-6, atol=1e-6,
                err_msg="Training and eval outputs should be identical"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
