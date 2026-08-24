"""
Comprehensive test suite for the CapsNet model.

This module contains all tests for the CapsNet model implementation,
covering initialization, forward pass, training, serialization,
model integration, metrics, and edge cases.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os
from typing import Tuple

from dl_techniques.models.capsnet.model import CapsNet, CapsuleAccuracy, create_capsnet
from dl_techniques.layers.capsules import PrimaryCapsule, RoutingCapsule
from dl_techniques.losses.capsule_margin_loss import capsule_margin_loss
from dl_techniques.utils.tensors import length

from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    build_seeded,
    as_array,
)


class TestCapsuleAccuracy:
    """Test suite for CapsuleAccuracy metric."""

    @pytest.fixture
    def metric(self):
        """Create CapsuleAccuracy metric instance."""
        return CapsuleAccuracy()

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions (capsule lengths)."""
        batch_size = 8
        num_classes = 10
        return tf.random.uniform([batch_size, num_classes], 0, 1)

    @pytest.fixture
    def sample_labels(self):
        """Create sample one-hot labels."""
        batch_size = 8
        num_classes = 10
        labels = tf.random.uniform([batch_size], 0, num_classes, dtype=tf.int32)
        return tf.one_hot(labels, num_classes)

    def test_metric_initialization(self, metric):
        """Test metric initialization."""
        assert metric.name == "capsule_accuracy"
        assert metric.total.numpy() == 0.0
        assert metric.count.numpy() == 0.0

    def test_metric_update_with_lengths(self, metric, sample_predictions, sample_labels):
        """Test metric update with capsule lengths."""
        metric.update_state(sample_labels, sample_predictions)

        result = metric.result()
        assert 0.0 <= result.numpy() <= 1.0
        assert metric.count.numpy() == sample_labels.shape[0]

    def test_metric_update_with_dict(self, metric, sample_predictions, sample_labels):
        """Test metric update with dictionary containing lengths."""
        pred_dict = {"length": sample_predictions, "other": tf.zeros_like(sample_predictions)}
        metric.update_state(sample_labels, pred_dict)

        result = metric.result()
        assert 0.0 <= result.numpy() <= 1.0

    def test_metric_reset(self, metric, sample_predictions, sample_labels):
        """Test metric reset functionality."""
        metric.update_state(sample_labels, sample_predictions)
        initial_result = metric.result()

        metric.reset_state()
        assert metric.total.numpy() == 0.0
        assert metric.count.numpy() == 0.0
        assert metric.result().numpy() == 0.0

    def test_metric_accuracy_calculation(self, metric):
        """Test accuracy calculation with known values."""
        # Create perfect predictions
        labels = tf.one_hot([0, 1, 2], 3)
        predictions = tf.constant([[0.9, 0.1, 0.1],
                                  [0.1, 0.9, 0.1],
                                  [0.1, 0.1, 0.9]])

        metric.update_state(labels, predictions)
        result = metric.result()
        assert result.numpy() == 1.0  # Perfect accuracy

        # Reset and test with wrong predictions
        metric.reset_state()
        wrong_predictions = tf.constant([[0.1, 0.9, 0.1],  # Wrong
                                        [0.9, 0.1, 0.1],   # Wrong
                                        [0.1, 0.1, 0.9]])  # Correct

        metric.update_state(labels, wrong_predictions)
        result = metric.result()
        assert abs(result.numpy() - 1/3) < 1e-6  # 1 out of 3 correct


class TestCapsNet:
    """Test suite for CapsNet model implementation."""

    @pytest.fixture
    def mnist_input_shape(self) -> Tuple[int, int, int]:
        """Create MNIST input shape."""
        return (28, 28, 1)

    @pytest.fixture
    def cifar_input_shape(self) -> Tuple[int, int, int]:
        """Create CIFAR-10 input shape."""
        return (32, 32, 3)

    @pytest.fixture
    def num_classes(self) -> int:
        """Number of classes for testing."""
        return 10

    @pytest.fixture
    def sample_mnist_data(self, mnist_input_shape):
        """Create sample MNIST-like data."""
        batch_size = 4
        return tf.random.uniform([batch_size] + list(mnist_input_shape), 0, 1)

    @pytest.fixture
    def sample_cifar_data(self, cifar_input_shape):
        """Create sample CIFAR-like data."""
        batch_size = 4
        return tf.random.uniform([batch_size] + list(cifar_input_shape), 0, 1)

    @pytest.fixture
    def sample_labels(self, num_classes):
        """Create sample one-hot labels."""
        batch_size = 4
        labels = tf.random.uniform([batch_size], 0, num_classes, dtype=tf.int32)
        return tf.one_hot(labels, num_classes)

    def test_initialization_defaults(self, num_classes):
        """Test initialization with default parameters."""
        capsnet = CapsNet(num_classes=num_classes)

        assert capsnet.num_classes == num_classes
        assert capsnet.routing_iterations == 3
        assert capsnet.conv_filters == [256, 256]
        assert capsnet.primary_capsules == 32
        assert capsnet.primary_capsule_dim == 8
        assert capsnet.digit_capsule_dim == 16
        assert capsnet.reconstruction is True
        assert capsnet.use_batch_norm is True
        assert capsnet.positive_margin == 0.9
        assert capsnet.negative_margin == 0.1
        assert capsnet.downweight == 0.5
        assert capsnet.reconstruction_weight == 0.01
        assert capsnet._layers_built is False

    def test_initialization_custom(self, num_classes, mnist_input_shape):
        """Test initialization with custom parameters."""
        capsnet = CapsNet(
            num_classes=num_classes,
            routing_iterations=5,
            conv_filters=[128, 256],
            primary_capsules=16,
            primary_capsule_dim=4,
            digit_capsule_dim=8,
            reconstruction=False,
            input_shape=mnist_input_shape,
            decoder_architecture=[256, 512],
            use_batch_norm=False,
            positive_margin=0.95,
            negative_margin=0.05,
            downweight=0.3,
            reconstruction_weight=0.005,
            name="custom_capsnet"
        )

        assert capsnet.num_classes == num_classes
        assert capsnet.routing_iterations == 5
        assert capsnet.conv_filters == [128, 256]
        assert capsnet.primary_capsules == 16
        assert capsnet.primary_capsule_dim == 4
        assert capsnet.digit_capsule_dim == 8
        assert capsnet.reconstruction is False
        assert capsnet._input_shape == mnist_input_shape
        assert capsnet.use_batch_norm is False
        assert capsnet.positive_margin == 0.95
        assert capsnet.negative_margin == 0.05
        assert capsnet.downweight == 0.3
        assert capsnet.reconstruction_weight == 0.005
        assert capsnet.name == "custom_capsnet"
        # Model layers are built only when first called, not during initialization
        assert capsnet._layers_built is False

    def test_initialization_with_regularization(self, num_classes):
        """Test initialization with regularization."""
        capsnet = CapsNet(
            num_classes=num_classes,
            kernel_regularizer="l2"
        )

        assert capsnet.kernel_regularizer is not None
        assert isinstance(capsnet.kernel_regularizer, keras.regularizers.L2)

    def test_parameter_validation(self):
        """Test parameter validation with invalid inputs."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            CapsNet(num_classes=0)

        with pytest.raises(ValueError, match="num_classes must be positive"):
            CapsNet(num_classes=-5)

        with pytest.raises(ValueError, match="routing_iterations must be positive"):
            CapsNet(num_classes=10, routing_iterations=0)

        with pytest.raises(ValueError, match="primary_capsules must be positive"):
            CapsNet(num_classes=10, primary_capsules=-1)

        with pytest.raises(ValueError, match="primary_capsule_dim must be positive"):
            CapsNet(num_classes=10, primary_capsule_dim=0)

        with pytest.raises(ValueError, match="digit_capsule_dim must be positive"):
            CapsNet(num_classes=10, digit_capsule_dim=-2)

    def test_build_process(self, num_classes, mnist_input_shape, sample_mnist_data):
        """Test model building process."""
        capsnet = CapsNet(num_classes=num_classes)

        # Build by calling with data
        outputs = capsnet(sample_mnist_data)

        assert capsnet._layers_built is True
        assert capsnet.built is True
        assert len(capsnet.conv_layers) == len(capsnet.conv_filters)
        assert capsnet.primary_caps is not None
        assert capsnet.digit_caps is not None
        assert isinstance(capsnet.primary_caps, PrimaryCapsule)
        assert isinstance(capsnet.digit_caps, RoutingCapsule)

    def test_build_with_reconstruction(self, num_classes, mnist_input_shape):
        """Test building with reconstruction enabled."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=True
        )

        # Build the model by calling it with sample data
        sample_data = tf.random.uniform([1] + list(mnist_input_shape), 0, 1)
        _ = capsnet(sample_data)

        assert capsnet.decoder is not None
        assert isinstance(capsnet.decoder, keras.Sequential)

    def test_build_without_reconstruction(self, num_classes, mnist_input_shape):
        """Test building without reconstruction."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=False
        )

        assert capsnet.decoder is None

    def test_forward_pass_without_reconstruction(self, num_classes, mnist_input_shape, sample_mnist_data):
        """Test forward pass without reconstruction."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=False
        )

        outputs = capsnet(sample_mnist_data)

        # Check output structure
        assert isinstance(outputs, dict)
        assert "digit_caps" in outputs
        assert "length" in outputs
        assert "reconstructed" not in outputs

        # Check output shapes
        batch_size = sample_mnist_data.shape[0]
        assert outputs["digit_caps"].shape == (batch_size, num_classes, capsnet.digit_capsule_dim)
        assert outputs["length"].shape == (batch_size, num_classes)

        # Check for valid values
        assert not np.any(np.isnan(outputs["digit_caps"].numpy()))
        assert not np.any(np.isnan(outputs["length"].numpy()))

        # Check length values are positive (they represent magnitudes)
        assert np.all(outputs["length"].numpy() >= 0)

    def test_forward_pass_with_reconstruction(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test forward pass with reconstruction."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=True
        )

        outputs = capsnet(sample_mnist_data, mask=sample_labels)

        # Check output structure
        assert isinstance(outputs, dict)
        assert "digit_caps" in outputs
        assert "length" in outputs
        assert "reconstructed" in outputs

        # Check reconstruction shape
        assert outputs["reconstructed"].shape == sample_mnist_data.shape

        # Check reconstruction values are in valid range (sigmoid output)
        recon_values = outputs["reconstructed"].numpy()
        assert np.all(recon_values >= 0)
        assert np.all(recon_values <= 1)

    def test_reconstruction_without_mask(self, num_classes, mnist_input_shape, sample_mnist_data):
        """Test reconstruction using predicted classes (no mask provided)."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=True
        )

        # Call without mask - should use predicted classes
        outputs = capsnet(sample_mnist_data)

        assert "reconstructed" in outputs
        assert outputs["reconstructed"].shape == sample_mnist_data.shape

    def test_different_input_shapes(self, num_classes):
        """Test with different input shapes."""
        test_shapes = [
            (28, 28, 1),  # MNIST
            (32, 32, 3),  # CIFAR-10
            (64, 64, 3),  # Larger images
        ]

        for shape in test_shapes:
            capsnet = CapsNet(
                num_classes=num_classes,
                input_shape=shape,
                reconstruction=False  # Faster testing
            )

            batch_data = tf.random.uniform([2] + list(shape), 0, 1)
            outputs = capsnet(batch_data)

            assert outputs["digit_caps"].shape == (2, num_classes, capsnet.digit_capsule_dim)
            assert outputs["length"].shape == (2, num_classes)

    def test_different_capsule_configurations(self, mnist_input_shape):
        """The capsule counts and dims must reach the parameterisation.

        These are structural knobs, so an output-difference check between the
        two configs would be satisfied by the different random draw alone. The
        weight-shape signature is the discriminating fact; the per-config shape
        assertions below are kept because they pin the *contract* (which the
        signature does not).
        """
        configs = [
            {"num_classes": 5, "primary_capsules": 16, "primary_capsule_dim": 4, "digit_capsule_dim": 8},
            {"num_classes": 20, "primary_capsules": 64, "primary_capsule_dim": 16, "digit_capsule_dim": 32},
        ]

        def _build(cfg):
            model = CapsNet(input_shape=mnist_input_shape, reconstruction=False, **cfg)
            model(tf.zeros([1] + list(mnist_input_shape)))
            return model

        builders = {
            i: (lambda cfg=cfg: _build(cfg)) for i, cfg in enumerate(configs)
        }
        assert_structural_knob_changes_weights(builders, knob="capsule configuration")

        for config in configs:
            capsnet = CapsNet(
                input_shape=mnist_input_shape,
                reconstruction=False,
                **config
            )

            batch_data = tf.random.uniform([2] + list(mnist_input_shape), 0, 1)
            outputs = capsnet(batch_data)

            assert outputs["digit_caps"].shape == (2, config["num_classes"], config["digit_capsule_dim"])
            assert outputs["length"].shape == (2, config["num_classes"])

    def test_different_conv_filters(self, num_classes, mnist_input_shape):
        """``conv_filters`` must build the requested conv stack.

        `length` has shape ``(2, num_classes)`` for every filter list, so the
        old shape assertion was true whether or not the list reached the graph.
        The stack depth is structural: one more entry means strictly more conv
        weight tensors.
        """
        filter_configs = [
            [128],
            [256, 128],
            [64, 128, 256],
        ]

        def _build(filters):
            model = CapsNet(
                num_classes=num_classes,
                input_shape=mnist_input_shape,
                conv_filters=filters,
                reconstruction=False,
            )
            model(tf.zeros([1] + list(mnist_input_shape)))
            return model

        builders = {
            tuple(f): (lambda f=f: _build(f)) for f in filter_configs
        }
        sigs = assert_structural_knob_changes_weights(builders, knob="conv_filters")
        for f in filter_configs:
            widths = [w[-1] for w in sigs[tuple(f)] if len(w) == 4]
            for requested in f:
                assert requested in widths, (
                    f"conv_filters={f} produced no conv with {requested} output "
                    f"channels; conv widths were {widths}"
                )

        for filters in filter_configs:
            capsnet = CapsNet(
                num_classes=num_classes,
                input_shape=mnist_input_shape,
                conv_filters=filters,
                reconstruction=False
            )

            batch_data = tf.random.uniform([2] + list(mnist_input_shape), 0, 1)
            outputs = capsnet(batch_data)

            assert outputs["length"].shape == (2, num_classes)

    def test_train_step(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test custom training step."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape
        )
        capsnet.compile(optimizer="adam", metrics=[CapsuleAccuracy()])

        # Test training step
        metrics = capsnet.train_step((sample_mnist_data, sample_labels))

        # Check returned metrics
        assert "loss" in metrics
        assert "margin_loss" in metrics
        assert "reconstruction_loss" in metrics
        # Metrics may be nested under 'compile_metrics'
        if "compile_metrics" in metrics:
            assert "capsule_accuracy" in metrics["compile_metrics"]
        else:
            assert "capsule_accuracy" in metrics

        # Check metric values are scalars and valid
        for metric_name, metric_value in metrics.items():
            if metric_name == "compile_metrics":
                # Check nested metrics
                for nested_name, nested_value in metric_value.items():
                    assert nested_value.shape == ()
                    assert not np.isnan(nested_value.numpy())
            else:
                assert metric_value.shape == ()
                assert not np.isnan(metric_value.numpy())
                if "loss" in metric_name:
                    assert metric_value.numpy() >= 0

    def test_test_step(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test custom test step."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape
        )
        capsnet.compile(optimizer="adam", metrics=[CapsuleAccuracy()])

        # Test evaluation step
        metrics = capsnet.test_step((sample_mnist_data, sample_labels))

        # Check returned metrics
        assert "loss" in metrics
        assert "margin_loss" in metrics
        assert "reconstruction_loss" in metrics
        # Metrics may be nested under 'compile_metrics'
        if "compile_metrics" in metrics:
            assert "capsule_accuracy" in metrics["compile_metrics"]
        else:
            assert "capsule_accuracy" in metrics

    def test_train_step_without_reconstruction(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test training step without reconstruction."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=False
        )
        capsnet.compile(optimizer="adam")

        metrics = capsnet.train_step((sample_mnist_data, sample_labels))

        # Reconstruction loss should be 0
        assert metrics["reconstruction_loss"].numpy() == 0.0

    def test_different_margin_parameters(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """The margin parameters must change the margin loss they parameterise.

        ``"margin_loss" in metrics`` is true for every margin setting, including
        one that never reaches the loss. These are value knobs -- they do not
        touch the parameterisation -- so two identically seeded models differ
        only in the margins, and the loss they report must differ.
        """
        margins = {
            "default": dict(positive_margin=0.9, negative_margin=0.1, downweight=0.5),
            "tight": dict(positive_margin=0.95, negative_margin=0.05, downweight=0.3),
        }

        def _loss(cfg):
            model = build_seeded(lambda: CapsNet(
                num_classes=num_classes,
                input_shape=mnist_input_shape,
                reconstruction=False,
                **cfg,
            ))
            model.compile(optimizer="adam")
            metrics = model.train_step((sample_mnist_data, sample_labels))
            assert "margin_loss" in metrics
            return float(as_array(metrics["margin_loss"]))

        losses = {k: _loss(cfg) for k, cfg in margins.items()}
        delta = abs(losses["default"] - losses["tight"])
        # Same seed, same parameterisation, same batch: the only difference is
        # the margins, so this delta is entirely attributable to them.
        assert delta > 1e-5, (
            f"margin parameters are a no-op: margin_loss was {losses} "
            f"(delta {delta:.3e})"
        )

    def test_serialization(self, num_classes, mnist_input_shape):
        """Test serialization and deserialization."""
        original_capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            routing_iterations=5,
            conv_filters=[128, 256],
            primary_capsules=16,
            primary_capsule_dim=4,
            digit_capsule_dim=8,
            reconstruction=False,
            use_batch_norm=False,
            positive_margin=0.95,
            negative_margin=0.05,
            downweight=0.3,
            reconstruction_weight=0.005
        )

        # Get configs
        config = original_capsnet.get_config()

        # Recreate the model
        recreated_capsnet = CapsNet.from_config(config)

        # Check configuration matches
        assert recreated_capsnet.num_classes == original_capsnet.num_classes
        assert recreated_capsnet.routing_iterations == original_capsnet.routing_iterations
        assert recreated_capsnet.conv_filters == original_capsnet.conv_filters
        assert recreated_capsnet.primary_capsules == original_capsnet.primary_capsules
        assert recreated_capsnet.primary_capsule_dim == original_capsnet.primary_capsule_dim
        assert recreated_capsnet.digit_capsule_dim == original_capsnet.digit_capsule_dim
        assert recreated_capsnet.reconstruction == original_capsnet.reconstruction
        assert recreated_capsnet.use_batch_norm == original_capsnet.use_batch_norm
        assert recreated_capsnet.positive_margin == original_capsnet.positive_margin
        assert recreated_capsnet.negative_margin == original_capsnet.negative_margin
        assert recreated_capsnet.downweight == original_capsnet.downweight
        assert recreated_capsnet.reconstruction_weight == original_capsnet.reconstruction_weight

    def test_model_save_load(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test saving and loading a CapsNet model.

        HISTORY -- this test carried `xfail(strict=True)` from 2026-08-18 until
        plan-2026-08-18T073231-52a93f8c/iter-1/step-8. The pinned defect: CapsNet
        lost EVERY kernel on a `.keras` round trip. Weight paths and shapes all
        matched (16 weights, identical paths) while the values did not --
        conv_1/kernel differed by 0.686, conv_2/kernel by 0.080,
        primary_caps/primary_conv/kernel by 0.045,
        digit_caps/capsule_transformation_weights by 0.022, i.e. the restored
        kernels were FRESH. Only the tensors whose trained value still equalled
        their initializer (biases, BN gamma/beta/moving stats) "matched", and
        only by coincidence. Cause: `CapsNet.build()` overrode `keras.Model.build`
        without building its sub-layer tree, which disables Keras'
        `is_default(self.build)` build-by-run fallback while `build_from_config`
        swallows the incomplete build in a bare `try/except` -- so the loss was
        silent. Fixed by `CapsNet._build_sublayer_tree` (see that method's
        `# DECISION .../D-006` anchor and decisions.md D-006).

        The pin was strict, so removing it is itself the red-proof. It is kept
        removed rather than deleted-and-forgotten because the original
        instrument -- SHAPE equality only -- was blind to this failure mode
        (reference_nested_sublayer_list_loses_weights.md). This test therefore
        now asserts the DISCRIMINATING quantity: the weight count BEFORE the
        first `call()` on the loaded model. A post-`call()` count is 16 either
        way, because lazy build re-creates fresh variables; only the pre-call
        count separates fixed from broken (measured: 0 before the fix, 16
        after). Per-weight VALUES are compared too, not just outputs.
        """
        # Create and compile model
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=False,  # Simpler for testing
            name="test_capsnet"
        )
        capsnet.compile(optimizer="adam", metrics=[CapsuleAccuracy()])

        # Train for one step to initialize all variables
        capsnet.train_step((sample_mnist_data, sample_labels))

        # Generate prediction before saving
        original_outputs = capsnet(sample_mnist_data, training=False)

        donor_weights = {w.path: keras.ops.convert_to_numpy(w) for w in capsnet.weights}
        assert len(donor_weights) == 16, (
            f"expected 16 donor weights for this configuration, got {len(donor_weights)}"
        )

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "capsnet_model.keras")

            # Save the model
            capsnet.save(model_path)

            # Load the model
            loaded_capsnet = keras.models.load_model(
                model_path,
                custom_objects={
                    "CapsNet": CapsNet,
                    "PrimaryCapsule": PrimaryCapsule,
                    "RoutingCapsule": RoutingCapsule,
                    "CapsuleAccuracy": CapsuleAccuracy,
                    "capsule_margin_loss": capsule_margin_loss,
                    "length": length
                }
            )

            # THE DISCRIMINATING MEASUREMENT, and it must run BEFORE the first
            # `call()` on the loaded model: a broken build lazily materializes
            # fresh variables on first call, so a post-call count is 16 either
            # way. Measured 0 before the step-8 fix, 16 after.
            assert len(loaded_capsnet.weights) == 16, (
                "loaded model has "
                f"{len(loaded_capsnet.weights)} weights before its first call, "
                "expected 16 -- the sub-layer tree was not built by "
                "`build_from_config`, so the saved arrays had nowhere to land"
            )

            # Per-weight VALUE equality. Shape equality was the ONLY check here,
            # and it is exactly the instrument that misses this repo's recorded
            # `List[List[Layer]]` round-trip failure, where weight count, layer
            # paths and parameter totals all matched while the restored kernels
            # were FRESH (reference_nested_sublayer_list_loses_weights.md).
            loaded_weights = {
                w.path: keras.ops.convert_to_numpy(w) for w in loaded_capsnet.weights
            }
            assert set(loaded_weights.keys()) == set(donor_weights.keys())
            for path, donor_value in donor_weights.items():
                np.testing.assert_allclose(
                    donor_value,
                    loaded_weights[path],
                    rtol=1e-6, atol=1e-7,
                    err_msg=f"weight '{path}' was not restored from the checkpoint",
                )

            # Generate prediction with loaded model
            loaded_outputs = loaded_capsnet(sample_mnist_data, training=False)

            assert set(original_outputs.keys()) == set(loaded_outputs.keys())
            for key in original_outputs.keys():
                np.testing.assert_allclose(
                    keras.ops.convert_to_numpy(original_outputs[key]),
                    keras.ops.convert_to_numpy(loaded_outputs[key]),
                    rtol=1e-5, atol=1e-6,
                    err_msg=f"'{key}' differs after a save/load round trip",
                )

    def test_model_save_load_with_reconstruction(
        self, num_classes, mnist_input_shape, sample_mnist_data
    ):
        """Round-trip the RECONSTRUCTION path, whose decoder was never probed.

        The decoder is a `keras.Sequential`, and `Sequential` normally builds its
        own sublayers by running them, so it was an open question whether it
        rescued itself. It does NOT, and this test can fail: measured
        2026-08-18 against a subclass that builds the whole tree EXCEPT the
        decoder, `loaded.weights` holds 16 of 22 before the first call and the
        six decoder tensors come back FRESH (decoder_hidden_1/kernel off by
        0.521, decoder_hidden_2/kernel by 0.302, decoder_output/kernel by
        0.219, the three biases by exactly the perturbation), which moves
        `reconstructed` by 0.582 while `digit_caps` and `length` stay at exactly
        0.0 -- so `test_model_save_load`, which runs `reconstruction=False`,
        cannot see this at all. The decoder's explicit build in
        `CapsNet._build_sublayer_tree` is therefore load-bearing, not
        belt-and-braces.
        """
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=True,
            name="recon_capsnet",
        )

        # Build, then move every weight off its initializer. A restored tensor
        # that still equals its initializer is indistinguishable from one that
        # was never loaded, which is how the biases "matched" under the defect.
        _ = capsnet(sample_mnist_data, training=False)
        for weight in capsnet.weights:
            weight.assign(weight + 0.02)

        donor_weights = {w.path: keras.ops.convert_to_numpy(w) for w in capsnet.weights}
        assert len(donor_weights) == 22, (
            f"expected 22 donor weights with reconstruction on, got {len(donor_weights)}"
        )

        original_outputs = capsnet(sample_mnist_data, training=False)
        assert "reconstructed" in original_outputs

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "capsnet_reconstruction.keras")
            capsnet.save(model_path)

            loaded_capsnet = keras.models.load_model(
                model_path,
                custom_objects={
                    "CapsNet": CapsNet,
                    "PrimaryCapsule": PrimaryCapsule,
                    "RoutingCapsule": RoutingCapsule,
                    "CapsuleAccuracy": CapsuleAccuracy,
                    "capsule_margin_loss": capsule_margin_loss,
                    "length": length,
                },
            )

            # BEFORE the first call: the discriminating count. 16 (not 22) is
            # the signature of an unbuilt decoder specifically.
            assert len(loaded_capsnet.weights) == 22, (
                f"loaded model has {len(loaded_capsnet.weights)} weights before "
                "its first call, expected 22 -- 16 means the decoder Sequential "
                "was not built, so its six tensors had nowhere to land"
            )

            loaded_weights = {
                w.path: keras.ops.convert_to_numpy(w) for w in loaded_capsnet.weights
            }
            assert set(loaded_weights.keys()) == set(donor_weights.keys())
            for path, donor_value in donor_weights.items():
                np.testing.assert_allclose(
                    donor_value,
                    loaded_weights[path],
                    rtol=1e-6, atol=1e-7,
                    err_msg=f"weight '{path}' was not restored from the checkpoint",
                )

            loaded_outputs = loaded_capsnet(sample_mnist_data, training=False)

            assert set(original_outputs.keys()) == set(loaded_outputs.keys())
            for key in original_outputs.keys():
                np.testing.assert_allclose(
                    keras.ops.convert_to_numpy(original_outputs[key]),
                    keras.ops.convert_to_numpy(loaded_outputs[key]),
                    rtol=1e-5, atol=1e-6,
                    err_msg=f"'{key}' differs after a save/load round trip",
                )

    def test_save_load_reconstruction_without_input_shape(
        self, num_classes, sample_mnist_data
    ):
        """The D-007 deviation branch: decoder created in ``build()``.

        Every other ``reconstruction=True`` call site in this suite passes
        ``input_shape=`` to ``__init__``, so the decoder is created eagerly
        there and the ANCHORED deviation -- the one branch where creation
        genuinely cannot happen in ``__init__``, because the decoder's output
        width is ``prod(input_shape[1:])`` and nothing knows it yet -- had no
        coverage at all.

        What this pins, in order: the decoder really is ``None`` after
        ``__init__`` (otherwise the branch is not being exercised);
        ``get_config()`` reports the shape captured during ``build()`` rather
        than ``None`` (otherwise the reload constructs a different model); the
        loaded model carries all 22 weights BEFORE its first ``call()`` (16
        would mean the decoder Sequential was never built); and all three
        outputs match exactly.
        """
        capsnet = CapsNet(
            num_classes=num_classes,
            reconstruction=True,
            name="recon_no_shape_capsnet",
        )
        assert capsnet.decoder is None, (
            "the decoder must NOT exist after __init__ when input_shape was "
            "not supplied -- otherwise this test is not exercising the D-007 "
            "deviation branch at all"
        )

        _ = capsnet(sample_mnist_data, training=False)
        assert capsnet.decoder is not None, "build() must have created it"

        config = capsnet.get_config()
        assert tuple(config["input_shape"]) == tuple(sample_mnist_data.shape[1:]), (
            f"get_config must report the shape captured in build(), got "
            f"{config['input_shape']!r}"
        )

        for weight in capsnet.weights:
            weight.assign(weight + 0.02)

        donor_weights = {
            w.path: keras.ops.convert_to_numpy(w) for w in capsnet.weights
        }
        assert len(donor_weights) == 22

        original_outputs = capsnet(sample_mnist_data, training=False)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "capsnet_recon_no_shape.keras")
            capsnet.save(model_path)

            loaded_capsnet = keras.models.load_model(
                model_path,
                custom_objects={
                    "CapsNet": CapsNet,
                    "PrimaryCapsule": PrimaryCapsule,
                    "RoutingCapsule": RoutingCapsule,
                    "CapsuleAccuracy": CapsuleAccuracy,
                    "capsule_margin_loss": capsule_margin_loss,
                    "length": length,
                },
            )

            assert len(loaded_capsnet.weights) == 22, (
                f"loaded model has {len(loaded_capsnet.weights)} weights before "
                "its first call, expected 22"
            )

            loaded_weights = {
                w.path: keras.ops.convert_to_numpy(w)
                for w in loaded_capsnet.weights
            }
            assert set(loaded_weights.keys()) == set(donor_weights.keys())
            for path, donor_value in donor_weights.items():
                np.testing.assert_allclose(
                    donor_value,
                    loaded_weights[path],
                    rtol=1e-6, atol=1e-7,
                    err_msg=f"weight '{path}' was not restored",
                )

            loaded_outputs = loaded_capsnet(sample_mnist_data, training=False)
            assert set(original_outputs.keys()) == set(loaded_outputs.keys())
            for key in original_outputs.keys():
                np.testing.assert_allclose(
                    keras.ops.convert_to_numpy(original_outputs[key]),
                    keras.ops.convert_to_numpy(loaded_outputs[key]),
                    rtol=1e-5, atol=1e-6,
                    err_msg=f"'{key}' differs after a save/load round trip",
                )

    def test_training_integration(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test training integration with fit() method."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            conv_filters=[64, 128],  # Smaller for faster testing
            reconstruction=False
        )
        capsnet.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            metrics=[CapsuleAccuracy()]
        )

        # Create dataset with repeated data
        expanded_data = tf.tile(sample_mnist_data, [2, 1, 1, 1])  # 8 samples
        expanded_labels = tf.tile(sample_labels, [2, 1])  # 8 labels

        dataset = tf.data.Dataset.from_tensor_slices((expanded_data, expanded_labels))
        dataset = dataset.batch(4)

        # Train for a few steps
        history = capsnet.fit(dataset, epochs=2, verbose=0)

        # Check that training metrics are recorded
        assert "loss" in history.history
        assert "margin_loss" in history.history
        assert "reconstruction_loss" in history.history
        assert "capsule_accuracy" in history.history
        assert len(history.history["loss"]) == 2  # 2 epochs

    def test_gradient_flow(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test gradient flow through the model."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=False
        )
        capsnet.compile(optimizer="adam")

        with tf.GradientTape() as tape:
            outputs = capsnet(sample_mnist_data, training=True)
            # Compute margin loss manually
            margin_loss = tf.reduce_mean(capsule_margin_loss(
                outputs["length"],
                sample_labels,
                capsnet.downweight,
                capsnet.positive_margin,
                capsnet.negative_margin
            ))

        # Get gradients
        grads = tape.gradient(margin_loss, capsnet.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads)

        # Check some gradients have non-zero values
        has_nonzero_grad = any(np.any(g.numpy() != 0) for g in grads if g is not None)
        assert has_nonzero_grad

    def test_create_capsnet_factory(self, num_classes, mnist_input_shape):
        """Test the create_capsnet factory function."""
        capsnet = create_capsnet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            optimizer="adam",
            learning_rate=0.001
        )

        assert isinstance(capsnet, CapsNet)
        assert capsnet.num_classes == num_classes
        assert capsnet._input_shape == mnist_input_shape
        # Build the model by calling it
        sample_data = tf.random.uniform([1] + list(mnist_input_shape), 0, 1)
        _ = capsnet(sample_data)
        assert capsnet.built is True
        assert capsnet.optimizer is not None
        assert len(capsnet.metrics) > 0  # Should have CapsuleAccuracy

    def test_create_capsnet_with_custom_optimizer(self, num_classes, mnist_input_shape):
        """Test create_capsnet with custom optimizer."""
        custom_optimizer = keras.optimizers.Adam(learning_rate=0.002)

        capsnet = create_capsnet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            optimizer=custom_optimizer
        )

        assert capsnet.optimizer == custom_optimizer

    def test_numerical_stability(self, num_classes):
        """Test model stability with extreme input values."""
        # Use larger input size to avoid negative output dimensions
        input_shape = (64, 64, 1)
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=input_shape,
            reconstruction=False
        )

        test_cases = [
            tf.zeros((2,) + input_shape),  # All zeros
            tf.ones((2,) + input_shape),   # All ones
            tf.random.uniform((2,) + input_shape, 0, 1e-6),  # Very small values
            tf.random.uniform((2,) + input_shape, 1-1e-6, 1),  # Very close to 1
        ]

        for i, test_input in enumerate(test_cases):
            outputs = capsnet(test_input)

            # Check for NaN/Inf values
            for key, tensor in outputs.items():
                assert not np.any(np.isnan(tensor.numpy())), f"NaN in {key} for test case {i}"
                assert not np.any(np.isinf(tensor.numpy())), f"Inf in {key} for test case {i}"

    def test_batch_size_independence(self, num_classes, mnist_input_shape):
        """Test that model works with different batch sizes."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=False
        )

        batch_sizes = [1, 2, 8, 16]

        for batch_size in batch_sizes:
            test_data = tf.random.uniform((batch_size,) + mnist_input_shape, 0, 1)
            outputs = capsnet(test_data)

            assert outputs["digit_caps"].shape[0] == batch_size
            assert outputs["length"].shape[0] == batch_size

    def test_routing_iterations(self, num_classes, mnist_input_shape, sample_mnist_data):
        """Test with different routing iterations."""
        routing_iterations = [1, 3, 5, 10]

        for iterations in routing_iterations:
            capsnet = CapsNet(
                num_classes=num_classes,
                input_shape=mnist_input_shape,
                routing_iterations=iterations,
                reconstruction=False
            )

            outputs = capsnet(sample_mnist_data)

            # Should work with any number of iterations
            assert outputs["length"].shape == (sample_mnist_data.shape[0], num_classes)

    def test_without_batch_norm(self, num_classes, mnist_input_shape, sample_mnist_data):
        """Test model without batch normalization."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            use_batch_norm=False,
            reconstruction=False
        )

        outputs = capsnet(sample_mnist_data)
        assert outputs["length"].shape == (sample_mnist_data.shape[0], num_classes)

    def test_reconstruction_weight_effect(self, num_classes, mnist_input_shape, sample_mnist_data, sample_labels):
        """Test effect of different reconstruction weights."""
        weights = [0.0, 0.001, 0.01, 0.1]

        for weight in weights:
            capsnet = CapsNet(
                num_classes=num_classes,
                input_shape=mnist_input_shape,
                reconstruction=True,
                reconstruction_weight=weight
            )
            capsnet.compile(optimizer="adam")

            metrics = capsnet.train_step((sample_mnist_data, sample_labels))

            # Total loss should include reconstruction component proportional to weight
            assert "loss" in metrics
            assert "reconstruction_loss" in metrics

    def test_invalid_input_shape_error(self, num_classes):
        """Test that invalid input shapes raise appropriate errors."""
        capsnet = CapsNet(num_classes=num_classes)

        # Test with wrong number of dimensions
        invalid_input = tf.random.uniform([4, 28, 28])  # Missing channel dimension

        with pytest.raises(ValueError, match="Expected 4D input"):
            capsnet(invalid_input)

    def test_model_summary(self, num_classes, mnist_input_shape):
        """Test model summary functionality."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape
        )

        # Should not raise an error
        capsnet.summary()

    def test_save_model_method(self, num_classes, mnist_input_shape):
        """The archive must contain the WEIGHTS, not just exist.

        Before D-053 this test asserted only ``os.path.exists`` and passed
        against a 31,587-byte archive holding zero weights.
        """
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "test_model.keras")

            # `save_model` builds an unbuilt model from its own `input_shape`.
            capsnet.save_model(model_path)

            assert os.path.exists(model_path)
            assert capsnet.built, "save_model must leave the model built"
            assert len(capsnet.trainable_weights) > 0

    def test_saving_an_unbuildable_capsnet_is_refused(self, num_classes):
        """No `input_shape` anywhere means no weights to save -- say so."""
        capsnet = CapsNet(num_classes=num_classes, reconstruction=False)
        assert not capsnet.built

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "empty.keras")
            with pytest.raises(ValueError, match="unbuilt CapsNet"):
                capsnet.save_model(model_path)
            assert not os.path.exists(model_path), (
                "an empty archive was written before the refusal"
            )

    def test_load_model_method(self, num_classes, mnist_input_shape):
        """The round trip must carry the weight VALUES, not just the class."""
        capsnet = CapsNet(
            num_classes=num_classes,
            input_shape=mnist_input_shape,
            reconstruction=False
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "test_model.keras")

            # Save model
            capsnet.save_model(model_path)

            # Load model using class method
            loaded_capsnet = CapsNet.load_model(model_path)

            assert isinstance(loaded_capsnet, CapsNet)
            assert loaded_capsnet.num_classes == num_classes

            original = capsnet.get_weights()
            restored = loaded_capsnet.get_weights()
            assert len(original) > 0, "the source model held no weights"
            assert len(restored) == len(original), (
                f"round trip restored {len(restored)} weight arrays, "
                f"saved {len(original)}"
            )
            for index, (before, after) in enumerate(zip(original, restored)):
                np.testing.assert_allclose(
                    before, after, rtol=0, atol=0,
                    err_msg=f"weight array {index} changed across the round trip"
                )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])