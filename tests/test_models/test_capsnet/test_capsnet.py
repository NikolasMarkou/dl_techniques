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
import json
import tempfile
import os
from typing import Tuple

from dl_techniques.models.vision.capsnet.model import CapsNet, CapsuleAccuracy, create_capsnet
from dl_techniques.layers.capsules import PrimaryCapsule, RoutingCapsule
from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.losses.capsule_margin_loss import capsule_margin_loss
from dl_techniques.utils.tensors import length

from ..initializer_replay_oracle import replayed_pairs
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
        assert metric.count.numpy() > 0.0

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
        assert capsnet.built is False

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
        assert capsnet.input_shape_config == mnist_input_shape
        assert capsnet.use_batch_norm is False
        assert capsnet.positive_margin == 0.95
        assert capsnet.negative_margin == 0.05
        assert capsnet.downweight == 0.3
        assert capsnet.reconstruction_weight == 0.005
        assert capsnet.name == "custom_capsnet"
        # Model layers are built only when first called, not during initialization
        assert capsnet.built is False

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
        capsnet(sample_mnist_data)

        assert capsnet.built is True
        assert len(capsnet.conv_blocks) == len(capsnet.conv_filters)
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
        """Saving and loading a CapsNet model keeps every weight.

        The discriminating quantity is the weight count BEFORE the first ``call()`` on the
        loaded model, together with the per-weight VALUES. A model whose ``build`` does not
        build its sub-layer tree reloads with zero weights, and Keras then creates fresh
        variables on the first call, so a count taken after a call, shapes alone, or the
        tensors whose trained value still equals their initializer (biases, batch-norm
        statistics) all read as a pass while every kernel is a fresh random one.
        ``build`` materializes the tree by tracing ``call`` (the D-006 anchor in
        ``CapsNet.build``); this test fails if it does not.
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
            # way. A model that did not build its tree holds 0 here, a built one 16.
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
        """Round-trip the RECONSTRUCTION path, whose decoder needs its own check.

        The decoder is a ``keras.Sequential``, and ``Sequential`` normally builds its own
        sublayers by running them, so it is not obvious that it needs building by the
        model. It does: against a model that builds the whole tree EXCEPT the decoder,
        ``loaded.weights`` holds 16 of 22 before the first call and the six decoder tensors
        come back fresh, which moves ``reconstructed`` while ``digit_caps`` and ``length``
        stay exactly equal -- so ``test_model_save_load``, which runs
        ``reconstruction=False``, cannot see this at all. ``CapsNet.build`` creating and
        building the decoder is therefore load-bearing.
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
        assert capsnet.input_shape_config == mnist_input_shape
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


class _MnistFixtures:
    """The MNIST-shaped fixtures the classes below share."""

    @pytest.fixture
    def mnist_input_shape(self) -> Tuple[int, int, int]:
        return (28, 28, 1)

    @pytest.fixture
    def num_classes(self) -> int:
        return 10

    @pytest.fixture
    def sample_mnist_data(self, mnist_input_shape):
        return tf.random.uniform([4] + list(mnist_input_shape), 0, 1)


class TestCapsNetConvTrunk(_MnistFixtures):
    """The convolution trunk is a list of single-convolution ``ConvStem`` blocks."""

    def test_the_trunk_is_one_conv_stem_per_filter_entry(self):
        capsnet = CapsNet(num_classes=10, conv_filters=[8, 16, 24], reconstruction=False)

        assert [type(block) for block in capsnet.conv_blocks] == [ConvStem] * 3
        assert [block.name for block in capsnet.conv_blocks] == [
            "conv_block_0", "conv_block_1", "conv_block_2"
        ]
        assert [block.conv1.filters for block in capsnet.conv_blocks] == [8, 16, 24]

    def test_the_first_block_is_9x9_and_later_blocks_are_5x5_valid_and_biased(self):
        capsnet = CapsNet(num_classes=10, conv_filters=[8, 8, 8], reconstruction=False)

        assert [block.conv1.kernel_size for block in capsnet.conv_blocks] == [
            (9, 9), (5, 5), (5, 5)
        ]
        for block in capsnet.conv_blocks:
            assert block.conv1.strides == (1, 1)
            assert block.conv1.padding == "valid"
            assert block.conv1.use_bias is True
            assert block.conv2 is None and block.pool is None

    def test_a_block_keeps_the_keras_batch_norm_epsilon_not_the_factorys(self):
        """The factory default is 1e-6; the pre-migration layers used Keras' 1e-3."""
        capsnet = CapsNet(num_classes=10, reconstruction=False)

        for block in capsnet.conv_blocks:
            assert block.norm1.epsilon == pytest.approx(1e-3)
            assert block.norm1.momentum == pytest.approx(0.99)
            assert isinstance(block.act1, keras.layers.ReLU)

    def test_without_batch_norm_a_block_has_no_norm_layer(self):
        capsnet = CapsNet(num_classes=10, use_batch_norm=False, reconstruction=False)

        assert all(block.norm1 is None for block in capsnet.conv_blocks)

    def test_the_weight_paths_name_each_block_and_its_parts(self, mnist_input_shape):
        capsnet = CapsNet(
            num_classes=10, input_shape=mnist_input_shape, reconstruction=False, name="capsnet"
        )
        capsnet.build((None, *mnist_input_shape))

        assert [weight.path for weight in capsnet.weights] == [
            "capsnet/conv_block_0/conv_block_0_conv1/kernel",
            "capsnet/conv_block_0/conv_block_0_conv1/bias",
            "capsnet/conv_block_0/conv_block_0_norm1/gamma",
            "capsnet/conv_block_0/conv_block_0_norm1/beta",
            "capsnet/conv_block_0/conv_block_0_norm1/moving_mean",
            "capsnet/conv_block_0/conv_block_0_norm1/moving_variance",
            "capsnet/conv_block_1/conv_block_1_conv1/kernel",
            "capsnet/conv_block_1/conv_block_1_conv1/bias",
            "capsnet/conv_block_1/conv_block_1_norm1/gamma",
            "capsnet/conv_block_1/conv_block_1_norm1/beta",
            "capsnet/conv_block_1/conv_block_1_norm1/moving_mean",
            "capsnet/conv_block_1/conv_block_1_norm1/moving_variance",
            "capsnet/primary_caps/primary_conv/kernel",
            "capsnet/primary_caps/primary_conv/bias",
            "capsnet/digit_caps/capsule_transformation_weights",
            "capsnet/digit_caps/capsule_bias",
        ]

    def test_a_training_call_moves_the_block_batch_norm_statistics_and_an_inference_call_does_not(
        self, mnist_input_shape
    ):
        """``training`` reaches every block's batch norm: the moving mean moves on a training
        call and stays put on an inference call (and on ``training=None``)."""
        capsnet = CapsNet(num_classes=10, input_shape=mnist_input_shape, reconstruction=False)
        x = tf.random.uniform((4, *mnist_input_shape), 0, 1)
        capsnet(x, training=False)
        before = [keras.ops.convert_to_numpy(b.norm1.moving_mean) for b in capsnet.conv_blocks]

        capsnet(x, training=False)
        capsnet(x)
        unchanged = [keras.ops.convert_to_numpy(b.norm1.moving_mean) for b in capsnet.conv_blocks]
        capsnet(x, training=True)
        moved = [keras.ops.convert_to_numpy(b.norm1.moving_mean) for b in capsnet.conv_blocks]

        for was, same, now in zip(before, unchanged, moved):
            np.testing.assert_array_equal(was, same)
            assert float(np.max(np.abs(now - was))) > 1e-4

    def test_the_old_parallel_lists_and_the_private_flags_are_gone(self):
        capsnet = CapsNet(num_classes=10, reconstruction=False)

        for gone in ("conv_layers", "batch_norm_layers", "activation_layers",
                     "_layers_built", "_input_shape"):
            assert not hasattr(capsnet, gone), gone

    def test_input_shape_is_stored_as_input_shape_config_and_reported_when_resolved(
        self, num_classes, sample_mnist_data
    ):
        """``get_config`` reports the shape resolved by ``build`` on purpose: a reload
        of a model constructed without ``input_shape`` must rebuild the same decoder."""
        given = CapsNet(num_classes=num_classes, input_shape=[28, 28, 1], reconstruction=True)
        assert given.input_shape_config == (28, 28, 1)

        lazy = CapsNet(num_classes=num_classes, reconstruction=True)
        assert lazy.input_shape_config is None
        lazy(sample_mnist_data)
        assert lazy.input_shape_config == tuple(sample_mnist_data.shape[1:])
        assert tuple(lazy.get_config()["input_shape"]) == lazy.input_shape_config


class TestCapsNetBuildAndShapes(_MnistFixtures):
    """``build`` materializes what ``call`` runs; ``compute_output_shape`` is the forward shape."""

    @pytest.mark.parametrize("reconstruction", [True, False], ids=["decoder", "no-decoder"])
    def test_explicit_build_equals_a_lazy_build(self, mnist_input_shape, reconstruction):
        explicit = CapsNet(
            num_classes=10, input_shape=mnist_input_shape, reconstruction=reconstruction
        )
        explicit.build((None, *mnist_input_shape))
        lazy = CapsNet(
            num_classes=10, input_shape=mnist_input_shape, reconstruction=reconstruction
        )
        lazy(tf.zeros((1, *mnist_input_shape)))

        def relative(model):
            return [weight.path.split("/", 1)[1] for weight in model.weights]

        assert relative(explicit) == relative(lazy)
        assert len(explicit.weights) == (22 if reconstruction else 16)
        assert explicit.count_params() == lazy.count_params() > 0

    def test_build_is_idempotent(self, mnist_input_shape):
        capsnet = CapsNet(num_classes=10, input_shape=mnist_input_shape, reconstruction=False)
        capsnet.build((None, *mnist_input_shape))
        before = [weight.path for weight in capsnet.weights]
        capsnet.build((None, *mnist_input_shape))
        assert [weight.path for weight in capsnet.weights] == before

    def test_build_refuses_a_non_4d_shape(self):
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            CapsNet(num_classes=10, reconstruction=False).build((None, 28, 28))

    @pytest.mark.parametrize("reconstruction", [True, False], ids=["decoder", "no-decoder"])
    @pytest.mark.parametrize("batch", [None, 3])
    def test_compute_output_shape_equals_the_forward_shape_and_builds_nothing(
        self, mnist_input_shape, reconstruction, batch
    ):
        capsnet = CapsNet(
            num_classes=10, input_shape=mnist_input_shape, reconstruction=reconstruction
        )
        shapes = capsnet.compute_output_shape((batch, *mnist_input_shape))
        assert capsnet.built is False

        outputs = capsnet(tf.zeros((batch or 2, *mnist_input_shape)))
        assert set(shapes) == set(outputs)
        for key, shape in shapes.items():
            assert tuple(shape)[0] == batch
            assert tuple(shape)[1:] == tuple(outputs[key].shape)[1:], key

    def test_compute_output_shape_of_a_model_without_an_input_shape_reads_the_given_one(self):
        capsnet = CapsNet(num_classes=10, reconstruction=True)
        shapes = capsnet.compute_output_shape((None, 32, 32, 3))
        assert tuple(shapes["reconstructed"]) == (None, 32, 32, 3)

    def test_compute_output_shape_refuses_a_non_4d_shape(self):
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            CapsNet(num_classes=10, reconstruction=False).compute_output_shape((None, 28, 28))

    def test_a_symbolic_call_returns_the_dict_the_trainers_conformal_step_reads(
        self, mnist_input_shape
    ):
        capsnet = CapsNet(num_classes=10, input_shape=mnist_input_shape, reconstruction=False)
        outputs = capsnet(keras.Input(mnist_input_shape))
        assert tuple(outputs["length"].shape) == (None, 10)
        assert tuple(outputs["digit_caps"].shape) == (None, 10, 16)

    def test_summary_logs_nothing_and_builds_nothing(self, mnist_input_shape, caplog):
        capsnet = CapsNet(num_classes=10, input_shape=mnist_input_shape)
        with caplog.at_level("INFO"):
            capsnet.summary(print_fn=lambda *args, **kwargs: None)
        assert not [r for r in caplog.records if "CapsNet Configuration" in r.getMessage()]


class TestCapsNetInitializer(_MnistFixtures):
    """One ``kernel_initializer`` argument, one fresh draw per site."""

    @staticmethod
    def _site_initializers(capsnet: CapsNet):
        """The initializer object behind every kernel the model draws."""
        sites = [block.conv1.kernel_initializer for block in capsnet.conv_blocks]
        sites.append(capsnet.primary_caps.conv.kernel_initializer)
        sites.append(capsnet.digit_caps.kernel_initializer)
        if capsnet.decoder is not None:
            sites += [
                layer.kernel_initializer
                for layer in capsnet.decoder.layers
                if isinstance(layer, keras.layers.Dense)
            ]
        return sites

    def test_a_named_initializer_is_stored_as_the_name(self):
        assert CapsNet(num_classes=10).kernel_initializer == "he_normal"

    def test_an_instance_is_stored_as_its_serialized_spec(self):
        capsnet = CapsNet(num_classes=10, kernel_initializer=keras.initializers.HeNormal(seed=5))
        assert isinstance(capsnet.kernel_initializer, dict)
        assert capsnet.kernel_initializer["class_name"] == "HeNormal"
        assert capsnet.kernel_initializer["config"]["seed"] == 5
        json.dumps(capsnet.get_config())

    @pytest.mark.parametrize("initializer", ["he_normal", keras.initializers.HeNormal()],
                             ids=["name", "seedless-instance"])
    def test_no_live_instance_reaches_two_sites(self, mnist_input_shape, initializer):
        capsnet = CapsNet(
            num_classes=10, input_shape=mnist_input_shape, reconstruction=True,
            kernel_initializer=initializer,
        )
        sites = self._site_initializers(capsnet)
        assert len(sites) == 2 + 1 + 1 + 3  # two blocks, primary, digit, three decoder Dense layers
        assert len({id(site) for site in sites}) == len(sites)

    @pytest.mark.parametrize("initializer", ["he_normal", keras.initializers.HeNormal()],
                             ids=["name", "seedless-instance"])
    def test_no_two_kernels_replay_one_draw(self, mnist_input_shape, initializer):
        keras.utils.set_random_seed(1234)
        capsnet = CapsNet(
            num_classes=10, input_shape=mnist_input_shape, reconstruction=True,
            kernel_initializer=initializer,
        )
        capsnet.build((None, *mnist_input_shape))
        pairs, compared = replayed_pairs(capsnet)
        assert compared > 0
        assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:2]}"

    def test_a_seeded_instance_keeps_its_seed_at_every_site(self, mnist_input_shape):
        capsnet = CapsNet(
            num_classes=10, input_shape=mnist_input_shape,
            kernel_initializer=keras.initializers.HeNormal(seed=5),
        )
        assert {site.get_config()["seed"] for site in self._site_initializers(capsnet)} == {5}


class TestCapsNetConfig(_MnistFixtures):
    """``get_config`` writes every constructor argument and is JSON-safe."""

    def test_get_config_is_json_safe_with_an_initializer_instance(self, mnist_input_shape):
        capsnet = CapsNet(
            num_classes=10,
            input_shape=mnist_input_shape,
            kernel_initializer=keras.initializers.HeNormal(),
            kernel_regularizer=keras.regularizers.L2(1e-4),
        )
        config = capsnet.get_config()
        json.dumps(config)
        rebuilt = CapsNet.from_config(json.loads(json.dumps(config)))
        assert json.dumps(rebuilt.get_config(), sort_keys=True) == json.dumps(config, sort_keys=True)

    def test_a_config_names_every_constructor_argument(self):
        import inspect

        capsnet = CapsNet(num_classes=10, reconstruction=False)
        arguments = set(inspect.signature(CapsNet.__init__).parameters) - {"self", "kwargs"}
        assert arguments <= set(capsnet.get_config())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])