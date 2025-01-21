"""Segmentation Loss Functions Module.

This module implements various loss functions for semantic segmentation tasks,
based on the paper "Loss Functions in the Era of Semantic Segmentation: A Survey and Outlook".

The implementation includes pixel-level, region-level, boundary-level, and combination losses,
with proper type hints and error handling.

Key Features:
    - Type-safe implementations with proper error checking
    - Configurable parameters for each loss function
    - Comprehensive documentation and examples
    - Memory-efficient tensor operations
"""


import tensorflow as tf
from dataclasses import dataclass
from typing import Optional, Union, Tuple

@dataclass
class LossConfig:
    """Configuration for loss function parameters.

    Args:
        num_classes: Number of segmentation classes
        smooth_factor: Smoothing factor to prevent division by zero
        focal_gamma: Focusing parameter for focal loss
        focal_alpha: Balancing parameter for focal loss
        tversky_alpha: False positive weight for Tversky loss
        tversky_beta: False negative weight for Tversky loss
        focal_tversky_gamma: Focal parameter for Focal Tversky loss
        combo_alpha: Weight for Dice component in Combo loss
        combo_beta: Weight for CE component in Combo loss
        boundary_theta: Distance threshold for boundary loss
    """
    num_classes: int
    smooth_factor: float = 1e-6
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    focal_tversky_gamma: float = 1.3
    combo_alpha: float = 0.5
    combo_beta: float = 0.5
    boundary_theta: float = 1.5


class SegmentationLosses:
    """Implementation of various segmentation loss functions.

    This class provides implementations of multiple loss functions commonly used
    in semantic segmentation tasks, including Cross-Entropy, Dice, Focal,
    Tversky, and various combinations.

    Args:
        config: LossConfig instance containing loss function parameters

    Raises:
        ValueError: If invalid configuration parameters are provided
    """

    def __init__(self, config: LossConfig) -> None:
        self._validate_config(config)
        self.config = config

    def _validate_config(self, config: LossConfig) -> None:
        """Validates the configuration parameters.

        Args:
            config: LossConfig instance to validate

        Raises:
            ValueError: If any parameter is invalid
        """
        if config.num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {config.num_classes}")
        if config.smooth_factor <= 0:
            raise ValueError(f"smooth_factor must be positive, got {config.smooth_factor}")
        if config.focal_gamma < 0:
            raise ValueError(f"focal_gamma must be non-negative, got {config.focal_gamma}")
        if not 0 <= config.focal_alpha <= 1:
            raise ValueError(f"focal_alpha must be in [0, 1], got {config.focal_alpha}")

    @staticmethod
    def _validate_inputs(
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            weights: Optional[tf.Tensor] = None
    ) -> None:
        """Validates input tensors.

        Args:
            y_true: Ground truth tensor
            y_pred: Prediction tensor
            weights: Optional weight tensor

        Raises:
            ValueError: If inputs have invalid shapes or types
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true shape {y_true.shape} != "
                f"y_pred shape {y_pred.shape}"
            )
        if weights is not None and weights.shape[-1] != y_true.shape[-1]:
            raise ValueError(
                f"Weight dimension mismatch: weights shape {weights.shape} != "
                f"num_classes {y_true.shape[-1]}"
            )

    def cross_entropy_loss(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            weights: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Implements weighted cross-entropy loss.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)
            weights: Optional class weights (num_classes,)

        Returns:
            Weighted cross-entropy loss

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        self._validate_inputs(y_true, y_pred, weights)
        y_true = tf.cast(y_true, tf.float32)

        # Add epsilon to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

        if weights is not None:
            weights = tf.cast(weights, tf.float32)
            ce_loss *= tf.reduce_sum(y_true * weights, axis=-1)

        return tf.reduce_mean(ce_loss)

    def dice_loss(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """Implements Dice loss.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Dice loss

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        self._validate_inputs(y_true, y_pred)
        y_true = tf.cast(y_true, tf.float32)

        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        denominator = (
                tf.reduce_sum(y_true, axis=[1, 2]) +
                tf.reduce_sum(y_pred, axis=[1, 2])
        )

        dice_coef = (numerator + self.config.smooth_factor) / (
                denominator + self.config.smooth_factor
        )
        return 1.0 - tf.reduce_mean(dice_coef)

    def focal_loss(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """Implements Focal loss.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Focal loss

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        self._validate_inputs(y_true, y_pred)
        y_true = tf.cast(y_true, tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_factor = tf.pow(1. - y_pred, self.config.focal_gamma)

        focal_ce = self.config.focal_alpha * focal_factor * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal_ce, axis=-1))

    def tversky_loss(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """Implements Tversky loss.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Tversky loss

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        self._validate_inputs(y_true, y_pred)
        y_true = tf.cast(y_true, tf.float32)

        numerator = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        false_positives = tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2])
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2])

        denominator = (
                numerator +
                self.config.tversky_alpha * false_positives +
                self.config.tversky_beta * false_negatives
        )

        tversky_coef = (numerator + self.config.smooth_factor) / (
                denominator + self.config.smooth_factor
        )
        return 1.0 - tf.reduce_mean(tversky_coef)

    def focal_tversky_loss(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """Implements Focal Tversky loss.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Focal Tversky loss

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        tversky_coef = 1.0 - self.tversky_loss(y_true, y_pred)
        focal_tversky = tf.pow(1.0 - tversky_coef, self.config.focal_tversky_gamma)
        return tf.reduce_mean(focal_tversky)

    def lovasz_softmax_loss(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """Implements Lovász-Softmax loss.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Lovász-Softmax loss

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        self._validate_inputs(y_true, y_pred)

        def lovasz_grad(gt_sorted: tf.Tensor) -> tf.Tensor:
            """Computes Lovász gradient.

            Args:
                gt_sorted: Sorted ground truth values

            Returns:
                Lovász gradient
            """
            gts = tf.reduce_sum(gt_sorted)
            intersection = gts - tf.cumsum(gt_sorted)
            union = gts + tf.cumsum(1. - gt_sorted)
            jaccard = 1. - intersection / union
            jaccard = tf.concat(
                (jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0
            )
            return jaccard

        y_true = tf.cast(y_true, tf.float32)
        losses = []

        for c in range(self.config.num_classes):
            target_c = y_true[..., c]
            pred_c = y_pred[..., c]

            # Sort predictions by target values
            sorted_indices = tf.argsort(
                tf.reshape(target_c, [-1]),
                direction='DESCENDING'
            )
            pred_sorted = tf.gather(tf.reshape(pred_c, [-1]), sorted_indices)

            loss_c = tf.reduce_mean(lovasz_grad(pred_sorted))
            losses.append(loss_c)

        return tf.reduce_mean(losses)

    def combo_loss(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """Implements Combo loss (combination of Dice and Cross-Entropy).

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Combo loss

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        dice = self.dice_loss(y_true, y_pred)
        ce = self.cross_entropy_loss(y_true, y_pred)
        return (
                self.config.combo_alpha * dice +
                self.config.combo_beta * ce
        )

    def boundary_loss(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """Implements Boundary loss.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Boundary loss

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        self._validate_inputs(y_true, y_pred)
        y_true = tf.cast(y_true, tf.float32)

        # Compute distance transform
        kernel = tf.ones((3, 3, 1))
        dt = tf.nn.erosion2d(
            y_true,
            kernel=kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC"
        )

        # Normalize distance transform
        dt = tf.clip_by_value(dt, 0, self.config.boundary_theta)
        dt = dt / self.config.boundary_theta

        boundary_loss = tf.reduce_mean(dt * tf.square(1 - y_pred))
        return boundary_loss

    def hausdorff_distance_loss(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor
    ) -> tf.Tensor:
        """Implements an approximation of Hausdorff Distance loss.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Approximated Hausdorff Distance loss

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        self._validate_inputs(y_true, y_pred)
        y_true = tf.cast(y_true, tf.float32)

        # Compute distance transforms
        kernel = tf.ones((3, 3, 1))
        dt_true = tf.nn.erosion2d(
            y_true,
            kernel=kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC"
        )
        dt_pred = tf.nn.erosion2d(
            y_pred,
            kernel=kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC"
        )

        # Compute Hausdorff distances
        hd_true_pred = tf.reduce_max(dt_true * y_pred, axis=[1, 2])
        hd_pred_true = tf.reduce_max(dt_pred * y_true, axis=[1, 2])

        # Symmetric Hausdorff distance
        hd = tf.maximum(hd_true_pred, hd_pred_true)
        return tf.reduce_mean(hd)


def create_loss_function(
        loss_name: str,
        config: Optional[LossConfig] = None
) -> tf.keras.losses.Loss:
    """Creates a Keras loss function from the specified loss.

    Args:
        loss_name: Name of the loss function to create
        config: Optional LossConfig instance for loss parameters

    Returns:
        Keras loss function

    Raises:
        ValueError: If loss_name is not recognized
    """
    if config is None:
        config = LossConfig(num_classes=1)  # Default single-class config

    losses = SegmentationLosses(config)

    class WrappedLoss(tf.keras.losses.Loss):
        def __init__(self, loss_fn, name=loss_name):
            super().__init__(name=name)
            self.loss_fn = loss_fn

        def call(self, y_true, y_pred):
            return self.loss_fn(y_true, y_pred)

    loss_map = {
        'cross_entropy': losses.cross_entropy_loss,
        'dice': losses.dice_loss,
        'focal': losses.focal_loss,
        'tversky': losses.tversky_loss,
        'focal_tversky': losses.focal_tversky_loss,
        'lovasz': losses.lovasz_softmax_loss,
        'combo': losses.combo_loss,
        'boundary': losses.boundary_loss,
        'hausdorff': losses.hausdorff_distance_loss
    }

    if loss_name not in loss_map:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Available losses: {list(loss_map.keys())}"
        )

    return WrappedLoss(loss_map[loss_name])


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Create sample data
    batch_size, height, width = 2, 64, 64
    num_classes = 3

    config = LossConfig(
        num_classes=num_classes,
        smooth_factor=1e-6,
        focal_gamma=2.0,
        focal_alpha=0.25,
        tversky_alpha=0.3,
        tversky_beta=0.7
    )

    # Create random sample tensors
    y_true = tf.random.uniform(
        (batch_size, height, width, num_classes),
        minval=0,
        maxval=1
    )
    y_pred = tf.random.uniform(
        (batch_size, height, width, num_classes),
        minval=0,
        maxval=1
    )

    # Normalize predictions to create valid probability distribution
    y_pred = tf.nn.softmax(y_pred, axis=-1)

    # Initialize loss functions
    losses = SegmentationLosses(config)

    # Calculate various losses
    results = {
        'Cross-Entropy': losses.cross_entropy_loss(y_true, y_pred),
        'Dice': losses.dice_loss(y_true, y_pred),
        'Focal': losses.focal_loss(y_true, y_pred),
        'Tversky': losses.tversky_loss(y_true, y_pred),
        'Focal Tversky': losses.focal_tversky_loss(y_true, y_pred),
        'Lovász-Softmax': losses.lovasz_softmax_loss(y_true, y_pred),
        'Combo': losses.combo_loss(y_true, y_pred),
        'Boundary': losses.boundary_loss(y_true, y_pred),
        'Hausdorff': losses.hausdorff_distance_loss(y_true, y_pred)
    }

    # Print results
    print("\nLoss Values:")
    for name, value in results.items():
        print(f"{name:15s}: {value:0.4f}")

    # Example of creating a Keras loss function
    keras_loss = create_loss_function('combo', config)
    print("\nKeras loss value:", keras_loss(y_true, y_pred).numpy())