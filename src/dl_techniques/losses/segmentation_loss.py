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

import keras
from keras import ops
from dataclasses import dataclass
from typing import Optional, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


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

    Example:
        >>> config = LossConfig(num_classes=3, focal_gamma=2.0)
        >>> losses = SegmentationLosses(config)
        >>> dice_loss = losses.dice_loss(y_true, y_pred)
    """

    def __init__(self, config: LossConfig) -> None:
        """Initialize the SegmentationLosses class.

        Args:
            config: Configuration parameters for loss functions
        """
        self._validate_config(config)
        self.config = config
        logger.info(f"Initialized SegmentationLosses with {config.num_classes} classes")

    def _validate_config(self, config: LossConfig) -> None:
        """Validate the configuration parameters.

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
        if config.tversky_alpha < 0 or config.tversky_beta < 0:
            raise ValueError(f"Tversky parameters must be non-negative")
        if config.combo_alpha < 0 or config.combo_beta < 0:
            raise ValueError(f"Combo parameters must be non-negative")

    @staticmethod
    def _validate_inputs(
            y_true: Any,
            y_pred: Any,
            weights: Optional[Any] = None
    ) -> None:
        """Validate input tensors.

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
            y_true: Any,
            y_pred: Any,
            weights: Optional[Any] = None
    ) -> Any:
        """Implement weighted cross-entropy loss.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)
            weights: Optional class weights (num_classes,)

        Returns:
            Weighted cross-entropy loss tensor

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        self._validate_inputs(y_true, y_pred, weights)
        y_true = ops.cast(y_true, "float32")

        # Add epsilon to prevent log(0)
        epsilon = 1e-7
        y_pred = ops.clip(y_pred, epsilon, 1.0 - epsilon)

        ce_loss = -ops.sum(y_true * ops.log(y_pred), axis=-1)

        if weights is not None:
            weights = ops.cast(weights, "float32")
            ce_loss = ce_loss * ops.sum(y_true * weights, axis=-1)

        return ops.mean(ce_loss)

    def dice_loss(
            self,
            y_true: Any,
            y_pred: Any
    ) -> Any:
        """Implement Dice loss.

        The Dice loss is based on the Dice coefficient, which measures the overlap
        between predicted and ground truth segmentations.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Dice loss tensor

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        self._validate_inputs(y_true, y_pred)
        y_true = ops.cast(y_true, "float32")

        numerator = 2 * ops.sum(y_true * y_pred, axis=[1, 2])
        denominator = (
                ops.sum(y_true, axis=[1, 2]) +
                ops.sum(y_pred, axis=[1, 2])
        )

        dice_coef = (numerator + self.config.smooth_factor) / (
                denominator + self.config.smooth_factor
        )
        return 1.0 - ops.mean(dice_coef)

    def focal_loss(
            self,
            y_true: Any,
            y_pred: Any
    ) -> Any:
        """Implement Focal loss.

        Focal loss addresses class imbalance by down-weighting easy examples
        and focusing on hard examples.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Focal loss tensor

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        self._validate_inputs(y_true, y_pred)
        y_true = ops.cast(y_true, "float32")

        epsilon = 1e-7
        y_pred = ops.clip(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * ops.log(y_pred)
        focal_factor = ops.power(1.0 - y_pred, self.config.focal_gamma)

        focal_ce = self.config.focal_alpha * focal_factor * cross_entropy
        return ops.mean(ops.sum(focal_ce, axis=-1))

    def tversky_loss(
            self,
            y_true: Any,
            y_pred: Any
    ) -> Any:
        """Implement Tversky loss.

        Tversky loss is a generalization of Dice loss that allows for different
        weighting of false positives and false negatives.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Tversky loss tensor

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        self._validate_inputs(y_true, y_pred)
        y_true = ops.cast(y_true, "float32")

        numerator = ops.sum(y_true * y_pred, axis=[1, 2])
        false_positives = ops.sum((1 - y_true) * y_pred, axis=[1, 2])
        false_negatives = ops.sum(y_true * (1 - y_pred), axis=[1, 2])

        denominator = (
                numerator +
                self.config.tversky_alpha * false_positives +
                self.config.tversky_beta * false_negatives
        )

        tversky_coef = (numerator + self.config.smooth_factor) / (
                denominator + self.config.smooth_factor
        )
        return 1.0 - ops.mean(tversky_coef)

    def focal_tversky_loss(
            self,
            y_true: Any,
            y_pred: Any
    ) -> Any:
        """Implement Focal Tversky loss.

        Combines the benefits of both Focal loss and Tversky loss by applying
        the focal mechanism to the Tversky index.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Focal Tversky loss tensor

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        # Calculate Tversky coefficient first
        y_true_cast = ops.cast(y_true, "float32")
        numerator = ops.sum(y_true_cast * y_pred, axis=[1, 2])
        false_positives = ops.sum((1 - y_true_cast) * y_pred, axis=[1, 2])
        false_negatives = ops.sum(y_true_cast * (1 - y_pred), axis=[1, 2])

        denominator = (
                numerator +
                self.config.tversky_alpha * false_positives +
                self.config.tversky_beta * false_negatives
        )

        tversky_coef = (numerator + self.config.smooth_factor) / (
                denominator + self.config.smooth_factor
        )

        # Apply focal mechanism
        focal_tversky = ops.power(1.0 - tversky_coef, self.config.focal_tversky_gamma)
        return ops.mean(focal_tversky)

    def lovasz_softmax_loss(
            self,
            y_true: Any,
            y_pred: Any
    ) -> Any:
        """Implement Lovász-Softmax loss.

        The Lovász-Softmax loss is based on the Lovász extension of submodular
        losses for multi-class segmentation.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Lovász-Softmax loss tensor

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        self._validate_inputs(y_true, y_pred)

        def lovasz_grad(gt_sorted: Any) -> Any:
            """Compute Lovász gradient.

            Args:
                gt_sorted: Sorted ground truth values

            Returns:
                Lovász gradient tensor
            """
            gts = ops.sum(gt_sorted)
            intersection = gts - ops.cumsum(gt_sorted, axis=0)
            union = gts + ops.cumsum(1.0 - gt_sorted, axis=0)
            jaccard = 1.0 - intersection / (union + self.config.smooth_factor)

            # Compute differences for gradient
            jaccard_diff = ops.concatenate([
                jaccard[0:1],
                jaccard[1:] - jaccard[:-1]
            ], axis=0)
            return jaccard_diff

        y_true = ops.cast(y_true, "float32")
        losses = []

        for c in range(self.config.num_classes):
            target_c = y_true[..., c]
            pred_c = y_pred[..., c]

            # Flatten tensors for sorting
            target_flat = ops.reshape(target_c, [-1])
            pred_flat = ops.reshape(pred_c, [-1])

            # Sort by prediction values in descending order
            # Since argsort doesn't support direction, we negate values to sort descending
            sorted_indices = ops.argsort(-pred_flat)
            target_sorted = ops.take(target_flat, sorted_indices)

            grad = lovasz_grad(target_sorted)
            loss_c = ops.sum(grad * ops.take(-pred_flat, sorted_indices))  # Use negated values
            losses.append(-loss_c)  # Negate back to get correct sign

        return ops.mean(ops.stack(losses))

    def combo_loss(
            self,
            y_true: Any,
            y_pred: Any
    ) -> Any:
        """Implement Combo loss (combination of Dice and Cross-Entropy).

        Combo loss combines the benefits of both Dice loss and Cross-Entropy loss
        for better segmentation performance.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Combo loss tensor

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
            y_true: Any,
            y_pred: Any
    ) -> Any:
        """Implement Boundary loss.

        Boundary loss focuses on the boundaries between different classes,
        using a simplified distance-based approach.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Boundary loss tensor

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        self._validate_inputs(y_true, y_pred)
        y_true = ops.cast(y_true, "float32")

        # Compute boundary map using gradient magnitude
        # This is a simplified version using difference operations instead of convolution
        def compute_boundary_map(mask: Any) -> Any:
            """Compute boundary map using difference-based edge detection.

            Args:
                mask: Input mask tensor

            Returns:
                Boundary map tensor
            """
            # Use simple differences to approximate gradients
            # Pad the tensor to handle boundaries
            padded_mask = ops.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='constant')

            # Compute gradients using differences
            grad_x = padded_mask[:, 1:-1, 2:, :] - padded_mask[:, 1:-1, :-2, :]
            grad_y = padded_mask[:, 2:, 1:-1, :] - padded_mask[:, :-2, 1:-1, :]

            # Compute gradient magnitude
            magnitude = ops.sqrt(grad_x**2 + grad_y**2 + self.config.smooth_factor)

            return magnitude

        boundary_map = compute_boundary_map(y_true)

        # Normalize boundary map
        boundary_map = ops.clip(boundary_map, 0, self.config.boundary_theta)
        boundary_map = boundary_map / self.config.boundary_theta

        boundary_loss = ops.mean(boundary_map * ops.square(1 - y_pred))
        return boundary_loss

    def hausdorff_distance_loss(
            self,
            y_true: Any,
            y_pred: Any
    ) -> Any:
        """Implement an approximation of Hausdorff Distance loss.

        This implements a simplified version of Hausdorff distance using
        morphological operations approximated with pooling.

        Args:
            y_true: Ground truth labels (batch_size, height, width, num_classes)
            y_pred: Predicted probabilities (batch_size, height, width, num_classes)

        Returns:
            Approximated Hausdorff Distance loss tensor

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        self._validate_inputs(y_true, y_pred)
        y_true = ops.cast(y_true, "float32")

        # Simplified distance approximation using difference operations
        def approximate_distance_transform(mask: Any) -> Any:
            """Approximate distance transform using morphological operations.

            Args:
                mask: Input mask tensor

            Returns:
                Approximated distance transform
            """
            # Use iterative dilation approximation with difference operations
            result = mask
            for iteration in range(3):  # Apply multiple iterations for distance approximation
                # Pad tensor for boundary handling
                padded = ops.pad(result, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='constant')

                # Approximate dilation using maximum of neighboring pixels
                dilated = ops.maximum(
                    ops.maximum(
                        ops.maximum(padded[:, :-2, 1:-1, :], padded[:, 2:, 1:-1, :]),
                        ops.maximum(padded[:, 1:-1, :-2, :], padded[:, 1:-1, 2:, :])
                    ),
                    result
                )
                result = dilated

            return result

        dt_true = approximate_distance_transform(y_true)
        dt_pred = approximate_distance_transform(y_pred)

        # Compute approximate Hausdorff distances
        hd_true_pred = ops.max(dt_true * y_pred, axis=[1, 2])
        hd_pred_true = ops.max(dt_pred * y_true, axis=[1, 2])

        # Symmetric Hausdorff distance
        hd = ops.maximum(hd_true_pred, hd_pred_true)
        return ops.mean(hd)


def create_loss_function(
        loss_name: str,
        config: Optional[LossConfig] = None
) -> keras.losses.Loss:
    """Create a Keras loss function from the specified loss.

    Args:
        loss_name: Name of the loss function to create. Available options:
            'cross_entropy', 'dice', 'focal', 'tversky', 'focal_tversky',
            'lovasz', 'combo', 'boundary', 'hausdorff'
        config: Optional LossConfig instance for loss parameters

    Returns:
        Keras loss function ready to use in model compilation

    Raises:
        ValueError: If loss_name is not recognized

    Example:
        >>> config = LossConfig(num_classes=3, focal_gamma=2.0)
        >>> loss_fn = create_loss_function('focal', config)
        >>> model.compile(optimizer='adam', loss=loss_fn)
    """
    if config is None:
        config = LossConfig(num_classes=1)  # Default single-class config
        logger.info("Using default LossConfig with single class")

    losses = SegmentationLosses(config)

    class WrappedLoss(keras.losses.Loss):
        """Wrapper class to make segmentation losses compatible with Keras."""

        def __init__(self, loss_fn: callable, name: str = loss_name):
            """Initialize the wrapped loss function.

            Args:
                loss_fn: The actual loss function to wrap
                name: Name of the loss function
            """
            super().__init__(name=name)
            self.loss_fn = loss_fn

        def call(self, y_true: Any, y_pred: Any) -> Any:
            """Call the wrapped loss function.

            Args:
                y_true: Ground truth labels
                y_pred: Predicted probabilities

            Returns:
                Loss value
            """
            return self.loss_fn(y_true, y_pred)

        def get_config(self) -> dict:
            """Get configuration for serialization.

            Returns:
                Configuration dictionary
            """
            config = super().get_config()
            config.update({'name': self.name})
            return config

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
        available_losses = list(loss_map.keys())
        logger.error(f"Unknown loss function: {loss_name}. Available: {available_losses}")
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Available losses: {available_losses}"
        )

    logger.info(f"Created {loss_name} loss function")
    return WrappedLoss(loss_map[loss_name])


# Example usage and testing
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
    y_true = ops.random.uniform(
        (batch_size, height, width, num_classes),
        minval=0,
        maxval=1
    )
    y_pred = ops.random.uniform(
        (batch_size, height, width, num_classes),
        minval=0,
        maxval=1
    )

    # Normalize predictions to create valid probability distribution
    y_pred = ops.softmax(y_pred, axis=-1)

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

    # Log results
    logger.info("Loss function test results:")
    for name, value in results.items():
        logger.info(f"{name:15s}: {float(value):0.4f}")

    # Example of creating a Keras loss function
    keras_loss = create_loss_function('combo', config)
    keras_loss_value = keras_loss(y_true, y_pred)
    logger.info(f"Keras loss value: {float(keras_loss_value):0.4f}")