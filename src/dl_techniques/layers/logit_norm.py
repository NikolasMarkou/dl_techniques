"""
LogitNorm Variations - A Comprehensive Overview
=============================================

This module implements different variations of LogitNorm layers for various classification scenarios.
LogitNorm helps with Out-of-Distribution (OOD) detection and confidence calibration by normalizing
network outputs.

Base Variations:
---------------
1. LogitNorm:
   - Standard implementation of logit normalization
   - Uses L2 normalization on logits
   - Best for basic classification tasks
   - Helps prevent overconfident predictions
   - Key parameter: constant (scaling factor)

2. CoupledLogitNorm:
   - Extends LogitNorm for multi-label classification
   - Creates interdependence between label predictions
   - Implements a "confidence budget" across labels
   - Key parameters: constant, coupling_strength
   - Returns both normalized logits and normalization factor

Advanced Variations:
------------------
3. DynamicCoupledLogitNorm:
   - Adaptive version of CoupledLogitNorm
   - Coupling strength changes during training
   - Uses moving averages to track distribution statistics
   - Automatically adjusts to data characteristics
   - Key features:
     * Trainable coupling strength
     * Min/max coupling constraints
     * Distribution-based adaptation
     * Momentum-based updates

4. OODRobustLogitNorm:
   - Specifically designed for OOD detection
   - Includes temperature scaling for better calibration
   - Computes energy scores for OOD detection
   - Key features:
     * Learnable temperature parameter
     * Energy-based OOD scoring
     * Returns (logits, energy, temperature)

Usage Guidelines:
---------------
- Basic Classification: Use standard LogitNorm
- Multi-label Tasks: Use CoupledLogitNorm
- Distribution Shifts: Use DynamicCoupledLogitNorm
- OOD Detection: Use OODRobustLogitNorm


Key Benefits:
-----------
1. Improved Calibration:
   - Better uncertainty estimates
   - More reliable confidence scores
   - Reduced overconfidence

2. OOD Robustness:
   - Better handling of distribution shifts
   - Early detection of OOD samples
   - More reliable predictions on edge cases

3. Training Stability:
   - Normalized gradient flow
   - Better convergence properties
   - Reduced sensitivity to initialization

References:
----------
1. "LogitNorm: Rethinking Confident Logits"
2. "Energy-based Out-of-distribution Detection"
3. "Temperature Scaling for Neural Networks"
"""


import tensorflow as tf
from keras.api.layers import Layer
from typing import Optional, Union, Tuple, Dict, Any


# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class LogitNorm(Layer):
    """
    LogitNorm layer for classification tasks.

    This layer implements standard logit normalization, which helps stabilize
    training and can improve model calibration.

    Args:
        constant: Scaling factor for normalization. Higher values produce more
                 spread-out logits.
        axis: Axis along which to perform normalization.
        epsilon: Small constant for numerical stability.
    """

    def __init__(
            self,
            constant: float = 1.0,
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._validate_inputs(constant, epsilon)
        self.constant = constant
        self.axis = axis
        self.epsilon = epsilon

    def _validate_inputs(self, constant: float, epsilon: float) -> None:
        """Validate initialization parameters."""
        if constant <= 0:
            raise ValueError(f"constant must be positive, got {constant}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Apply logit normalization.

        Args:
            inputs: Input logits tensor
            training: Whether in training mode (unused)

        Returns:
            Normalized logits tensor
        """
        # Cast inputs to float
        inputs = tf.cast(inputs, self.compute_dtype)

        # Compute L2 norm
        x_squared = tf.square(inputs)
        x_norm = tf.sqrt(
            tf.reduce_sum(x_squared, axis=self.axis, keepdims=True) + self.epsilon
        )

        # Normalize logits
        return inputs / (x_norm * self.constant)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "constant": self.constant,
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config


# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class CoupledLogitNorm(Layer):
    """
    Coupled LogitNorm layer for multi-label classification.

    This layer implements a modified version of LogitNorm that deliberately couples
    label predictions through normalization, creating a form of "confidence budget"
    across labels.

    Args:
        constant: Scaling factor for normalization. Higher values reduce coupling.
        coupling_strength: Additional factor to control coupling strength (1.0 = normal LogitNorm).
        axis: Axis along which to perform normalization.
        epsilon: Small constant for numerical stability.
    """

    def __init__(
            self,
            constant: float = 1.0,
            coupling_strength: float = 1.0,
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._validate_inputs(constant, coupling_strength, epsilon)

        self.constant = constant
        self.coupling_strength = coupling_strength
        self.axis = axis
        self.epsilon = epsilon

    def _validate_inputs(self, constant: float, coupling_strength: float, epsilon: float) -> None:
        """Validate initialization parameters."""
        if constant <= 0:
            raise ValueError(f"constant must be positive, got {constant}")
        if coupling_strength < 0:
            raise ValueError(f"coupling_strength must be positive, got {coupling_strength}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply coupled logit normalization.

        Args:
            inputs: Input logits tensor
            training: Whether in training mode (unused)

        Returns:
            Tuple of (normalized_logits, normalizing_factor)
        """
        # Cast inputs to float
        inputs = tf.cast(inputs, self.compute_dtype)

        # Compute L2 norm with coupling strength
        x_squared = tf.square(inputs)
        x_norm = tf.reduce_sum(x_squared, axis=self.axis, keepdims=True)
        x_norm = tf.pow(x_norm + self.epsilon, self.coupling_strength / 2.0)

        # Normalize logits
        normalized_logits = inputs / (x_norm * self.constant)

        return normalized_logits, x_norm

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "constant": self.constant,
            "coupling_strength": self.coupling_strength,
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config


# ---------------------------------------------------------------------


class CoupledMultiLabelHead(Layer):
    """
    Multi-label classification head with coupled logit normalization.

    This head applies coupled LogitNorm followed by sigmoid activation,
    creating interdependence between label predictions.
    """

    def __init__(
            self,
            constant: float = 1.0,
            coupling_strength: float = 1.0,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.logit_norm = CoupledLogitNorm(
            constant=constant,
            coupling_strength=coupling_strength
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply coupled normalization and sigmoid activation."""
        normalized_logits, _ = self.logit_norm(inputs)
        return tf.sigmoid(normalized_logits)


# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class DynamicCoupledLogitNorm(Layer):
    """
    Dynamic Coupled LogitNorm layer for multi-label classification.

    This layer implements a modified version of LogitNorm with dynamic coupling strength
    that adapts during training based on the input distribution and gradient flow.

    Args:
        constant: Scaling factor for normalization. Higher values reduce coupling.
        initial_coupling: Initial coupling strength value (default=1.0).
        min_coupling: Minimum allowed coupling strength (default=0.1).
        max_coupling: Maximum allowed coupling strength (default=5.0).
        adaptation_rate: Learning rate for coupling strength adaptation (default=0.01).
        axis: Axis along which to perform normalization.
        epsilon: Small constant for numerical stability.
    """

    def __init__(
            self,
            constant: float = 1.0,
            initial_coupling: float = 1.0,
            min_coupling: float = 0.1,
            max_coupling: float = 5.0,
            adaptation_rate: float = 0.01,
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._validate_inputs(
            constant, initial_coupling, min_coupling,
            max_coupling, adaptation_rate, epsilon
        )

        self.constant = constant
        self.min_coupling = min_coupling
        self.max_coupling = max_coupling
        self.adaptation_rate = adaptation_rate
        self.axis = axis
        self.epsilon = epsilon

        # Initialize trainable coupling strength
        self.coupling_strength = self.add_weight(
            name='coupling_strength',
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_coupling),
            constraint=lambda x: tf.clip_by_value(x, min_coupling, max_coupling),
            trainable=True
        )

    def _validate_inputs(
            self,
            constant: float,
            initial_coupling: float,
            min_coupling: float,
            max_coupling: float,
            adaptation_rate: float,
            epsilon: float
    ) -> None:
        """Validate initialization parameters."""
        if constant <= 0:
            raise ValueError(f"constant must be positive, got {constant}")
        if initial_coupling <= 0:
            raise ValueError(f"initial_coupling must be positive, got {initial_coupling}")
        if min_coupling <= 0:
            raise ValueError(f"min_coupling must be positive, got {min_coupling}")
        if max_coupling <= min_coupling:
            raise ValueError(
                f"max_coupling must be greater than min_coupling, "
                f"got {max_coupling} <= {min_coupling}"
            )
        if adaptation_rate <= 0:
            raise ValueError(f"adaptation_rate must be positive, got {adaptation_rate}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer."""
        # Create moving averages for adaptive coupling
        self.moving_norm_mean = self.add_weight(
            name='moving_norm_mean',
            shape=(),
            initializer='zeros',
            trainable=False
        )
        self.moving_norm_var = self.add_weight(
            name='moving_norm_var',
            shape=(),
            initializer='ones',
            trainable=False
        )
        super().build(input_shape)

    def _compute_coupling_update(
            self,
            x_norm: tf.Tensor,
            training: Optional[bool]
    ) -> None:
        """
        Compute and apply coupling strength update based on input statistics.

        Args:
            x_norm: Current batch normalization factors
            training: Whether in training mode
        """
        if training:
            # Compute current batch statistics
            batch_mean = tf.reduce_mean(x_norm)
            batch_var = tf.reduce_variance(x_norm)

            # Update moving averages
            momentum = 0.9
            self.moving_norm_mean.assign(
                momentum * self.moving_norm_mean +
                (1 - momentum) * batch_mean
            )
            self.moving_norm_var.assign(
                momentum * self.moving_norm_var +
                (1 - momentum) * batch_var
            )

            # Compute coupling adjustment based on distribution stability
            distribution_stability = tf.sqrt(
                batch_var / (self.moving_norm_var + self.epsilon)
            )

            # Update coupling strength
            coupling_update = self.adaptation_rate * (
                    distribution_stability - 1.0
            )
            self.coupling_strength.assign_add(coupling_update)

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply dynamic coupled logit normalization.

        Args:
            inputs: Input logits tensor
            training: Whether in training mode

        Returns:
            Tuple of (normalized_logits, normalizing_factor)
        """
        # Cast inputs to float
        inputs = tf.cast(inputs, self.compute_dtype)

        # Compute L2 norm with current coupling strength
        x_squared = tf.square(inputs)
        x_norm = tf.reduce_sum(x_squared, axis=self.axis, keepdims=True)
        x_norm = tf.pow(
            x_norm + self.epsilon,
            self.coupling_strength / 2.0
        )

        # Update coupling strength during training
        self._compute_coupling_update(x_norm, training)

        # Normalize logits
        normalized_logits = inputs / (x_norm * self.constant)

        return normalized_logits, x_norm

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "constant": self.constant,
            "initial_coupling": self.coupling_strength.numpy(),
            "min_coupling": self.min_coupling,
            "max_coupling": self.max_coupling,
            "adaptation_rate": self.adaptation_rate,
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config


# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class OODRobustLogitNorm(Layer):
    """
    LogitNorm layer specifically designed for OOD robustness.

    Features:
    - Temperature scaling for calibration
    - Energy-based OOD detection
    - Adaptive margin for decision boundary adjustment
    """

    def __init__(
            self,
            constant: float = 1.0,
            initial_temperature: float = 1.0,
            energy_margin: float = 1.0,
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._validate_inputs(constant, initial_temperature, energy_margin, epsilon)

        self.constant = constant
        self.energy_margin = energy_margin
        self.axis = axis
        self.epsilon = epsilon

        # Temperature parameter for calibration
        self.temperature = self.add_weight(
            name='temperature',
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_temperature),
            constraint=lambda x: tf.maximum(x, epsilon),
            trainable=True
        )

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Apply OOD-robust logit normalization.

        Returns:
            Tuple of (normalized_logits, energy_scores, temperature)
        """
        inputs = tf.cast(inputs, self.compute_dtype)

        # L2 normalization
        x_squared = tf.square(inputs)
        x_norm = tf.sqrt(
            tf.reduce_sum(x_squared, axis=self.axis, keepdims=True) + self.epsilon
        )

        # Temperature scaling
        scaled_inputs = inputs / self.temperature

        # Normalize logits
        normalized_logits = scaled_inputs / (x_norm * self.constant)

        # Compute energy score for OOD detection
        # Lower energy typically indicates in-distribution samples
        energy = -tf.reduce_logsumexp(normalized_logits, axis=self.axis)

        return normalized_logits, energy, self.temperature

# ---------------------------------------------------------------------
