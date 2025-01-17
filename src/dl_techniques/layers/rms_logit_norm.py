"""
# Neural Network Normalization Layers

A comprehensive collection of normalization layers focusing on logit normalization
and RMS normalization for various deep learning tasks, particularly targeting
confidence calibration and out-of-distribution detection.

## Layer Types

### 1. RMSNorm
- Root mean square normalization
- Stabilizes training through magnitude normalization
- Best for general neural network layers
- Key parameter: constant (scaling factor)

### 2. LogitNorm
- L2 normalization with temperature scaling
- Specifically designed for network output layers
- Helps prevent overconfident predictions
- Key parameter: temperature (calibration factor)

### 3. CoupledLogitNorm
- Multi-label classification variant
- Creates label prediction interdependence
- Implements confidence budgeting across labels
- Returns normalized logits and norm factor
- Key parameters: constant, coupling_strength

### 4. DynamicCoupledLogitNorm
- Adaptive coupling mechanism
- Self-adjusting normalization strength
- Features:
    * Trainable coupling parameter
    * Constrained coupling range
    * Distribution-aware adaptation
    * Momentum-based statistics

### 5. OODRobustLogitNorm
- Specialized for OOD detection
- Temperature calibration
- Energy-based scoring
- Features:
    * Learnable temperature
    * Energy scoring for OOD
    * Returns logits, energy scores, and temperature

### 6. HybridLogitRMSNorm
- Combined RMS and Logit normalization
- Sequential application of both normalizations
- Optional learnable weighted combination
- Features:
    * Two-stage normalization process
    * Configurable temperature and RMS scaling
    * Training stability with calibration benefits
- Key parameters: temperature, rms_constant, weighted_combination

## Usage Recommendations
- General Layers: RMSNorm
- Classification Output: LogitNorm
- Multi-label Tasks: CoupledLogitNorm
- Distribution Shifts: DynamicCoupledLogitNorm
- OOD Detection: OODRobustLogitNorm
- Complex Tasks: HybridLogitRMSNorm (when both stability and calibration are crucial)

## Implementation Benefits

### 1. Training Stability
- Controlled gradient magnitudes
- Improved convergence
- Initialization robustness
- Enhanced gradient flow (HybridLogitRMSNorm)

### 2. Model Calibration
- Better uncertainty quantification
- Reliable confidence scores
- Reduced overconfidence issues
- Multi-stage calibration (HybridLogitRMSNorm)

### 3. OOD Detection
- Distribution shift handling
- Anomaly detection capability
- Robust edge case behavior
- Enhanced detection through combined normalization

### 4. Hybrid Benefits (HybridLogitRMSNorm)
- Combined stability and calibration
- Adaptive normalization weighting
- Flexible architecture integration
- Improved robustness through dual normalization

## Layer Selection Guide

When choosing a normalization layer, consider:

1. Task Type:
   - Single-label classification → LogitNorm or HybridLogitRMSNorm
   - Multi-label classification → CoupledLogitNorm
   - General feature extraction → RMSNorm
   - OOD detection → OODRobustLogitNorm

2. Training Characteristics:
   - Unstable training → RMSNorm or HybridLogitRMSNorm
   - Overconfidence issues → LogitNorm
   - Distribution shifts → DynamicCoupledLogitNorm
   - Complex scenarios → HybridLogitRMSNorm with weighted combination

3. Model Architecture:
   - Deep networks → Consider HybridLogitRMSNorm for stability
   - Output layers → LogitNorm variants
   - Hidden layers → RMSNorm or weighted HybridLogitRMSNorm

## References
[1] "Mitigating Neural Network Overconfidence with Logit Normalization"
[2] "Root Mean Square Layer Normalization"
[3] "Energy-based Out-of-distribution Detection"
[4] "Temperature Scaling for Neural Networks"
[5] "Hybrid Normalization for Deep Neural Networks"
"""

import keras
import tensorflow as tf
from keras.api.layers import Layer
from typing import Optional, Union, Tuple, Dict, Any


# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class RMSNorm(Layer):
    """
    Root Mean Square Normalization layer for classification tasks.

    This layer implements root mean square normalization by normalizing inputs by their
    RMS value. Unlike LogitNorm which uses L2 normalization, RMSNorm uses root mean
    square for normalization, which can help stabilize training and improve model
    robustness.

    The normalization is computed as:
        output = input / sqrt(mean(input^2) + epsilon) * constant

    This implementation differs from LogitNorm in that it:
    - Uses mean of squared values rather than sum
    - Applies a constant scaling factor instead of temperature
    - Does not specifically target logit calibration

    Args:
        constant: float, default=1.0
            Scaling factor applied after normalization. Higher values produce
            outputs with larger magnitudes.
        axis: int, default=-1
            Axis along which to compute RMS statistics. The default (-1)
            computes RMS over the last dimension.
        epsilon: float, default=1e-7
            Small constant added to denominator for numerical stability.

    Inputs:
        A tensor of any rank

    Outputs:
        A tensor of the same shape as the input, normalized by RMS values

    References:
        "Root Mean Square Layer Normalization", 2019
        https://arxiv.org/abs/1910.07467
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
@keras.utils.register_keras_serializable()
class LogitNorm(Layer):
    """
    LogitNorm layer for classification tasks.

    This layer implements logit normalization by applying L2 normalization with a learned temperature
    parameter. This helps stabilize training and can improve model calibration.

    Args:
        temperature: Float, temperature scaling parameter. Higher values produce more spread-out logits.
        axis: Integer, axis along which to perform normalization.
        epsilon: Float, small constant for numerical stability.

    References:
        - Paper: "Mitigating Neural Network Overconfidence with Logit Normalization"
    """

    def __init__(
            self,
            temperature: float = 0.04,  # Default from paper for CIFAR-10
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._validate_inputs(temperature, epsilon)
        self.temperature = temperature
        self.axis = axis
        self.epsilon = epsilon

    def _validate_inputs(self, temperature: float, epsilon: float) -> None:
        """Validate initialization parameters."""
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
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

        # Compute L2 norm along specified axis
        norm = tf.sqrt(
            tf.maximum(
                tf.reduce_sum(tf.square(inputs), axis=self.axis, keepdims=True),
                self.epsilon
            )
        )

        # Normalize logits and scale by temperature
        return inputs / (norm * self.temperature)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "temperature": self.temperature,
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config


# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
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

@keras.utils.register_keras_serializable()
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
            initializer=keras.initializers.Constant(initial_coupling),
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
            batch_var = tf.math.reduce_variance(x_norm)

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

@keras.utils.register_keras_serializable()
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
            initializer=keras.initializers.Constant(initial_temperature),
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

@keras.utils.register_keras_serializable()
class HybridLogitRMSNorm(Layer):
    """
    Hybrid normalization layer combining LogitNorm and RMSNorm approaches.

    This layer implements a novel combination of logit normalization and root mean square
    normalization, providing both calibration benefits and training stability. The layer
    first applies RMS normalization followed by temperature-scaled L2 normalization.

    The normalization process occurs in two steps:
    1. RMS Normalization: x_rms = x / sqrt(mean(x^2) + epsilon)
    2. Logit Normalization: output = x_rms / (||x_rms||_2 * temperature)

    Args:
        temperature: float, default=0.04
            Temperature scaling parameter for logit normalization. Higher values
            produce more spread-out logits.
        rms_constant: float, default=1.0
            Scaling factor for RMS normalization. Controls the magnitude of
            intermediate activations.
        axis: int, default=-1
            Axis along which to perform normalization operations.
        epsilon: float, default=1e-7
            Small constant added for numerical stability.
        weighted_combination: bool, default=False
            If True, uses learnable weights to combine RMS and Logit normalizations.

    Attributes:
        alpha: tf.Variable, optional
            Learnable weight for combining normalizations when weighted_combination=True.

    Example:
        ```python
        # Create layer
        norm_layer = HybridLogitRMSNorm(temperature=0.04, rms_constant=1.0)

        # Apply to logits
        normalized_logits = norm_layer(logits)
        ```

    References:
        - "Root Mean Square Layer Normalization"
        - "Mitigating Neural Network Overconfidence with Logit Normalization"
    """

    def __init__(
            self,
            temperature: float = 0.04,
            rms_constant: float = 1.0,
            axis: int = -1,
            epsilon: float = 1e-7,
            weighted_combination: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._validate_inputs(temperature, rms_constant, epsilon)

        self.temperature = temperature
        self.rms_constant = rms_constant
        self.axis = axis
        self.epsilon = epsilon
        self.weighted_combination = weighted_combination

        # Initialize alpha if using weighted combination
        if weighted_combination:
            self.alpha = self.add_weight(
                name='alpha',
                shape=(),
                initializer=keras.initializers.Constant(0.5),
                constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0),
                trainable=True
            )

    def _validate_inputs(
            self,
            temperature: float,
            rms_constant: float,
            epsilon: float
    ) -> None:
        """Validate initialization parameters."""
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if rms_constant <= 0:
            raise ValueError(f"rms_constant must be positive, got {rms_constant}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def _apply_rms_norm(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply RMS normalization to inputs."""
        mean_square = tf.reduce_mean(
            tf.square(inputs),
            axis=self.axis,
            keepdims=True
        )
        rms_norm = tf.math.rsqrt(mean_square + self.epsilon)
        return inputs * rms_norm * self.rms_constant

    def _apply_logit_norm(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply logit normalization to inputs."""
        square_sum = tf.reduce_sum(
            tf.square(inputs),
            axis=self.axis,
            keepdims=True
        )
        l2_norm = tf.math.rsqrt(square_sum + self.epsilon)
        return inputs * l2_norm / self.temperature

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, Dict[str, tf.Tensor]]]:
        """
        Apply hybrid normalization to inputs.

        Args:
            inputs: Input tensor to normalize
            training: Whether in training mode (unused)

        Returns:
            If weighted_combination is False:
                Normalized tensor
            If weighted_combination is True:
                Tuple of (normalized tensor, dict with intermediate values)
        """
        # Cast inputs to compute dtype
        inputs = tf.cast(inputs, self.compute_dtype)

        # Apply RMS normalization
        rms_normalized = self._apply_rms_norm(inputs)

        # Apply Logit normalization
        logit_normalized = self._apply_logit_norm(rms_normalized)

        if self.weighted_combination:
            # Combine normalizations using learned weight
            final_output = (
                    self.alpha * logit_normalized +
                    (1 - self.alpha) * rms_normalized
            )

            # Return output and intermediate values
            return final_output, {
                'rms_normalized': rms_normalized,
                'logit_normalized': logit_normalized,
                'alpha': self.alpha
            }

        return logit_normalized

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'rms_constant': self.rms_constant,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'weighted_combination': self.weighted_combination
        })
        return config

# ---------------------------------------------------------------------
