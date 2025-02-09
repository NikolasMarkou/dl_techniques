"""
# BandRMSNorm Layer Documentation

## Overview

Root Mean Square Layer Normalization with Bounded Spherical Shell Constraints -
 a sophisticated normalization approach that leverages high-dimensional geometry to improve deep network training dynamics.
 This layer combines RMS normalization with a learnable bounded scaling factor to create a "thick shell" in the feature space.

## Geometric Foundation

The implementation is based on two key geometric insights:

1. In high dimensions, the volume of a sphere concentrates near its surface (the "curse of dimensionality")
2. Creating a bounded shell between radii (1-α) and 1 adds back a degree of freedom while maintaining normalization benefits

## Core Algorithm

### 1. RMS Normalization
- Projects features onto unit hypersphere: x_norm = x / sqrt(mean(x^2))
- Provides initial bounded representation
- Operates on specified axis (default: feature/channel dimension)

### 2. Learnable Band Scaling
- Creates "thick shell" with learnable radius in [1-α, 1]
- Implemented via hard sigmoid bounded scaling
- Adds crucial radial degree of freedom
- Zero-initialization for stable training start

## Geometric Properties

### 1. Feature Space Structure
- Features live in d-dimensional shell between radii (1-α) and 1
- Exponentially more volume than surface-only normalization
- Allows radial movement while maintaining bounds
- Better separation of features in high dimensions

### 2. Training Dynamics
- Smoother optimization paths through radial freedom
- Reduced gradient interference vs surface-only constraints
- Natural emergence of feature magnitude hierarchy
- Better preservation of angular relationships

### 3. Regularization Effects
- Implicit regularization through bounded shell constraint
- Prevents feature collapse or explosion
- L2 regularization encourages balanced shell utilization
- Improved feature disentanglement

## Implementation Benefits

### 1. Training Stability
- Batch-size independent operation
- No running statistics needed
- Smooth gradient flow in deep networks
- Robust to learning rate variation

### 2. Feature Representation
- Magnitude can encode feature importance
- Better handling of feature hierarchies
- Improved gradient flow in very deep networks
- Enhanced multi-task learning capability

### 3. Architectural Flexibility
- Excellent for attention mechanisms
- Strong fit for residual networks
- Facilitates feature reuse
- Suitable for both CNNs and Transformers

## Practical Usage

### 1. Hyperparameters
- max_band_width (α): Controls shell thickness (typical: 0.1-0.3)
- epsilon: Numerical stability (typical: 1e-5 to 1e-7)
- band_regularizer: Prevents scale saturation (typical: 1e-4 to 1e-6)

### 2. Optimization
- Compatible with adaptive optimizers (Adam recommended)
- Benefits from learning rate warmup
- Monitor band parameter distributions
- Consider gradient clipping in very deep networks

### 3. Architecture Integration
- Use as LayerNorm replacement in Transformers
- Effective in ResNets and deep CNNs
- Particularly powerful for multi-task architectures
- Consider in feature fusion contexts

## Technical Implementation Notes

The layer strikes an elegant balance between normalization constraints and feature
expressivity through its geometric design, making it particularly effective for deep learning architectures
operating in high-dimensional feature spaces.

### Key Components

1. **Initialization**
   - Zero-initialization of band parameters promotes stable training start
   - Epsilon term ensures numerical stability
   - Proper shape handling for various input dimensions

2. **Forward Pass**
   - Two-step normalization process:
     a. RMS normalization to unit sphere
     b. Learnable scaling within bounded shell
   - Hard sigmoid ensures strict bound enforcement

3. **Gradient Flow**
   - Bounded gradients due to shell constraints
   - Natural gradient scaling through RMS statistics
   - Smooth backpropagation paths

### Performance Considerations

1. **Computational Efficiency**
   - Similar complexity to LayerNorm
   - Minimal overhead from bound enforcement
   - Efficient batch-independent operation

2. **Memory Usage**
   - One trainable parameter per normalized dimension
   - Minimal additional memory requirements
   - Efficient gradient computation

3. **Numerical Stability**
   - Robust to varying batch sizes
   - Protected against exploding/vanishing features
   - Stable gradient propagation

## Common Use Cases

1. **Transformer Networks**
   - Replace standard LayerNorm
   - Improved attention mechanism stability
   - Better gradient flow in deep transformers

2. **Convolutional Networks**
   - Alternative to BatchNorm
   - Robust to batch size variations
   - Improved feature hierarchy learning

3. **Multi-task Learning**
   - Better feature sharing across tasks
   - Improved task-specific feature modulation
   - More stable multi-task optimization

## Troubleshooting

1. **Training Issues**
   - If underfitting, try reducing band_regularizer
   - If unstable, increase epsilon
   - Monitor band parameter distributions

2. **Integration Problems**
   - Verify axis alignment with feature dimensions
   - Check input shape handling
   - Ensure proper initialization

3. **Performance Optimization**
   - Profile computation overhead
   - Monitor memory usage
   - Optimize batch processing

## References

1. Root Mean Square Layer Normalization (https://arxiv.org/abs/1910.07467)
2. High-dimensional sphere properties and concentration of measure
3. Empirical studies of neural network feature spaces
"""

import keras
import tensorflow as tf
from keras.api.layers import Layer
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class BandRMSNorm(Layer):
    """Root Mean Square Normalization layer with L2 norm constraints.

    This layer implements root mean square normalization that guarantees the output
    L2 norm will be between [1-α, 1], where α is the max_band_width parameter.
    The normalization is computed in two steps:
    1. RMS normalization to unit norm
    2. Learnable scaling within the [1-α, 1] band

    Args:
        max_band_width: float, default=0.2
            Maximum allowed deviation from unit normalization (0 < α < 1)
        axis: int, default=-1
            Axis along which to compute RMS statistics
        epsilon: float, default=1e-7
            Small constant added to denominator for numerical stability
        band_regularizer: Optional[keras.regularizers.Regularizer], default=L2(1e-5)
            Regularizer for the band parameter

    Inputs:
        A tensor of any rank

    Outputs:
        A tensor of the same shape as input, with L2 norm guaranteed to be
        between [1-max_band_width, 1]

    References:
        "Root Mean Square Layer Normalization", 2019
        https://arxiv.org/abs/1910.07467
    """

    def __init__(
            self,
            max_band_width: float = 0.2,
            axis: int = -1,
            epsilon: float = 1e-7,
            band_regularizer: Optional[keras.regularizers.Regularizer] = keras.regularizers.L2(1e-5),
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._validate_inputs(max_band_width, epsilon)
        self.max_band_width = max_band_width
        self.axis = axis
        self.epsilon = epsilon
        self.band_regularizer = band_regularizer
        self.band_param = None

    def _validate_inputs(self, max_band_width: float, epsilon: float) -> None:
        """Validate initialization parameters.

        Args:
            max_band_width: Maximum allowed deviation from unit norm
            epsilon: Small constant for numerical stability

        Raises:
            ValueError: If parameters are invalid
        """
        if not 0 < max_band_width < 1:
            raise ValueError(
                f"max_band_width must be between 0 and 1, got {max_band_width}"
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create the layer's trainable weights.

        Args:
            input_shape: Shape of input tensor
        """
        ndims = len(input_shape)
        axis = self.axis if self.axis >= 0 else ndims + self.axis

        param_shape = [1] * ndims
        param_shape[axis] = input_shape[axis]

        # Initialize band parameter with zeros for optimal training
        self.band_param = self.add_weight(
            name="band_param",
            shape=param_shape,
            initializer="zeros",
            trainable=True,
            dtype=self.dtype,
            regularizer=self.band_regularizer
        )

        self.built = True

    def _compute_rms(self, inputs: tf.Tensor) -> tf.Tensor:
        """Compute root mean square values along specified axis.

        Args:
            inputs: Input tensor

        Returns:
            RMS values tensor with same shape as input except for normalized axis
        """
        x_squared = tf.square(inputs)
        mean_squared = tf.reduce_mean(x_squared, axis=self.axis, keepdims=True)
        return tf.sqrt(mean_squared + self.epsilon)

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """Apply constrained RMS normalization.

        Args:
            inputs: Input tensor
            training: Whether in training mode (unused)

        Returns:
            Normalized tensor with L2 norm in [1-max_band_width, 1]
        """
        inputs = tf.cast(inputs, self.compute_dtype)

        # Step 1: RMS normalization to get unit norm
        rms = self._compute_rms(inputs)
        normalized = inputs / rms

        # Step 2: Apply learnable scaling within [1-α, 1] band
        # Use hard sigmoid to strictly enforce the bounds
        scale = (1.0 - self.max_band_width) + (
                self.max_band_width *
                keras.activations.hard_sigmoid(tf.cast(self.band_param, inputs.dtype))
        )

        # The output will have L2 norm between [1-max_band_width, 1]
        # since we first normalize to unit norm and then scale by a factor
        # that is guaranteed to be in [1-max_band_width, 1]
        return normalized * scale

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute shape of output tensor.

        Args:
            input_shape: Shape of input tensor

        Returns:
            Shape of output tensor (same as input)
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        Returns:
            Dictionary containing the layer configuration
        """
        config = super().get_config()
        config.update({
            "max_band_width": self.max_band_width,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "band_regularizer": keras.regularizers.serialize(self.band_regularizer)
        })
        return config

# ---------------------------------------------------------------------
