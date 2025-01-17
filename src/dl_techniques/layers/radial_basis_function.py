"""
Radial Basis Function (RBF) Neural Network Layer Implementation
============================================================

Theory
------
Radial Basis Function Networks (RBFNs) are a class of neural networks that use radial basis functions
as activation functions. They have several key properties that make them useful for various tasks:

1. Universal Approximation: RBFNs can approximate any continuous function on a compact domain
   to arbitrary precision, given sufficient hidden units.

2. Local Reception: Unlike sigmoid-based networks, RBF units respond maximally to inputs near
   their centers and decrease monotonically with distance. This creates localized receptive fields.

3. Geometric Interpretation: Each RBF unit defines a Gaussian "bump" in the input space, centered
   at its center vector. The network output is a weighted sum of these bumps.

Mathematical Foundation
---------------------
For an input vector x, each RBF unit i computes:
    φᵢ(x) = exp(-γᵢ ||x - cᵢ||²)
where:
    - cᵢ is the center vector for unit i
    - γᵢ is the width parameter (inverse of variance)
    - ||·||² denotes squared Euclidean distance

The complete network computation is:
    y(x) = Σ wᵢφᵢ(x)
where wᵢ are the output layer weights.

Training Process
--------------
The network learns through three main parameters:
1. Centers (cᵢ): Determine where each RBF unit is positioned in input space
2. Widths (γᵢ): Control how quickly the response falls off with distance
3. Output weights (wᵢ): Determine how RBF responses are combined

Implementation Details
--------------------
This implementation:
- Uses fully trainable centers and widths
- Employs efficient broadcasting for distance calculations
- Ensures numerical stability through proper initialization
- Supports both training and inference phases
- Implements proper serialization for model saving/loading

Common Applications
-----------------
1. Function Approximation
2. Time Series Prediction
3. Pattern Recognition
4. Control Systems
5. Signal Processing
6. Interpolation Problems

Advantages
---------
1. Fast training compared to backprop-only networks
2. Good generalization
3. Interpretable hidden layer representations
4. Insensitive to order of training data
5. Single hidden layer sufficient for most applications

Limitations
----------
1. Curse of dimensionality in high-dimensional spaces
2. May require more units than sigmoid networks
3. Sensitive to center initialization
4. Limited extrapolation capabilities

References
---------
1. Broomhead, D. S., & Lowe, D. (1988). Radial basis functions, multi-variable
   functional interpolation and adaptive networks.
2. Moody, J., & Darken, C. J. (1989). Fast learning in networks of locally-tuned
   processing units.
3. Powell, M. J. D. (1987). Radial basis functions for multivariable interpolation:
   a review.
"""

import keras
import tensorflow as tf
from keras.api.layers import Layer
from typing import Union, Tuple, Any

# ---------------------------------------------------------------------


class RBFLayer(Layer):
    """
    Custom Radial Basis Function Layer implementation.

    This layer implements RBF units where the activation of each unit
    is computed using a Gaussian kernel. The centers and widths of the
    RBF units are learned during training.

    Args:
        units (int): Number of RBF units in the layer
        gamma (float, optional): Initial value for the width parameter.
            Defaults to 1.0
        initializer (str, optional): Initializer for the centers.
            Defaults to 'glorot_uniform'

    Attributes:
        units (int): Number of RBF units
        gamma (float): Width parameter for RBF units
        centers (tf.Variable): Centers of RBF units
        widths (tf.Variable): Widths of RBF units
    """

    def __init__(
            self,
            units: int,
            gamma: float = 1.0,
            initializer: str = 'glorot_uniform',
            **kwargs: Any
    ) -> None:
        """Initialize the RBF layer."""
        if units < 1:
            raise ValueError(f"Number of units must be positive, got {units}")
        if gamma <= 0:
            raise ValueError(f"Gamma must be positive, got {gamma}")

        super().__init__(**kwargs)
        self.units = units
        self.gamma = gamma
        self.initializer = keras.initializers.get(initializer)
        self.centers = None
        self.widths = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Build the layer by creating the trainable weights.

        Args:
            input_shape: Shape of the input tensor

        Raises:
            ValueError: If input shape has less than 2 dimensions
        """
        if len(input_shape) < 2:
            raise ValueError(f"Input shape must have at least 2 dimensions, got {len(input_shape)}")

        input_dim = input_shape[-1]

        self.centers = self.add_weight(
            name='centers',
            shape=(self.units, input_dim),
            initializer=self.initializer,
            trainable=True,
            dtype=tf.float32
        )

        self.widths = self.add_weight(
            name='widths',
            shape=(self.units,),
            initializer=tf.constant_initializer(self.gamma),
            trainable=True,
            dtype=tf.float32,
            constraint=keras.constraints.NonNeg()
        )

        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the layer.

        Computes the RBF activation for the input tensor using learned
        centers and widths.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim)

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, units)
        """
        # Expand dimensions for broadcasting
        inputs_expanded = tf.expand_dims(inputs, axis=1)  # (batch_size, 1, input_dim)
        centers_expanded = tf.expand_dims(self.centers, axis=0)  # (1, units, input_dim)

        # Compute squared Euclidean distances
        distances = tf.reduce_sum(
            tf.square(inputs_expanded - centers_expanded),
            axis=-1
        )  # (batch_size, units)

        # Apply RBF activation
        return tf.exp(-self.widths * distances)

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute the shape of the layer's output.

        Args:
            input_shape: Shape of the input tensor

        Returns:
            tuple: Shape of the output tensor
        """
        return input_shape[:-1] + (self.units,)

    def get_config(self) -> dict:
        """
        Get the layer's configuration.

        Returns:
            dict: Configuration dictionary
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'gamma': self.gamma,
            'initializer': keras.initializers.serialize(self.initializer)
        })
        return config

# ---------------------------------------------------------------------
