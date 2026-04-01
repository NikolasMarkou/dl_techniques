"""
# Capsule Networks (CapsNet) Implementation

A comprehensive Keras-based implementation of Capsule Networks as proposed by
Sabour et al. in "Dynamic Routing Between Capsules" (2017).

## Architecture Overview

```
Input --> Conv Layers --> Primary Capsule --> CapsuleBlock
                                               |
                                               +--> RoutingCapsule
                                               |      |
                                               |      +--> Dynamic Routing
                                               |
                                               +--> (Optional)
                                                      |
                                                      +--> Dropout
                                                      |
                                                      +--> LayerNorm
```

## Core Components

### 1. Primary Capsule Layer
Transforms traditional CNN features into capsule vectors.

### 2. Routing Capsule Layer
Implements dynamic routing mechanism between capsule layers.

### 3. Capsule Block
A complete block combining routing capsules with regularization.

## Dynamic Routing Algorithm

The routing process iteratively refines the connection strengths between input
and output capsules through an agreement mechanism.

## References

Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between capsules.
In Advances in Neural Information Processing Systems (pp. 3856-3866).
"""

import keras
from keras import ops
from typing import Optional, Tuple, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .activations.squash import SquashLayer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PrimaryCapsule(keras.layers.Layer):
    """Primary capsule layer that converts CNN features into capsule vectors.

    Applies a Conv2D with ``filters = num_capsules * dim_capsules``, reshapes
    the output into capsule format ``(B, N, D)``, and applies the squash
    non-linearity ``v = (||s||^2 / (1 + ||s||^2)) * (s / ||s||)`` to produce
    unit-bounded capsule activations whose length represents detection
    probability and orientation encodes instantiation parameters.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input (B, H, W, C)                  │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  Conv2D                              │
        │  filters = num_caps * dim_caps       │
        │  kernel_size, strides, padding       │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  Reshape ──► (B, N*num_caps, D)      │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  Squash (axis=-1)                    │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  Output (B, total_capsules, D)       │
        └──────────────────────────────────────┘

    :param num_capsules: Number of primary capsules per spatial location.
    :type num_capsules: int
    :param dim_capsules: Dimension of each capsule vector.
    :type dim_capsules: int
    :param kernel_size: Size of the convolutional kernel.
    :type kernel_size: Union[int, Tuple[int, int]]
    :param strides: Stride length for convolution. Defaults to 1.
    :type strides: Union[int, Tuple[int, int]]
    :param padding: Padding type (``'valid'`` or ``'same'``). Defaults to ``'valid'``.
    :type padding: str
    :param kernel_initializer: Initializer for the conv kernel.
        Defaults to ``'he_normal'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for the conv kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param use_bias: Whether to include bias terms in convolution. Defaults to ``True``.
    :type use_bias: bool
    :param squash_axis: Axis for squashing. Defaults to -1.
    :type squash_axis: int
    :param squash_epsilon: Epsilon for numerical stability in squashing.
    :type squash_epsilon: Optional[float]
    :param kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        num_capsules: int,
        dim_capsules: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = 1,
        padding: str = "valid",
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        squash_axis: int = -1,
        squash_epsilon: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_capsules <= 0:
            raise ValueError(f"num_capsules must be positive, got {num_capsules}")
        if dim_capsules <= 0:
            raise ValueError(f"dim_capsules must be positive, got {dim_capsules}")
        if padding not in ["valid", "same"]:
            raise ValueError(f"padding must be one of 'valid' or 'same', got {padding}")

        # Store configuration
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias
        self.squash_axis = squash_axis
        self.squash_epsilon = squash_epsilon

        # CREATE sub-layers in __init__ (modern Keras 3 pattern)
        self.conv = keras.layers.Conv2D(
            filters=self.num_capsules * self.dim_capsules,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            name="primary_conv"
        )

        self.squash_layer = SquashLayer(
            axis=self.squash_axis,
            epsilon=self.squash_epsilon,
            name="primary_squash"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer based on input shape.

        :param input_shape: Shape of input tensor as tuple.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Validate input shape
        if len(input_shape) != 4:  # [batch_size, height, width, channels]
            raise ValueError(f"Expected 4D input shape [batch, height, width, channels], got {input_shape}")

        # BUILD sub-layers explicitly (critical for serialization)
        self.conv.build(input_shape)

        # Compute conv output shape to build squash layer
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        h, w = conv_output_shape[1], conv_output_shape[2]
        num_spatial_capsules = h * w

        # Build squash layer with reshaped dimensions
        squash_input_shape = (
            input_shape[0],
            num_spatial_capsules * self.num_capsules,
            self.dim_capsules
        )
        self.squash_layer.build(squash_input_shape)

        super().build(input_shape)

        logger.info(f"Built PrimaryCapsule layer: {self.num_capsules} capsules, "
                   f"{self.dim_capsules} dimensions each")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass for primary capsule layer.

        :param inputs: Input tensor of shape ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, total_capsules, dim_capsules)``.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(inputs)[0]

        # Apply convolution
        conv_output = self.conv(inputs, training=training)

        # Reshape to capsule format
        # Calculate dimensions based on the output of the convolutional layer
        h, w = ops.shape(conv_output)[1], ops.shape(conv_output)[2]
        num_spatial_capsules = h * w

        # Reshape to [batch_size, num_spatial_capsules * num_capsules, dim_capsules]
        capsules = ops.reshape(
            conv_output,
            [batch_size, num_spatial_capsules * self.num_capsules, self.dim_capsules]
        )

        # Apply squashing
        return self.squash_layer(capsules, training=training)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape based on input shape.

        :param input_shape: Shape of input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Shape of output tensor as tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        # Compute the shape after convolution
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        h, w = conv_output_shape[1], conv_output_shape[2]
        num_spatial_capsules = h * w

        # Shape after reshaping and squashing
        return (
            input_shape[0],  # batch size
            num_spatial_capsules * self.num_capsules,
            self.dim_capsules
        )

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "squash_axis": self.squash_axis,
            "squash_epsilon": self.squash_epsilon,
        })
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RoutingCapsule(keras.layers.Layer):
    """Capsule layer with iterative dynamic routing between capsules.

    Implements the dynamic routing mechanism from Sabour et al. (2017).
    Input capsules are transformed via learned matrices ``W`` to produce
    prediction vectors ``u_hat = W * u_i``. Routing logits ``b_ij`` are
    iteratively updated by the agreement ``u_hat . v_j`` between predictions
    and output capsules. Routing coefficients ``c_ij = softmax(b_ij)``
    weight the predictions to produce ``s_j = sum(c_ij * u_hat)``, which
    is squashed to yield the output capsule ``v_j``.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────┐
        │  Input (B, N_in, D_in)                 │
        └──────────────────┬─────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │  Transform: u_hat = W * u_i            │
        │  W: (1, N_in, N_out, D_out, D_in)     │
        └──────────────────┬─────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │  ┌──── Routing Iteration ────────────┐ │
        │  │ c = softmax(b)                    │ │
        │  │ s = sum(c * u_hat) + bias         │ │
        │  │ v = squash(s)                     │ │
        │  │ b += u_hat . v  (agreement)       │ │
        │  └──── repeat K times ───────────────┘ │
        └──────────────────┬─────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │  Output (B, N_out, D_out)              │
        └────────────────────────────────────────┘

    :param num_capsules: Number of output capsules. Must be positive.
    :type num_capsules: int
    :param dim_capsules: Dimension of each output capsule vector. Must be positive.
    :type dim_capsules: int
    :param routing_iterations: Number of dynamic routing iterations. Defaults to 3.
    :type routing_iterations: int
    :param kernel_initializer: Initializer for transformation matrices ``W``.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for transformation matrices.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param use_bias: Whether to include bias in routing. Defaults to ``True``.
    :type use_bias: bool
    :param squash_axis: Axis for squashing non-linearity. Defaults to -2.
    :type squash_axis: int
    :param squash_epsilon: Epsilon for numerical stability in squashing.
    :type squash_epsilon: Optional[float]
    :param kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        num_capsules: int,
        dim_capsules: int,
        routing_iterations: int = 3,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        squash_axis: int = -2,
        squash_epsilon: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_capsules <= 0:
            raise ValueError(f"num_capsules must be positive, got {num_capsules}")
        if dim_capsules <= 0:
            raise ValueError(f"dim_capsules must be positive, got {dim_capsules}")
        if routing_iterations <= 0:
            raise ValueError(f"routing_iterations must be positive, got {routing_iterations}")

        # Store configuration
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routing_iterations = routing_iterations
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias
        self.squash_axis = squash_axis
        self.squash_epsilon = squash_epsilon

        # CREATE squashing layer in __init__ (modern Keras 3 pattern)
        self.squash_layer = SquashLayer(
            axis=self.squash_axis,
            epsilon=self.squash_epsilon,
            name="routing_squash"
        )

        # Weight variables - created in build()
        self.W = None
        self.bias = None

        # Shape attributes - set in build()
        self.num_input_capsules = None
        self.input_dim_capsules = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build layer weights based on input shape.

        :param input_shape: Shape of input tensor ``(B, N_in, D_in)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape [batch, num_capsules, dim_capsules], got {input_shape}")

        # Extract dimensions from input shape
        self.num_input_capsules = input_shape[1]
        self.input_dim_capsules = input_shape[2]

        # Create weight matrix for transformations between input and output capsules
        # Shape: [1, num_input_capsules, num_capsules, dim_capsules, input_dim_capsules]
        self.W = self.add_weight(
            shape=[1, self.num_input_capsules, self.num_capsules,
                   self.dim_capsules, self.input_dim_capsules],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="capsule_transformation_weights"
        )

        # Create bias term if requested
        if self.use_bias:
            self.bias = self.add_weight(
                shape=[1, 1, self.num_capsules, self.dim_capsules, 1],
                initializer="zeros",
                trainable=True,
                name="capsule_bias"
            )

        # BUILD squashing layer - it needs output shape
        squash_input_shape = (input_shape[0], 1, self.num_capsules, self.dim_capsules, 1)
        self.squash_layer.build(squash_input_shape)

        super().build(input_shape)

        logger.info(f"Built RoutingCapsule layer: {self.num_input_capsules} -> {self.num_capsules} capsules, "
                   f"{self.routing_iterations} routing iterations")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass implementing dynamic routing between capsules.

        :param inputs: Input tensor of shape ``(B, N_in, D_in)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, num_capsules, dim_capsules)``.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(inputs)[0]

        # Prepare inputs by expanding dimensions for matrix multiplication
        # [batch_size, num_input_capsules, input_dim_capsules] ->
        # [batch_size, num_input_capsules, 1, 1, input_dim_capsules]
        inputs_expanded = ops.expand_dims(ops.expand_dims(inputs, axis=-1), axis=2)

        # Tile inputs for efficient broadcasting with weights
        # [batch_size, num_input_capsules, 1, 1, input_dim_capsules] ->
        # [batch_size, num_input_capsules, num_capsules, 1, input_dim_capsules]
        inputs_tiled = ops.tile(
            inputs_expanded,
            [1, 1, self.num_capsules, 1, 1]
        )

        # Calculate predicted output vectors (u_hat) through transformation
        # [1, num_input_capsules, num_capsules, dim_capsules, input_dim_capsules] *
        # [batch_size, num_input_capsules, num_capsules, 1, input_dim_capsules] ->
        # [batch_size, num_input_capsules, num_capsules, dim_capsules, 1]
        u_hat = ops.matmul(self.W, inputs_tiled)

        # Initialize routing logits (b) to zero
        b = ops.zeros([batch_size, self.num_input_capsules, self.num_capsules, 1, 1])

        # Perform iterative dynamic routing
        for i in range(self.routing_iterations):
            # Convert logits to routing weights using softmax
            # Shape: [batch_size, num_input_capsules, num_capsules, 1, 1]
            c = keras.activations.softmax(b, axis=2)

            # Weighted sum of predictions to get capsule outputs
            # [batch_size, num_input_capsules, num_capsules, dim_capsules, 1] *
            # [batch_size, num_input_capsules, num_capsules, 1, 1] ->
            # [batch_size, 1, num_capsules, dim_capsules, 1]
            s = ops.sum(c * u_hat, axis=1, keepdims=True)

            # Add bias if requested
            if self.use_bias and self.bias is not None:
                s += self.bias

            # Apply squashing non-linearity
            v = self.squash_layer(s, training=training)

            # For all but the last iteration, update routing logits
            if i < self.routing_iterations - 1:
                # Tile output capsules to match input capsules for agreement calculation
                # [batch_size, 1, num_capsules, dim_capsules, 1] ->
                # [batch_size, num_input_capsules, num_capsules, dim_capsules, 1]
                v_tiled = ops.tile(v, [1, self.num_input_capsules, 1, 1, 1])

                # Calculate agreement between predictions and outputs
                # [batch_size, num_input_capsules, num_capsules, dim_capsules, 1] *
                # [batch_size, num_input_capsules, num_capsules, dim_capsules, 1] ->
                # [batch_size, num_input_capsules, num_capsules, 1, 1]
                agreement = ops.sum(
                    u_hat * v_tiled,
                    axis=-2,
                    keepdims=True
                )

                # Update routing logits based on agreement
                b += agreement

        # Final output: shape [batch_size, num_capsules, dim_capsules]
        return ops.reshape(v, (batch_size, self.num_capsules, self.dim_capsules))

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape based on input shape.

        :param input_shape: Shape of input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Shape of output tensor as tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return (input_shape[0], self.num_capsules, self.dim_capsules)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules,
            "routing_iterations": self.routing_iterations,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "squash_axis": self.squash_axis,
            "squash_epsilon": self.squash_epsilon,
        })
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CapsuleBlock(keras.layers.Layer):
    """Complete capsule block combining dynamic routing with optional regularization.

    Wraps a ``RoutingCapsule`` layer with optional dropout and layer
    normalization to create a stackable capsule processing unit. The processing
    flow is ``x = RoutingCapsule(inputs) -> Dropout -> LayerNorm``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input (B, N_in, D_in)               │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  RoutingCapsule                      │
        │  W transform ──► routing ──► squash  │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  [Optional] Dropout                  │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  [Optional] LayerNormalization       │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  Output (B, N_out, D_out)            │
        └──────────────────────────────────────┘

    :param num_capsules: Number of output capsules. Must be positive.
    :type num_capsules: int
    :param dim_capsules: Dimension of each output capsule vector. Must be positive.
    :type dim_capsules: int
    :param routing_iterations: Number of routing iterations. Defaults to 3.
    :type routing_iterations: int
    :param dropout_rate: Dropout rate in ``[0.0, 1.0)``. Defaults to 0.0.
    :type dropout_rate: float
    :param use_layer_norm: Whether to use layer normalization. Defaults to ``False``.
    :type use_layer_norm: bool
    :param kernel_initializer: Initializer for transformation matrices.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for transformation matrices.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param use_bias: Whether to use biases in routing. Defaults to ``True``.
    :type use_bias: bool
    :param squash_axis: Axis for squashing. Defaults to -2.
    :type squash_axis: int
    :param squash_epsilon: Epsilon for numerical stability in squashing.
    :type squash_epsilon: Optional[float]
    :param kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        num_capsules: int,
        dim_capsules: int,
        routing_iterations: int = 3,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        squash_axis: int = -2,
        squash_epsilon: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dropout_rate < 0.0 or dropout_rate >= 1.0:
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")
        if not isinstance(use_layer_norm, bool):
            raise TypeError(f"use_layer_norm must be boolean, got {type(use_layer_norm)}")

        # Store configuration parameters
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routing_iterations = routing_iterations
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias
        self.squash_axis = squash_axis
        self.squash_epsilon = squash_epsilon

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern)
        self.capsule_layer = RoutingCapsule(
            num_capsules=self.num_capsules,
            dim_capsules=self.dim_capsules,
            routing_iterations=self.routing_iterations,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            squash_axis=self.squash_axis,
            squash_epsilon=self.squash_epsilon,
            name="block_capsules"
        )

        # Optional regularization layers
        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(self.dropout_rate, name="block_dropout")
        else:
            self.dropout = None

        if self.use_layer_norm:
            self.layer_norm = keras.layers.LayerNormalization(
                axis=-1,
                name="block_norm"
            )
        else:
            self.layer_norm = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer based on input shape.

        :param input_shape: Shape of input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # BUILD all sub-layers explicitly (critical for serialization)
        self.capsule_layer.build(input_shape)

        # Compute output shape for building optional layers
        output_shape = self.capsule_layer.compute_output_shape(input_shape)

        if self.dropout is not None:
            self.dropout.build(output_shape)

        if self.layer_norm is not None:
            self.layer_norm.build(output_shape)

        super().build(input_shape)

        logger.info(f"Built CapsuleBlock: {self.num_capsules} capsules, "
                   f"dropout={self.dropout_rate}, layer_norm={self.use_layer_norm}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the capsule block.

        :param inputs: Input tensor of shape ``(B, N_in, D_in)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, num_capsules, dim_capsules)``.
        :rtype: keras.KerasTensor
        """
        # Process through capsule layer
        x = self.capsule_layer(inputs, training=training)

        # Apply optional regularization
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        if self.layer_norm is not None:
            x = self.layer_norm(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape based on input shape.

        :param input_shape: Shape of input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Shape of output tensor as tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return self.capsule_layer.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules,
            "routing_iterations": self.routing_iterations,
            "dropout_rate": self.dropout_rate,
            "use_layer_norm": self.use_layer_norm,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "squash_axis": self.squash_axis,
            "squash_epsilon": self.squash_epsilon,
        })
        return config

# ---------------------------------------------------------------------