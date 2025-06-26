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

```
        [Input: Feature Maps]
               ↓
        ┌─────────────────┐
        │   Conv2D Layer  │  ← Kernel size, strides, padding
        └─────────────────┘
               ↓
        ┌─────────────────┐
        │     Reshape     │  ← To [batch, num_capsules*h*w, dim_capsules]
        └─────────────────┘
               ↓
        ┌─────────────────┐
        │  Squash Function│  ← Non-linearity for capsules
        └─────────────────┘
               ↓
[Output: Primary Capsules Vectors]
```

### 2. Routing Capsule Layer
Implements dynamic routing mechanism between capsule layers.

```
    [Input: Lower-level Capsules]
               ↓
┌─────────────────────────────────┐
│      Transformation Matrix      │ ← W_ij transforms u_i to û_j|i
└─────────────────────────────────┘
               ↓
┌─────────────────────────────────┐
│      Initialize Routing         │ ← b_ij = 0
│         Coefficients            │
└─────────────────────────────────┘
               ↓
┌─────────────────────────────────┐
│      Routing Iterations         │
│                                 │
│    ┌───────────────────────┐    │
│    │ Softmax to get c_ij   │    │ ← c_ij = softmax(b_ij)
│    └───────────────────────┘    │
│               ↓                 │
│    ┌───────────────────────┐    │
│    │ Weighted sum to get   │    │ ← s_j = Σ c_ij * û_j|i
│    │ capsule input s_j     │    │
│    └───────────────────────┘    │
│               ↓                 │
│    ┌───────────────────────┐    │
│    │ Squash to get capsule │    │ ← v_j = squash(s_j)
│    │ output v_j            │    │
│    └───────────────────────┘    │
│               ↓                 │
│    ┌───────────────────────┐    │
│    │ Update routing logits │    │ ← b_ij += v_j • û_j|i
│    └───────────────────────┘    │
│                                 │
└─────────────────────────────────┘
               ↓
   [Output: Higher-level Capsules]
```

### 3. Capsule Block
A complete block combining routing capsules with regularization.

```
    [Input: Lower-level Capsules]
               ↓
    ┌─────────────────────────┐
    │     Routing Capsule     │
    └─────────────────────────┘
               ↓
    ┌─────────────────────────┐
    │     Dropout (Opt.)      │
    └─────────────────────────┘
               ↓
    ┌─────────────────────────┐
    │   Layer Norm (Opt.)     │
    └─────────────────────────┘
               ↓
   [Output: Higher-level Capsules]
```

## Squash Function

The squash function is a non-linearity that ensures the length of the output vector is between 0 and 1:

```
             ||s||²
squash(s) = --------- * (s / ||s||)
            1 + ||s||²
```

Where:
- s is the input vector
- ||s|| is the Euclidean norm of s
- The output vector has the same orientation as the input but with length compressed to [0,1]

## Margin Loss

For class k:

```
L_k = T_k * max(0, m⁺ - ||v_k||)² + λ(1-T_k) * max(0, ||v_k|| - m⁻)²
```

Where:
- T_k = 1 if class k is present
- m⁺ = 0.9 (margin for correct class)
- m⁻ = 0.1 (margin for incorrect classes)
- λ = 0.5 (down-weighting factor for absent classes)

## References

Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between capsules.
In Advances in Neural Information Processing Systems (pp. 3856-3866).
"""

import keras
from keras import ops, backend
from typing import Optional, Tuple, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SquashLayer(keras.layers.Layer):
    """Applies squashing non-linearity to vectors (capsules).

    The squashing function ensures that:
    1. Short vectors get shrunk to almost zero length
    2. Long vectors get shrunk to a length slightly below 1
    3. Vector orientation is preserved

    This is commonly used in Capsule Networks to ensure capsule outputs
    have meaningful magnitudes while preserving their directional information.
    The squashing function is defined as:

    .. math::
        \\text{squash}(\\mathbf{v}) = \\frac{||\\mathbf{v}||^2}{1 + ||\\mathbf{v}||^2} \\cdot \\frac{\\mathbf{v}}{||\\mathbf{v}||}

    Args:
        axis: Integer, axis along which to compute the norm. Defaults to -1.
        epsilon: Float, small constant for numerical stability. If None, uses keras.backend.epsilon().
        **kwargs: Additional keyword arguments to pass to the Layer base class.

    Input shape:
        Arbitrary tensor of rank >= 1.

    Output shape:
        Same as input shape.

    Example:
        >>> layer = SquashLayer()
        >>> x = keras.random.normal((32, 10, 16))
        >>> y = layer(x)
        >>> print(y.shape)
        (32, 10, 16)

        >>> # Custom axis for squashing
        >>> layer = SquashLayer(axis=1)
        >>> x = keras.random.normal((32, 10, 16))
        >>> y = layer(x)
        >>> print(y.shape)
        (32, 10, 16)
    """

    def __init__(
        self,
        axis: int = -1,
        epsilon: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon if epsilon is not None else backend.epsilon()

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.debug(f"Initialized SquashLayer with axis={axis}, epsilon={self.epsilon}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Store for serialization
        self._build_input_shape = input_shape

        logger.debug(f"Building SquashLayer with input_shape={input_shape}")

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None) -> Any:
        """Apply squashing non-linearity.

        Args:
            inputs: Input tensor to be squashed.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Squashed vectors with norm between 0 and 1.
        """
        # Compute squared norm along specified axis
        squared_norm = ops.sum(
            ops.square(inputs),
            axis=self.axis,
            keepdims=True
        )

        # Safe norm computation to avoid division by zero
        safe_norm = ops.sqrt(squared_norm + self.epsilon)

        # Compute scale factor: ||v||^2 / (1 + ||v||^2)
        scale = squared_norm / (1.0 + squared_norm)

        # Compute unit vector
        unit_vector = inputs / safe_norm

        # Apply squashing: scale * unit_vector
        return scale * unit_vector

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PrimaryCapsule(keras.layers.Layer):
    """Primary Capsule Layer implementation.

    This layer implements the primary capsule layer which converts regular CNN features
    into capsule format. It consists of a convolutional layer followed by reshaping
    and squashing operations.

    Args:
        num_capsules: Number of primary capsules to create
        dim_capsules: Dimension of each capsule's output vector
        kernel_size: Size of the convolutional kernel
        strides: Stride length for convolution
        padding: Type of padding for convolution ('valid' or 'same')
        kernel_initializer: Initializer for the conv kernel
        kernel_regularizer: Regularizer function for the conv kernel
        use_bias: Whether to include bias terms
        squash_axis: Axis along which to apply squashing operation
        squash_epsilon: Epsilon for numerical stability in squashing
        name: Optional name for the layer
        **kwargs: Additional keyword arguments for the base class
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
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

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

        # Store build input shape for serialization
        self._build_input_shape = None

        # Layers to be initialized in build()
        self.conv = None
        self.squash_layer = None

    def build(self, input_shape) -> None:
        """Build the layer based on input shape.

        Args:
            input_shape: Shape of input tensor
        """
        # Store for serialization
        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 4:  # [batch_size, height, width, channels]
            raise ValueError(f"Expected 4D input shape [batch, height, width, channels], got {input_shape}")

        # Create the convolutional layer
        self.conv = keras.layers.Conv2D(
            filters=self.num_capsules * self.dim_capsules,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            name=f"{self.name}_conv" if self.name else "conv"
        )

        # Build the convolutional layer
        self.conv.build(input_shape)

        # Create squashing layer
        self.squash_layer = SquashLayer(
            axis=self.squash_axis,
            epsilon=self.squash_epsilon,
            name=f"{self.name}_squash" if self.name else "squash"
        )

        super().build(input_shape)

        logger.info(f"Built PrimaryCapsule layer: {self.num_capsules} capsules, "
                   f"{self.dim_capsules} dimensions each")

    def call(self, inputs, training: Optional[bool] = None) -> Any:
        """Forward pass for primary capsule layer.

        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels]
            training: Whether in training mode

        Returns:
            Output tensor of shape [batch_size, num_capsules, dim_capsules]
        """
        batch_size = ops.shape(inputs)[0]

        # Apply convolution
        conv_output = self.conv(inputs, training=training)

        # Reshape to capsule format
        # Calculate dimensions based on the output of the convolutional layer
        h, w = ops.shape(conv_output)[1], ops.shape(conv_output)[2]
        num_spatial_capsules = h * w

        # Reshape to [batch_size, h*w * num_capsules, dim_capsules]
        capsules = ops.reshape(
            conv_output,
            [batch_size, num_spatial_capsules * self.num_capsules, self.dim_capsules]
        )

        # Apply squashing
        return self.squash_layer(capsules, training=training)

    def compute_output_shape(self, input_shape) -> Tuple[Optional[int], ...]:
        """Compute output shape based on input shape.

        Args:
            input_shape: Shape of input tensor

        Returns:
            Shape of output tensor
        """
        # Compute the shape after convolution
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        h, w = conv_output_shape[1], conv_output_shape[2]
        num_spatial_capsules = h * w

        # Shape after reshaping and squashing
        return tuple([
            input_shape[0],  # batch size
            num_spatial_capsules * self.num_capsules,
            self.dim_capsules
        ])

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration
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

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RoutingCapsule(keras.layers.Layer):
    """Capsule layer with dynamic routing between capsules.

    Implements the iterative dynamic routing mechanism that determines how lower-level
    capsules should contribute to higher-level capsules. This is the key innovation
    of Capsule Networks that replaces max-pooling with a more sophisticated routing
    mechanism based on agreement between predictions and outputs.

    ## Dynamic Routing Algorithm

    The routing process iteratively refines the connection strengths between input
    and output capsules through an agreement mechanism:

    1. **Initialize**: All routing coefficients start equal (uniform routing)
    2. **For each iteration**:
       - Convert routing logits to coefficients via softmax
       - Compute weighted sum of input predictions
       - Apply squashing non-linearity to get output
       - Measure agreement between predictions and outputs
       - Update routing logits based on agreement
    3. **Converge**: Strong agreements strengthen connections, weak ones weaken

    ## Mathematical Formulation

    For each routing iteration:

    .. math::
        c_{ij} = \\text{softmax}(b_{ij}) \\quad \\text{(routing coefficients)}

    .. math::
        s_j = \\sum_i c_{ij} \\hat{u}_{j|i} \\quad \\text{(weighted sum)}

    .. math::
        v_j = \\text{squash}(s_j) \\quad \\text{(output capsule)}

    .. math::
        b_{ij} \\leftarrow b_{ij} + \\hat{u}_{j|i} \\cdot v_j \\quad \\text{(update logits)}

    Where:
    - :math:`c_{ij}`: routing coefficient from input capsule i to output capsule j
    - :math:`b_{ij}`: routing logit (accumulated agreement)
    - :math:`\\hat{u}_{j|i}`: prediction from input capsule i for output capsule j
    - :math:`v_j`: final output of capsule j
    - :math:`s_j`: weighted input to capsule j

    ## Intuitive Understanding

    Consider face recognition:
    - **Lower capsules**: Detect eyes, nose, mouth features
    - **Higher capsules**: Detect complete faces
    - **Without routing**: All features contribute equally to face detection
    - **With routing**:
        - Iteration 1: Eyes strongly predict "face present", mouth prediction is weaker
        - Iteration 2: Eye→face connections strengthened, mouth→face connections weakened
        - Iteration 3: Stable routing where relevant features dominate face detection

    ## Why Multiple Iterations?

    - **Iteration 1**: Uniform routing, initial outputs computed
    - **Iteration 2**: Routing refined based on agreement, better connections strengthened
    - **Iteration 3**: Further refinement, typically reaches stable routing pattern
    - **3+ iterations**: Diminishing returns, computational cost increases linearly

    The iterative process creates a competitive mechanism where input capsules compete
    to contribute to output capsules based on prediction quality, leading to more
    interpretable and hierarchical representations.

    Args:
        num_capsules: Number of output capsules to produce.
        dim_capsules: Dimension of each output capsule's vector.
        routing_iterations: Number of dynamic routing iterations. The paper recommends 3
            iterations as a good balance between performance and computational cost.
            More iterations provide diminishing returns.
        kernel_initializer: Initializer for the transformation matrices W that transform
            input capsule predictions to output capsule space.
        kernel_regularizer: Regularizer function applied to the transformation matrices.
        use_bias: Whether to include bias terms in the routing computation.
        squash_axis: Axis along which to apply the squashing non-linearity. Default is -2
            to squash along the capsule dimension.
        squash_epsilon: Small constant for numerical stability in the squashing function.
        name: Optional name for the layer.
        **kwargs: Additional keyword arguments passed to the Layer base class.

    Input shape:
        3D tensor with shape: ``(batch_size, num_input_capsules, input_dim_capsules)``

    Output shape:
        3D tensor with shape: ``(batch_size, num_capsules, dim_capsules)``

    Example:
        >>> # Route from 1152 primary capsules (8D) to 10 digit capsules (16D)
        >>> routing_layer = RoutingCapsule(
        ...     num_capsules=10,
        ...     dim_capsules=16,
        ...     routing_iterations=3
        ... )
        >>> primary_caps = keras.random.normal((32, 1152, 8))
        >>> digit_caps = routing_layer(primary_caps)
        >>> print(digit_caps.shape)
        (32, 10, 16)

    References:
        Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between capsules.
        In Advances in Neural Information Processing Systems (pp. 3856-3866).
    """

    def __init__(
        self,
        num_capsules: int,
        dim_capsules: int,
        routing_iterations: int = 3,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        squash_axis: int = -2,  # Different default for routing capsules
        squash_epsilon: Optional[float] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

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

        # Store build input shape for serialization
        self._build_input_shape = None

        # These will be set in build()
        self.W = None
        self.bias = None
        self.num_input_capsules = None
        self.input_dim_capsules = None
        self.squash_layer = None

    def build(self, input_shape) -> None:
        """Build layer weights based on input shape.

        Args:
            input_shape: Shape of input tensor [batch_size, num_input_capsules, input_dim_capsules]
        """
        # Store for serialization
        self._build_input_shape = input_shape

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

        # Create squashing layer
        self.squash_layer = SquashLayer(
            axis=self.squash_axis,
            epsilon=self.squash_epsilon,
            name=f"{self.name}_squash" if self.name else "squash"
        )

        super().build(input_shape)

        logger.info(f"Built RoutingCapsule layer: {self.num_input_capsules} -> {self.num_capsules} capsules, "
                   f"{self.routing_iterations} routing iterations")

    def call(self, inputs, training: Optional[bool] = None) -> Any:
        """Forward pass implementing dynamic routing between capsules.

        Args:
            inputs: Input tensor of shape [batch_size, num_input_capsules, input_dim_capsules]
            training: Whether in training mode

        Returns:
            Output tensor of shape [batch_size, num_capsules, dim_capsules]
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

    def compute_output_shape(self, input_shape) -> Tuple[Optional[int], ...]:
        """Compute output shape based on input shape.

        Args:
            input_shape: Shape of input tensor

        Returns:
            Shape of output tensor
        """
        return tuple([input_shape[0], self.num_capsules, self.dim_capsules])

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration
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

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CapsuleBlock(keras.layers.Layer):
    """
    A complete capsule block with optional dropout and normalization.

    This block combines a RoutingCapsule layer with optional regularization
    techniques like dropout and layer normalization.

    Args:
        num_capsules: Number of output capsules
        dim_capsules: Dimension of each output capsule's vector
        routing_iterations: Number of routing iterations
        dropout_rate: Dropout rate (0.0 means no dropout)
        use_layer_norm: Whether to use layer normalization
        kernel_initializer: Initializer for the transformation matrices
        kernel_regularizer: Regularizer function for the transformation matrices
        use_bias: Whether to use biases in routing
        squash_axis: Axis along which to apply squashing operation
        squash_epsilon: Epsilon for numerical stability in squashing
        name: Optional name for the layer
        **kwargs: Additional keyword arguments for the base class
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
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

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

        # Store build input shape for serialization
        self._build_input_shape = None

        # Create layers (will be initialized in build)
        self.capsule_layer = None
        self.dropout = None
        self.layer_norm = None

    def build(self, input_shape) -> None:
        """Build the layer based on input shape.

        Args:
            input_shape: Shape of input tensor
        """
        # Store for serialization
        self._build_input_shape = input_shape

        # Create the routing capsule layer
        self.capsule_layer = RoutingCapsule(
            num_capsules=self.num_capsules,
            dim_capsules=self.dim_capsules,
            routing_iterations=self.routing_iterations,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            squash_axis=self.squash_axis,
            squash_epsilon=self.squash_epsilon,
            name=f"{self.name}_capsules" if self.name else None
        )

        # Build the capsule layer
        self.capsule_layer.build(input_shape)

        # Optional regularization layers
        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(self.dropout_rate)

        if self.use_layer_norm:
            self.layer_norm = keras.layers.LayerNormalization(
                axis=-1,
                name=f"{self.name}_norm" if self.name else None
            )
            # Build layer normalization if used
            output_shape = self.capsule_layer.compute_output_shape(input_shape)
            self.layer_norm.build(output_shape)

        super().build(input_shape)

        logger.info(f"Built CapsuleBlock: {self.num_capsules} capsules, "
                   f"dropout={self.dropout_rate}, layer_norm={self.use_layer_norm}")

    def call(self, inputs, training: Optional[bool] = None) -> Any:
        """Forward pass through the capsule block.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Processed output tensor
        """
        # Process through capsule layer
        x = self.capsule_layer(inputs, training=training)

        # Apply optional regularization
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x

    def compute_output_shape(self, input_shape) -> Tuple[Optional[int], ...]:
        """Compute output shape based on input shape.

        Args:
            input_shape: Shape of input tensor

        Returns:
            Shape of output tensor
        """
        return self.capsule_layer.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration
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

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------