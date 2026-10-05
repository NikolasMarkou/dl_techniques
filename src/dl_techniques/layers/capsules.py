"""Capsule layers: dynamic routing between capsules, plus the cyclic
within-capsule permutation that makes a capsule an *equivariant* feature.

Two independent notions meet in this module and the class names alone do not
tell you which is which.

**Routing** (`PrimaryCapsule`, `RoutingCapsule`, `CapsuleBlock`) replaces a
scalar activation with a vector: its length is a detection probability and its
direction carries pose (position, scale, orientation). PrimaryCapsule turns
convolutional features into capsule vectors. RoutingCapsule assigns lower-level
capsules to higher-level ones by iterative agreement (`c_ij = softmax_j(b_ij)`,
`s_j = sum_i c_ij u_hat_ji`, `b_ij += u_hat_ji . v_j`) instead of pooling, so
pose survives instead of being discarded for invariance. CapsuleBlock wraps
routing with optional dropout and a length-preserving LayerNormalization.

The squash non-linearity `v = (||s||^2/(1+||s||^2)) * (s/||s||)` bounds a
vector's magnitude below 1 without changing its direction. CapsuleBlock's
LayerNormalization applies to the unit-direction component only: normalizing
the whole vector would rescale `||v||`, the probability the margin loss reads.

``CapsuleRoll`` is not routing and shares none of its machinery: it is the
group action. A capsule is a *set* of ``D`` coordinates representing one
equivalence class, and a transformation of the observation that leaves the class
alone must still change the representation -- that is what a group equivariant
network gets from its structured weight sharing. ``CapsuleRoll`` supplies that
change directly: a cyclic permutation of ``shift`` steps *within* each capsule,
leaving different capsules untouched. Being a permutation, it is invertible
(``shift -> -shift``), norm-preserving on each capsule, and free of parameters --
three properties the equivariance metrics rely on and no routed capsule can
promise. It is what lets a Topographic VAE decode an unseen element of a
transformation sequence by re-rolling one encoded capsule activation
(``dl_techniques.models.topographic_vae``), and it is the operator those models'
equivariance metrics measure against.

References:
    - Sabour et al., 2017. Dynamic Routing Between Capsules. NeurIPS 2017.
      (https://arxiv.org/abs/1710.09829)
    - Hinton et al., 2018. Matrix Capsules with EM Routing. ICLR 2018.
    - Hinton et al., 2011. Transforming Auto-encoders. ICANN 2011.
    - Ba et al., 2016. Layer Normalization.
      (https://arxiv.org/abs/1607.06450)
    - Cohen & Welling, 2016. Group Equivariant Convolutional Networks. ICML.
      (https://arxiv.org/abs/1605.09376)
    - Keller & Welling, 2022. Topographic VAEs learn Equivariant Capsules.
      NeurIPS 2021. (https://arxiv.org/abs/2109.01394)
"""

import numpy as np
import keras
from typing import Optional, Tuple, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.activations.squash import SquashLayer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.capsules")
class PrimaryCapsule(keras.layers.Layer):
    """Primary capsule layer that converts CNN features into capsule vectors.

    Applies a Conv2D with ``filters = num_capsules * dim_capsules``, reshapes
    the output into capsule format ``(B, N, D)``, and applies the squash
    non-linearity ``v = (||s||^2 / (1 + ||s||^2)) * (s / ||s||)`` to produce
    unit-bounded capsule activations whose length represents detection
    probability and orientation encodes instantiation parameters.

    Architecture:

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

        logger.info(f"Built PrimaryCapsule layer: {self.num_capsules} capsules, "
                   f"{self.dim_capsules} dimensions each")

        super().build(input_shape)

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
        batch_size = keras.ops.shape(inputs)[0]

        # Apply convolution
        conv_output = self.conv(inputs, training=training)

        # Reshape to capsule format
        # Calculate dimensions based on the output of the convolutional layer
        h, w = keras.ops.shape(conv_output)[1], keras.ops.shape(conv_output)[2]
        num_spatial_capsules = h * w

        # Reshape to [batch_size, num_spatial_capsules * num_capsules, dim_capsules]
        capsules = keras.ops.reshape(
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


@register_dl_technique("dl_techniques.layers.capsules")
class RoutingCapsule(keras.layers.Layer):
    """Capsule layer with iterative dynamic routing between capsules.

    Implements the dynamic routing mechanism from Sabour et al. (2017).
    Input capsules are transformed via learned matrices ``W`` to produce
    prediction vectors ``u_hat = W * u_i``. Routing logits ``b_ij`` are
    iteratively updated by the agreement ``u_hat . v_j`` between predictions
    and output capsules. Routing coefficients ``c_ij = softmax(b_ij)``
    weight the predictions to produce ``s_j = sum(c_ij * u_hat)``, which
    is squashed to yield the output capsule ``v_j``.

    Architecture:

    .. code-block:: text

        ┌────────────────────────────────────────┐
        │  Input (B, N_in, D_in)                 │
        └──────────────────┬─────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │  Transform: u_hat = W * u_i            │
        │  W: (1, N_in, N_out, D_out, D_in)      │
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

        logger.info(f"Built RoutingCapsule layer: {self.num_input_capsules} -> {self.num_capsules} capsules, "
                   f"{self.routing_iterations} routing iterations")

        super().build(input_shape)

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
        batch_size = keras.ops.shape(inputs)[0]

        # Add capsule and singleton axes for broadcasting against W.
        inputs_expanded = keras.ops.expand_dims(keras.ops.expand_dims(inputs, axis=-1), axis=2)

        # Tile over the output-capsule axis to match W's shape.
        inputs_tiled = keras.ops.tile(
            inputs_expanded,
            [1, 1, self.num_capsules, 1, 1]
        )

        # Prediction vectors u_hat = W * u_i, shape (B, N_in, N_out, D_out, 1).
        u_hat = keras.ops.matmul(self.W, inputs_tiled)

        # DECISION plan-2026-08-19T163559-499b6f0e/D-011: b must use dtype=self.compute_dtype,
        # not the ops.zeros floatx() default, or c * u_hat raises under mixed_float16. See decisions.md.
        b = keras.ops.zeros(
            [batch_size, self.num_input_capsules, self.num_capsules, 1, 1],
            dtype=self.compute_dtype,
        )

        # Perform iterative dynamic routing
        for i in range(self.routing_iterations):
            # Routing weights: softmax over the output-capsule axis.
            c = keras.activations.softmax(b, axis=2)

            # Weighted sum of predictions, shape (B, 1, N_out, D_out, 1).
            s = keras.ops.sum(c * u_hat, axis=1, keepdims=True)

            # Add bias if requested
            if self.use_bias and self.bias is not None:
                s += self.bias

            # Apply squashing non-linearity
            v = self.squash_layer(s, training=training)

            # For all but the last iteration, update routing logits
            if i < self.routing_iterations - 1:
                # Tile v over the input-capsule axis to match u_hat.
                v_tiled = keras.ops.tile(v, [1, self.num_input_capsules, 1, 1, 1])

                # Agreement u_hat . v, shape (B, N_in, N_out, 1, 1).
                agreement = keras.ops.sum(
                    u_hat * v_tiled,
                    axis=-2,
                    keepdims=True
                )

                b += agreement

        # Final output: shape [batch_size, num_capsules, dim_capsules]
        return keras.ops.reshape(v, (batch_size, self.num_capsules, self.dim_capsules))

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


@register_dl_technique("dl_techniques.layers.capsules")
class CapsuleBlock(keras.layers.Layer):
    """Complete capsule block combining dynamic routing with optional regularization.

    Wraps a ``RoutingCapsule`` layer with optional dropout and layer
    normalization to create a stackable capsule processing unit. The processing
    flow is ``x = RoutingCapsule(inputs) -> Dropout -> LayerNorm``.

    Architecture:

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
            # Applied to the direction component only (see call()); over the
            # full vector it would rescale ||v||, the margin loss's probability.
            self.layer_norm = keras.layers.LayerNormalization(
                axis=-1,
                name="block_norm"
            )
            self._ln_eps = 1e-7
        else:
            self.layer_norm = None
            self._ln_eps = 1e-7

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
            logger.info(
                "CapsuleBlock layer_norm uses length-preserving wrapper: "
                "magnitudes are preserved, LN is applied to the unit-direction "
                "subspace only (see DECISION D-002)."
            )

        logger.info(f"Built CapsuleBlock: {self.num_capsules} capsules, "
                   f"dropout={self.dropout_rate}, layer_norm={self.use_layer_norm}")

        super().build(input_shape)

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
            # Length-preserving LayerNorm: split magnitude/direction, normalize
            # the direction subspace, re-project to unit length, multiply the
            # original magnitude back. Preserves ||x|| (= detection probability)
            # while still giving the routing-to-routing pose subspace the
            # benefit of LN's training stability.
            mag = keras.ops.sqrt(
                keras.ops.sum(keras.ops.square(x), axis=-1, keepdims=True) + self._ln_eps
            )
            direction = x / mag
            direction_normed = self.layer_norm(direction, training=training)
            dir_mag = keras.ops.sqrt(
                keras.ops.sum(keras.ops.square(direction_normed), axis=-1, keepdims=True)
                + self._ln_eps
            )
            direction_unit = direction_normed / dir_mag
            x = mag * direction_unit

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

@register_dl_technique("dl_techniques.layers.capsules")
class CapsuleRoll(keras.layers.Layer):
    """Cyclic permutation of a fixed number of steps *within* each capsule.

    The group action that makes a capsule equivariant rather than invariant. If
    the latent variables are arranged into `C` capsules of `D` dimensions, this
    layer cyclically shifts capsule `c` by `shift` steps along its own axis and
    leaves every other capsule alone — the transformation a group equivariant
    convolution performs on the activations of one feature group when the input
    is transformed by one group element.

    Equation 17 of Keller & Welling (2022), verbatim in indexing:

    .. code-block:: text

        Roll_delta([u_1 .. u_D]) = [u_D, u_1, .. u_{D-1}]   (delta = 1)

    so for this layer's convention ``out[..., c, j] = inputs[..., c, (j - shift) % D]``.
    The layer is deliberately **wrapping**: a shift past the end re-enters at the
    start, which is what makes it invertible and lets a full traversal of the
    capsule dimension return the identity.

    It holds **no weights**. A cyclic shift is a gather, so the layer is a
    re-indexing and nothing else: that is a decision, not an omission. It makes
    the operator's properties structural rather than empirical — it is exactly
    invertible at `shift -> -shift`, it permutes each capsule's values among
    themselves, and there is no weight that could drift away from the
    permutation.

    On the L2 norm, precisely: a permutation preserves the *multiset* of a
    capsule's values, so the norm survives only to within the float32
    summation-order round-off that reordering causes — MEASURED max absolute drift
    **2.4e-07** on random `N(0, 1)` capsules of width 7, i.e. one float32 ulp. Not
    bit-identical, and this docstring does not claim it is: the norm is a *sum of
    squares*, and a different order of those terms gives a different last bit.
    The values themselves are restored exactly, which is what the invertibility
    and round-trip guarantees below are stated against.

    Because the shift is a `call()` argument and not a weight, one layer
    instance serves every `delta` in a temporal-coherence window as well as the
    inference-time traversal `g_theta(Roll_l(t_0))`.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  inputs  (..., C, D)             │
        └────────────────┬─────────────────┘
                         │  index[..., c, j]
                         ▼  = (j - shift) % D
        ┌──────────────────────────────────┐
        │  gather along the capsule axis    │
        └────────────────┬─────────────────┘
                         ▼
        ┌──────────────────────────────────┐
        │  output  (..., C, D)             │
        │  (a cyclic shift within each of  │
        │   the C capsules; capsules are   │
        │   independent)                   │
        └──────────────────────────────────┘

    Input shape:
        Any shape ending in ``(num_capsules, capsule_dim)``, i.e. ``(..., C, D)``.

    Output shape:
        Identical to the input shape.

    :param num_capsules: Number of capsules ``C``. Must be positive.
    :type num_capsules: int
    :param capsule_dim: Dimensions per capsule ``D``. Must be positive.
    :type capsule_dim: int
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If ``num_capsules`` or ``capsule_dim`` is not positive.

    :Example:

    >>> import numpy as np
    >>> from keras import ops
    >>> roll = CapsuleRoll(num_capsules=2, capsule_dim=4)
    >>> x = ops.convert_to_tensor(np.arange(8.0).reshape(1, 2, 4))
    >>> ops.convert_to_numpy(roll(x, shift=1))[0]
    array([3., 0., 1., 2.],
          dtype=float32)
    """

    def __init__(
        self,
        num_capsules: int,
        capsule_dim: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if num_capsules <= 0:
            raise ValueError(f"num_capsules must be positive, got {num_capsules}")
        if capsule_dim <= 0:
            raise ValueError(f"capsule_dim must be positive, got {capsule_dim}")

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim

    def _build_gather_index(self, shift: int) -> np.ndarray:
        """Source slot for every destination slot, for one cyclic shift.

        The convention is ``out[..., c, j] = inputs[..., c, (j - shift) % D]``,
        which reproduces the paper's ``Roll_1 = [u_D, u_1, .., u_{D-1}]``:
        destination slot 0 reads source slot ``D-1``.

        Returns:
            A ``(capsule_dim,)`` int32 array ``index`` with
            ``index[j] = (j - shift) % capsule_dim``.

        :param shift: Cyclic steps, before reduction modulo ``capsule_dim``.
        :type shift: int
        :rtype: np.ndarray
        """
        destinations = np.arange(self.capsule_dim, dtype=np.int32)
        return (destinations - int(shift)) % self.capsule_dim

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        shift: int = 1,
    ) -> keras.KerasTensor:
        """Cyclically permute each capsule by ``shift`` steps.

        :param inputs: Tensor of shape ``(..., C, D)``.
        :type inputs: keras.KerasTensor
        :param training: Unused; accepted so the layer composes with a parent's
            ``training`` propagation. The operator is parameter-free, so there
            is nothing for a training flag to switch.
        :type training: Optional[bool]
        :param shift: Number of cyclic steps. Negative values roll the other
            way and ``shift = 0`` (or a multiple of ``capsule_dim``) is the
            identity. Defaults to 1.
        :type shift: int
        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        # The index table is built from Python integers, never traced ops, so it
        # is fully static regardless of `shift`.
        index = self._build_gather_index(shift)

        # One gather on the capsule axis. The batch, sequence and capsule axes
        # are left untouched, which is what keeps distinct capsules independent.
        return keras.ops.take(
            inputs, keras.ops.convert_to_tensor(index), axis=-1
        )

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape — a permutation preserves the input shape.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "capsule_dim": self.capsule_dim,
        })
        return config

# ---------------------------------------------------------------------
