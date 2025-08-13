"""
This module provides the implementation of `CountingFFN`, a custom Keras 3 layer
designed to explicitly model and integrate counting mechanisms within a sequence.
It offers a powerful alternative to standard Feed-Forward Network (FFN) blocks,
especially for tasks where sequence-level enumeration, frequency, or positional
awareness is a critical component.

The core idea is to equip the network with the ability to identify specific
features or "events" at each position in the sequence, aggregate their counts
across a defined scope, and then fuse this count-based information back into the
token representations. This provides a strong inductive bias for tasks that
benefit from understanding "how many" of something has occurred.

Core Concepts:
-------------
The `CountingFFN` layer operates through a sequence of carefully designed steps:

1.  **Feature Identification & Soft Event Creation:**
    An input projection combined with a sigmoid activation function identifies
    "countable" features. This transforms each token's representation into a
    vector of "soft events"—continuous values between 0 and 1 that represent the
    probability of an event's occurrence at that position.

2.  **Scoped Event Aggregation:**
    The soft events are aggregated according to a specified `counting_scope`,
    which determines the context for the count:
    - **global:** A sum across the entire sequence, providing each token with a
      global count of all events.
    - **causal:** A cumulative sum from the beginning of the sequence to the
      current token, modeling a forward-looking count.
    - **local:** A bidirectional cumulative sum, capturing counts from both the
      start of the sequence and the end, providing rich local context.

3.  **Count Information Transformation:**
    The aggregated count vectors are passed through a non-linear transformation
    (a dense layer with configurable activation) to process and enrich the raw
    count information, preparing it for integration.

4.  **Gated Integration:**
    A dynamic gating mechanism, controlled by another projection from the original
    input, learns to blend the original token representation with the newly
    computed count-based information. This allows the model to adaptively decide
    how much counting information to incorporate for each token.
"""

import keras
from typing import Literal, Tuple, Optional, Union, Callable, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CountingFFN(keras.layers.Layer):
    """
    A Feed-Forward Network that learns to count events in a sequence.

    This layer identifies "countable" features in the input sequence, aggregates
    their counts, and integrates this information back into each token's
    representation. The counting can be performed globally, locally, or causally.

    The layer works by:
    1.  A "key" projection identifies what to count, creating a "soft" event
        occurrence probability for each token using a sigmoid activation.
    2.  These events are aggregated based on the `counting_scope`:
        - 'global': Sums events across the entire sequence.
        - 'causal': A cumulative sum, counting events from the start.
        - 'local': A bidirectional cumulative sum, capturing context from
                   both directions.
    3.  The aggregated counts are transformed into a feature-rich representation
        using a configurable activation function.
    4.  A learned gate blends the original input with the transformed count
        information, allowing the model to adaptively use this new feature.

    Args:
        output_dim: Integer, the final output dimension of the layer. Must be positive.
            For use in residual architectures, this should match the input dimension.
        count_dim: Integer, the intermediate dimension for the counting projection.
            Must be positive. Controls the complexity of features that can be counted.
        counting_scope: Literal['global', 'local', 'causal'], the scope of counting.
            - 'global': Sum events across entire sequence
            - 'local': Bidirectional cumulative sum (forward + backward)
            - 'causal': Forward-only cumulative sum
            Defaults to 'local'.
        activation: Union[str, Callable], activation function for count transformation.
            Can be string name ('gelu', 'relu') or callable. Defaults to 'gelu'.
        use_bias: Boolean, whether to use bias terms in dense layers. Defaults to True.
        kernel_initializer: Union[str, keras.initializers.Initializer], initializer
            for kernel weights. Defaults to 'glorot_uniform'.
        bias_initializer: Union[str, keras.initializers.Initializer], initializer
            for bias weights. Defaults to 'zeros'.
        kernel_regularizer: Optional[keras.regularizers.Regularizer], regularizer
            for kernel weights.
        bias_regularizer: Optional[keras.regularizers.Regularizer], regularizer
            for bias weights.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, input_dim)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`

    Raises:
        ValueError: If output_dim or count_dim is not positive.
        ValueError: If counting_scope is not one of 'global', 'local', 'causal'.

    Example:
        ```python
        # Basic usage with causal counting and default GELU activation
        layer = CountingFFN(output_dim=768, count_dim=128, counting_scope='causal')

        # Advanced configuration with ReLU activation and regularization
        layer = CountingFFN(
            output_dim=512,
            count_dim=64,
            counting_scope='local',
            activation='relu',
            use_bias=False,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # Using Swish activation for smoother gradients
        layer = CountingFFN(
            output_dim=768,
            count_dim=128,
            activation='swish'
        )

        # In a transformer-style model
        input_dim = 768
        inputs = keras.Input(shape=(128, input_dim))
        # output_dim must match input_dim for the residual connection
        counted_features = CountingFFN(
            output_dim=input_dim,
            count_dim=128,
            activation='gelu'
        )(inputs)
        model = keras.Model(inputs, counted_features)
        ```

    Note:
        This layer is particularly effective for tasks requiring an understanding
        of feature frequency, enumeration, or relative positioning within a
        sequence. The `counting_scope` parameter is crucial for tailoring the
        layer's behavior to the specific task (e.g., 'causal' for auto-regressive
        models). This implementation follows the modern Keras 3 pattern where all
        sub-layers are created in __init__ and Keras handles the building automatically.
    """

    def __init__(
        self,
        output_dim: int,
        count_dim: int,
        counting_scope: Literal["global", "local", "causal"] = "local",
        activation: Union[str, Callable] = "gelu",
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if count_dim <= 0:
            raise ValueError(f"count_dim must be positive, got {count_dim}")

        valid_scopes = ["global", "local", "causal"]
        if counting_scope not in valid_scopes:
            raise ValueError(
                f"counting_scope must be one of {valid_scopes}, got '{counting_scope}'"
            )

        # Store ALL configuration arguments as instance attributes
        self.output_dim = output_dim
        self.count_dim = count_dim
        self.counting_scope = counting_scope
        self.activation_fn = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Store original activation identifier for serialization
        self.activation = activation

        # Determine count input dimension based on counting scope
        # 'local' concatenates forward and backward counts, so it's 2x the count_dim
        count_input_dim = self.count_dim * 2 if self.counting_scope == "local" else self.count_dim

        # CREATE all sub-layers in __init__ (MODERN PATTERN)

        # Layer to identify "countable events" with sigmoid activation
        self.key_projection = keras.layers.Dense(
            self.count_dim,
            activation="sigmoid",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="key_projection",
        )

        # Layer to transform aggregated counts with configurable activation
        self.count_transform = keras.layers.Dense(
            self.output_dim,
            activation=None,  # Apply activation in call() for better control
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="count_transform",
        )

        # Gating layer to blend count info with original input
        self.gate = keras.layers.Dense(
            self.output_dim,
            activation="sigmoid",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="gate",
        )

        logger.info(
            f"Created CountingFFN: output_dim={self.output_dim}, "
            f"count_dim={self.count_dim}, counting_scope='{self.counting_scope}', "
            f"activation='{activation}', count_input_dim={count_input_dim}"
        )

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass computation.

        Args:
            inputs: Input tensor with shape (batch_size, sequence_length, input_dim).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor with shape (batch_size, sequence_length, output_dim).
        """
        # Validate input shape during first call
        input_shape = keras.ops.shape(inputs)
        if len(inputs.shape) < 3:
            raise ValueError(
                f"Input must be 3D tensor (batch_size, sequence_length, input_dim), "
                f"got shape {inputs.shape}"
            )

        # Log dimension mismatch warning if output_dim != input_dim
        input_dim = inputs.shape[-1]
        if self.output_dim != input_dim:
            logger.warning(
                f"output_dim ({self.output_dim}) does not match input_dim ({input_dim}). "
                "The layer will not perform a residual-style blend."
            )

        # 1. Identify what to count for each token
        # Shape: (batch, seq, count_dim)
        countable_events = self.key_projection(inputs, training=training)

        # 2. Aggregate counts based on the specified scope
        if self.counting_scope == "global":
            # Sum across the sequence and broadcast back
            global_sum = keras.ops.sum(countable_events, axis=1, keepdims=True)
            aggregated_counts = keras.ops.broadcast_to(
                global_sum, keras.ops.shape(countable_events)
            )
        elif self.counting_scope == "causal":
            # Count everything up to the current token (forward cumulative sum)
            aggregated_counts = keras.ops.cumsum(countable_events, axis=1)
        else:  # 'local'
            # Forward pass: count up to current token
            forward_counts = keras.ops.cumsum(countable_events, axis=1)
            # Backward pass: count from current token to the end
            reversed_events = keras.ops.flip(countable_events, axis=1)
            backward_counts_rev = keras.ops.cumsum(reversed_events, axis=1)
            backward_counts = keras.ops.flip(backward_counts_rev, axis=1)
            # Combine both directions
            aggregated_counts = keras.ops.concatenate(
                [forward_counts, backward_counts], axis=-1
            )

        # 3. Transform the aggregated counts with configurable activation
        # Shape: (batch, seq, output_dim)
        transformed_counts = self.count_transform(aggregated_counts, training=training)
        transformed_counts = self.activation_fn(transformed_counts)

        # 4. Create a gate to blend the information
        # Shape: (batch, seq, output_dim)
        gate_values = self.gate(inputs, training=training)

        # 5. Blend the original input with the count information
        # The gate decides how much count information to let through
        # Note: This assumes output_dim matches input_dim for proper blending
        # If dimensions don't match, this becomes a weighted transformation
        if self.output_dim == inputs.shape[-1]:
            # Standard gated blending when dimensions match
            output = (gate_values * transformed_counts) + ((1 - gate_values) * inputs)
        else:
            # When dimensions don't match, just return the gated transformed counts
            output = gate_values * transformed_counts

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple.
        """
        # Output shape: (batch_size, sequence_length, output_dim)
        return tuple(input_shape[:-1]) + (self.output_dim,)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        This method returns ALL arguments needed to recreate the layer
        via __init__. Uses keras serializers for complex objects.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "count_dim": self.count_dim,
            "counting_scope": self.counting_scope,
            "activation": self.activation,  # Store original activation for serialization
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    # NOTE: get_build_config() and build_from_config() are REMOVED
    # These are deprecated methods that cause serialization issues in Keras 3
    # The modern pattern handles building automatically

# ---------------------------------------------------------------------