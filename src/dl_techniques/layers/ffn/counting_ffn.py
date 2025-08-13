import keras
from typing import Literal, Tuple, Optional, Union, Any, Dict

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
    4.  A learned gate controls the integration of count information:
        - If output_dim == input_dim: Performs residual-style blending between
          the original input and transformed counts.
        - If output_dim != input_dim: Gates the transformed counts directly
          (no residual connection possible due to dimension mismatch).

    Args:
        output_dim: Integer, the final output dimension of the layer. For residual
            architectures, this should match the input dimension to enable
            residual-style blending. Must be positive.
        count_dim: Integer, the intermediate dimension for the counting projection.
            Controls the complexity of features that can be counted. Must be positive.
        counting_scope: String, the scope of counting. Must be one of
            'global', 'local', or 'causal'. Defaults to 'local'.
        activation: String name of activation function or activation function to use
            for the count transformation layer. Defaults to 'gelu'.
        use_bias: Boolean, whether to use bias terms in dense layers.
            Defaults to True.
        kernel_initializer: String or initializer instance for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer instance for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
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
        # Residual-style usage (output_dim matches input_dim for blending)
        input_dim = 768
        inputs = keras.Input(shape=(128, input_dim))
        counted_features = CountingFFN(
            output_dim=input_dim,  # Same as input for residual blending
            count_dim=128,
            counting_scope='causal',
            activation='gelu'
        )(inputs)

        # Dimension-changing usage (no residual blending)
        projection_layer = CountingFFN(
            output_dim=512,  # Different from input_dim
            count_dim=64,
            counting_scope='local',
            activation='swish'
        )(inputs)  # Input: 768-dim, Output: 512-dim

        # Different counting scopes for different use cases
        global_counts = CountingFFN(
            output_dim=256,
            count_dim=32,
            counting_scope='global'  # Each token sees global sequence counts
        )(inputs)

        causal_counts = CountingFFN(
            output_dim=256,
            count_dim=32,
            counting_scope='causal'  # For autoregressive models
        )(inputs)

        # Advanced configuration with regularization
        layer = CountingFFN(
            output_dim=512,
            count_dim=128,
            counting_scope='local',
            activation='relu',
            use_bias=False,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )
        ```

    Note:
        This layer is particularly effective for tasks requiring an understanding
        of feature frequency, enumeration, or relative positioning within a
        sequence. The `counting_scope` parameter is crucial for tailoring the
        layer's behavior to the specific task (e.g., 'causal' for auto-regressive
        models). When output_dim matches input_dim, the layer performs residual-style
        blending; otherwise, it acts as a dimension-changing transformation layer.
    """

    def __init__(
        self,
        output_dim: int,
        count_dim: int,
        counting_scope: Literal["global", "local", "causal"] = "local",
        activation: Union[str, callable] = "gelu",
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
        if counting_scope not in ["global", "local", "causal"]:
            raise ValueError(
                f"counting_scope must be one of 'global', 'local', 'causal', "
                f"but got {counting_scope}"
            )

        # Store configuration parameters
        self.output_dim = output_dim
        self.count_dim = count_dim
        self.counting_scope = counting_scope
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Store original activation for serialization
        self._activation_identifier = activation

        # CREATE all sub-layers in __init__ following modern Keras 3 pattern
        # Layer to identify "countable events"
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
        # Note: The actual input dimension is determined dynamically based on counting_scope
        # For 'local' scope, we concatenate forward and backward counts (2 * count_dim)
        # For 'global' and 'causal' scopes, it's just count_dim
        count_transform_input_dim = self.count_dim * 2 if self.counting_scope == "local" else self.count_dim

        self.count_transform = keras.layers.Dense(
            self.output_dim,
            activation=self.activation,
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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the Counting FFN and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(
                f"Input must be at least 2D, got {len(input_shape)}D: {input_shape}"
            )
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Input feature dimension must be specified")

        logger.info(
            f"Building CountingFFN: input_dim={input_dim}, output_dim={self.output_dim}, "
            f"count_dim={self.count_dim}, counting_scope='{self.counting_scope}', "
            f"activation='{self._activation_identifier}'"
        )

        if self.output_dim != input_dim:
            logger.warning(
                f"output_dim ({self.output_dim}) does not match input_dim ({input_dim}). "
                "The layer will use gated count transformation instead of residual-style blending."
            )

        # Build sub-layers in computational order for robust serialization
        # 1. Build key projection (takes original input)
        self.key_projection.build(input_shape)

        # 2. Build gate (takes original input)
        self.gate.build(input_shape)

        # 3. Build count transform (takes aggregated counts)
        count_input_dim = self.count_dim
        if self.counting_scope == "local":
            count_input_dim *= 2  # Forward and backward counts are concatenated

        count_transform_input_shape = tuple(input_shape[:-1]) + (count_input_dim,)
        self.count_transform.build(count_transform_input_shape)

        # Always call parent build at the end
        super().build(input_shape)

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
            # Count everything up to the current token
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

        # 4. Create a gate to blend the information
        # Shape: (batch, seq, output_dim)
        gate_values = self.gate(inputs, training=training)

        # 5. Blend the count information based on dimensions compatibility
        input_dim = keras.ops.shape(inputs)[-1]

        if self.output_dim == input_dim:
            # When dimensions match, perform residual-style blending with original input
            # The gate decides how much count information vs original input to use
            output = (gate_values * transformed_counts) + ((1 - gate_values) * inputs)
        else:
            # When dimensions don't match, we can't blend with original input
            # Instead, gate controls how much of the transformed counts to use
            # Gate of 1.0 = full transformed counts, gate of 0.0 = zeros
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
        return tuple(input_shape[:-1]) + (self.output_dim,)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "count_dim": self.count_dim,
            "counting_scope": self.counting_scope,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config