import keras
from typing import Literal


@keras.saving.register_keras_serializable()
class CountingFFN(keras.layers.Layer):
    """
    A Feed-Forward Network that learns to count events in a sequence.

    This layer identifies "countable" features in the input sequence, aggregates
    their counts, and integrates this information back into each token's
    representation. The counting can be performed globally, locally, or causally.

    The process is as follows:
    1.  A "key" projection identifies what to count, with a sigmoid activation
        creating a "soft" event occurrence probability for each token.
    2.  These events are aggregated based on the `counting_scope`:
        - 'global': Sums events across the entire sequence.
        - 'causal': A cumulative sum, counting events from the start.
        - 'local': A bidirectional cumulative sum, capturing context from
                   both directions.
    3.  A learned gate blends the original input with the transformed count
        information.

    Args:
        output_dim: The final output dimension (must match hidden_size).
        count_dim: The intermediate dimension for the counting projection.
        counting_scope: The scope of counting. One of 'global', 'local', 'causal'.
        use_bias: Whether to use bias in the dense layers.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias weights.
        kernel_regularizer: Regularizer for kernel weights.
        bias_regularizer: Regularizer for bias weights.
    """

    def __init__(
            self,
            output_dim: int,
            count_dim: int,
            counting_scope: Literal["global", "local", "causal"] = "local",
            use_bias: bool = True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        if counting_scope not in ["global", "local", "causal"]:
            raise ValueError(
                f"counting_scope must be one of 'global', 'local', 'causal', "
                f"but got {counting_scope}"
            )
        self.output_dim = output_dim
        self.count_dim = count_dim
        self.counting_scope = counting_scope
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        hidden_size = input_shape[-1]

        # 1. Layer to identify "countable events"
        self.key_projection = keras.layers.Dense(
            self.count_dim,
            activation="sigmoid",
            use_bias=self.use_bias,
            name="key_projection",
        )

        # 2. Layer to transform aggregated counts back to hidden_size
        count_input_dim = self.count_dim
        if self.counting_scope == "local":
            count_input_dim *= 2  # Forward and backward counts are concatenated

        self.count_transform = keras.layers.Dense(
            hidden_size,
            activation="gelu",
            use_bias=self.use_bias,
            name="count_transform",
        )

        # 3. Gating layer to blend count info with original input
        self.gate = keras.layers.Dense(
            hidden_size,
            activation="sigmoid",
            use_bias=self.use_bias,
            name="gate",
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        # 1. Identify what to count for each token
        # Shape: (batch, seq, count_dim)
        countable_events = self.key_projection(inputs)

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

        # 3. Transform the aggregated counts
        # Shape: (batch, seq, hidden_size)
        transformed_counts = self.count_transform(aggregated_counts)

        # 4. Create a gate to blend the information
        # Shape: (batch, seq, hidden_size)
        gate_values = self.gate(inputs)

        # 5. Blend the original input with the count information
        # The gate decides how much count information to let through
        output = (gate_values * transformed_counts) + ((1 - gate_values) * inputs)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "count_dim": self.count_dim,
            "counting_scope": self.counting_scope,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config