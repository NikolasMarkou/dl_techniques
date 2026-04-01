"""
Quantile prediction head for probabilistic forecasting with fixed I/O.

This layer serves as the final output stage for a deep forecasting model,
transforming a latent feature representation into a set of quantile
predictions for a future time horizon. It enables probabilistic forecasting,
which moves beyond single-point predictions to provide a richer,
uncertainty-aware view of the future.

The architecture is intentionally simple: a linear projection from the
encoder's feature space to the target space defined by the quantiles and the
forecast horizon. It consists of a single Dense layer that maps the input
features to a flat vector of size ``output_length * num_quantiles``, followed
by a reshape operation to structure the output.

This design assumes that the upstream encoder network is responsible for
extracting all necessary complex, non-linear patterns from the input time
series. This head then acts as a simple, learnable mapping from that rich
representation to the parameters of the forecast distribution.

The layer is designed to be trained with a quantile loss function
(pinball loss):

    L_tau(y, y_hat) = max((y - y_hat) * tau, (y - y_hat) * (tau - 1))

References:
    - Koenker, R., & Bassett Jr, G. (1978). Regression Quantiles.
      Econometrica. https://www.jstor.org/stable/1913643
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class QuantileHead(keras.layers.Layer):
    """
    Quantile prediction head for probabilistic time series forecasting.

    Takes encoded features and projects them to quantile predictions across a
    specified forecast horizon via a single Dense layer followed by a reshape.
    Optionally flattens the input sequence before projection and optionally
    enforces non-crossing quantiles via cumulative softplus deltas.

    When ``enforce_monotonicity=True``, the network outputs raw values
    [r_0, r_1, r_2, ...] and the final quantiles are computed as:

        Q_0 = r_0
        Q_i = Q_{i-1} + Softplus(r_i)  for i > 0

    This guarantees Q_0 <= Q_1 <= Q_2, preventing "crossing quantiles".

    **Architecture Overview:**

    .. code-block:: text

        Input: [batch, seq, feature_dim]
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Flatten ─► [batch, seq*dim]     │ ← (if flatten_input=True)
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Dropout(rate=dropout_rate)      │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Dense(output_length * Q)        │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Reshape ─► [batch, L, Q]        │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Monotonicity Constraint         │ ← (if enforce_monotonicity=True)
        │  Q_0 = r_0                       │
        │  Q_i = Q_0 + cumsum(softplus(r)) │
        └──────────────┬───────────────────┘
                       │
                       ▼
        Output: [batch, output_length, num_quantiles]

    :param num_quantiles: Number of quantiles to predict simultaneously.
    :type num_quantiles: int
    :param output_length: Length of the forecast horizon.
    :type output_length: int
    :param dropout_rate: Dropout probability applied before projection.
        Defaults to 0.1.
    :type dropout_rate: float
    :param use_bias: Whether to include learnable bias terms.
        Defaults to True.
    :type use_bias: bool
    :param flatten_input: If True, the input tensor is flattened (preserving
        batch) before the dense projection, allowing the head to learn from
        the full sequence history. Requires fixed sequence length.
        Defaults to False.
    :type flatten_input: bool
    :param enforce_monotonicity: If True, enforces non-decreasing quantile
        predictions (Q_i <= Q_{i+1}) via cumulative softplus deltas.
        Defaults to False.
    :type enforce_monotonicity: bool
    :param kernel_initializer: Initializer for projection weights.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for projection biases.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        num_quantiles: int,
        output_length: int,
        dropout_rate: float = 0.1,
        use_bias: bool = True,
        flatten_input: bool = False,
        enforce_monotonicity: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_quantiles <= 0:
            raise ValueError(f"num_quantiles must be positive, got {num_quantiles}")
        if output_length <= 0:
            raise ValueError(f"output_length must be positive, got {output_length}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.num_quantiles = num_quantiles
        self.output_length = output_length
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.flatten_input = flatten_input
        self.enforce_monotonicity = enforce_monotonicity
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        self.projection = keras.layers.Dense(
            units=self.output_length * self.num_quantiles,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="quantile_projection"
        )

        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                rate=self.dropout_rate,
                name="quantile_dropout"
            )
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``flatten_input=True`` and input is not 3D or
            has undefined dimensions.
        """
        # Handle logical reshaping for the build process
        # If flattening is enabled, the Dense layer needs to see the flattened dimension
        dense_input_shape = input_shape

        if self.flatten_input:
            # Expecting (Batch, Seq, Feat)
            if len(input_shape) != 3:
                raise ValueError(
                    f"flatten_input=True expects a 3D input tensor (Batch, Seq, Feat), "
                    f"but received shape {input_shape}."
                )

            seq_len = input_shape[-2]
            features = input_shape[-1]

            # Dense layer weights depend on a fixed input dimension.
            if features is None or seq_len is None:
                raise ValueError(
                    "flatten_input=True requires both sequence length and feature dimension "
                    "to be defined (not None) to build the projection layer weights. "
                    f"Received shape: {input_shape}"
                )

            flat_dim = features * seq_len
            dense_input_shape = (input_shape[0], flat_dim)

        # Build sub-layers
        if self.dropout is not None:
            self.dropout.build(dense_input_shape)

        self.projection.build(dense_input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Predict quantiles from the input feature vector.

        :param inputs: Input tensor of shape (batch, features) or
            (batch, seq, features) when ``flatten_input=True``.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode for dropout.
        :type training: Optional[bool]
        :return: Quantile predictions of shape
            (batch_size, output_length, num_quantiles).
        :rtype: keras.KerasTensor
        """
        x = inputs

        # 1. FLATTEN (Configuration Option)
        if self.flatten_input:
            input_shape = ops.shape(x)
            batch_size = input_shape[0]
            # Reshape to (Batch, Seq*Dim) using -1 to infer dimension
            x = ops.reshape(x, (batch_size, -1))

        # 2. DROPOUT
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # 3. PROJECTION
        # Project features to flattened quantile predictions
        quantile_preds = self.projection(x, training=training)

        # 4. RESHAPE OUTPUT
        # Reshape to [batch_size, output_length, num_quantiles]
        # Using -1 for batch dimension handles dynamic batch sizes and symbolic tensors
        quantiles = ops.reshape(
            quantile_preds,
            (-1, self.output_length, self.num_quantiles)
        )

        # 5. MONOTONICITY (Configuration Option)
        # Ensures Q(tau_i) <= Q(tau_{i+1})
        if self.enforce_monotonicity and self.num_quantiles > 1:
            # Split the first quantile from the rest
            # q0: (Batch, Len, 1)
            q0 = quantiles[:, :, 0:1]

            # The rest are interpreted as deltas
            # rest: (Batch, Len, num_quantiles - 1)
            rest = quantiles[:, :, 1:]

            # Force deltas to be positive using softplus
            deltas = ops.softplus(rest)

            # Accumulate deltas
            accumulated_deltas = ops.cumsum(deltas, axis=-1)

            # Add base to accumulation
            subsequent_quantiles = q0 + accumulated_deltas

            # Recombine
            quantiles = ops.concatenate([q0, subsequent_quantiles], axis=-1)

        return quantiles

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0]
        return (batch_size, self.output_length, self.num_quantiles)

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_quantiles": self.num_quantiles,
            "output_length": self.output_length,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "flatten_input": self.flatten_input,
            "enforce_monotonicity": self.enforce_monotonicity,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
        })
        return config

# ---------------------------------------------------------------------
