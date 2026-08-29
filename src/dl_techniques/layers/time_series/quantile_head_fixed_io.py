"""
Quantile prediction head for a fixed forecast horizon.

This layer is the output stage of a forecasting model. It turns encoded
features into one value per quantile for every step of a fixed horizon.
That gives a predicted range instead of a single number.

The head itself is a linear map. One Dense layer projects the features to
``output_length * num_quantiles`` values. A reshape then splits that flat
vector into a (horizon, quantile) grid. All the non-linear work is expected
to happen in the encoder upstream.

An optional constraint keeps the quantiles in order. Without it the model
can predict a 90th percentile below its own 10th percentile.

Train the layer with the pinball loss:

    L_tau(y, y_hat) = max((y - y_hat) * tau, (y - y_hat) * (tau - 1))

References:
    - Koenker, R., & Bassett Jr, G. (1978). Regression Quantiles.
      Econometrica. https://www.jstor.org/stable/1913643
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, Any
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.quantile_head_fixed_io")
class QuantileHead(keras.layers.Layer):
    """
    Quantile prediction head for a fixed forecast horizon.

    Takes encoded features and projects them to quantile predictions for a
    fixed number of future steps. The projection is a single Dense layer
    followed by a reshape. Set ``flatten_input=True`` to feed the head a
    whole sequence instead of one feature vector.

    With ``enforce_monotonicity=True`` the Dense output is read as
    [r_0, r_1, r_2, ...] and the quantiles are built from it:

        Q_0 = r_0
        Q_i = Q_0 + sum_{j=1..i} Softplus(r_j)   for i > 0

    Softplus is positive, so each quantile is at least as large as the one
    before it. The constraint is applied to the reshaped Dense output, not
    to a separate delta head, and it only runs when ``num_quantiles > 1``.

    Here B is the batch size, D the feature width, S the sequence length,
    L is ``output_length`` and Q is ``num_quantiles``.

    **Architecture Overview:**

    .. code-block:: text

        Input: [B, D], or [B, S, D] if flatten_input
                       │
                       ▼
        ┌──────────────────────────────────┐
        │ Reshape [B, S*D]  (flatten only) │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │ Dropout(dropout_rate) (rate > 0) │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │ Dense(L * Q), linear             │
        └───────────────┬──────────────────┘
                        │ [B, L*Q]
                        ▼
        ┌──────────────────────────────────┐
        │ Reshape (-1, L, Q)               │
        └───────────────┬──────────────────┘
                        │ [B, L, Q]
              ┌─────────┴─────────┐
              │                   │ (monotonic, Q > 1)
              │                   ▼
              │       ┌─────────────────────────┐
              │       │ q0   = r[..., 0:1]      │
              │       │ rest = r[..., 1:]       │
              │       │ q0 + cumsum(softplus)   │
              │       │ concat -> [B, L, Q]     │
              │       └────────────┬────────────┘
              ▼                    ▼
        Output: [B, L, Q]

    Dropout is created only when ``dropout_rate > 0``; otherwise that stage
    is absent, not a no-op layer.

    :param num_quantiles: Number of quantiles to predict at once. Must be
        positive.
    :type num_quantiles: int
    :param output_length: Length of the forecast horizon. Must be positive.
    :type output_length: int
    :param dropout_rate: Dropout probability applied before the projection.
        Must be in [0, 1]. A value of 0 removes the dropout layer.
        Defaults to 0.1.
    :type dropout_rate: float
    :param use_bias: Whether the Dense projection has a bias term.
        Defaults to True.
    :type use_bias: bool
    :param flatten_input: If True, a 3D input is flattened to
        (batch, seq * features) before the projection, so the head sees the
        whole history. The sequence length and feature width must both be
        known at build time. Defaults to False.
    :type flatten_input: bool
    :param enforce_monotonicity: If True, quantiles are non-decreasing along
        the last axis. Defaults to False.
    :type enforce_monotonicity: bool
    :param kernel_initializer: Initializer for the projection weights.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the projection bias.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        2D tensor (batch, features). With ``flatten_input=True``, a 3D
        tensor (batch, seq_len, features) with both trailing dimensions
        known.

    Output shape:
        3D tensor (batch, output_length, num_quantiles).

    Example:
        .. code-block:: python

            head = QuantileHead(num_quantiles=3, output_length=24)
            y = head(keras.random.normal((8, 64)))
            # y.shape == (8, 24, 3)

    Note:
        Pass a 3D tensor only with ``flatten_input=True``. With
        ``flatten_input=False`` the final reshape folds the sequence axis
        into the batch axis instead of raising: a (4, 7, 16) input returns
        (28, 5, 3), which disagrees with ``compute_output_shape``.
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
        """
        Return the constructor arguments needed to rebuild this layer.

        :return: Serializable configuration dictionary.
        :rtype: dict[str, Any]
        """
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
