"""Perceiver-style transformer block with decoupled cross-attention.

Implements :class:`PerceiverTransformerLayer`, a pre-norm transformer block
that attends a small, fixed-size latent array to a separate key/value array
instead of running self-attention on one sequence. A standard block costs
``O(N^2)`` in the sequence length; this block costs ``O(M*N)``, where ``M``
is the latent length and ``N`` the key/value length, so cost stays linear in
input size for a fixed latent size. Both halves keep the usual pre-norm,
residual shape:

``L' = L + CrossAttention(LN(L), LN(X))``
``L_out = L' + MLP(LN(L'))``

``call()`` accepts one tensor (self-attention, query and key/value share the
input) or a ``[query, kv]`` pair (cross-attention).

References:
    - Jaegle et al., 2021. Perceiver: General Perception with Iterative
      Attention. (https://arxiv.org/abs/2103.03206)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
"""

import keras
from typing import Optional, Any, Dict, Tuple, Union, List

from ..attention.perceiver_attention import PerceiverAttention
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.transformers.perceiver_transformer")
class PerceiverTransformerLayer(keras.layers.Layer):
    """
    Complete Perceiver transformer block with cross-attention.

    Implements asymmetric cross-attention where a small query array attends
    to a (potentially much larger) key/value array, achieving ``O(M*N)``
    complexity instead of ``O(N^2)``. The block uses pre-normalization
    with residual connections around both the cross-attention and the MLP.

    ``L' = L + CrossAttention(LN(L), LN(X))``
    ``L_out = L' + MLP(LN(L'))``

    Architecture:

    .. code-block:: text

        ┌───────────────────────────────────────┐
        │  Query Input (B, M, dim)              │
        │  KV Input    (B, N, dim)              │
        └──────────┬────────────┬───────────────┘
                   ▼            ▼
        ┌───────────────────────────────────────┐
        │  LN(query) ─► Q                       │
        │  LN(kv)    ─► K, V                    │
        │  Cross-Attention(Q, K, V)             │
        │  + Residual (query)                   │
        └──────────────────┬────────────────────┘
                           ▼
        ┌───────────────────────────────────────┐
        │  LN ─► MLP ─► [Dropout]               │
        │  + Residual                           │
        └──────────────────┬────────────────────┘
                           ▼
        ┌───────────────────────────────────────┐
        │  Output (B, M, dim)                   │
        └───────────────────────────────────────┘

    :param dim: Hidden dimension of the block.
    :type dim: int
    :param num_heads: Number of attention heads. Default: 8.
    :type num_heads: int
    :param mlp_ratio: MLP expansion ratio. Default: 4.0.
    :type mlp_ratio: float
    :param dropout_rate: Dropout rate. Default: 0.0.
    :type dropout_rate: float
    :param activation: Activation for the MLP. Default: ``'gelu'``.
    :type activation: Union[str, callable]
    :param use_bias: Whether projections use bias. Default: True.
    :type use_bias: bool
    :param kernel_initializer: Kernel weight initializer.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Bias weight initializer.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Kernel weight regularizer.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Bias weight regularizer.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the base Layer.
    :type kwargs: Any

    :raises ValueError: If dim, num_heads, mlp_ratio, or dropout_rate are invalid.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout_rate: float = 0.0,
            activation: Union[str, callable] = "gelu",
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Calculate MLP hidden dimension
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        # Sub-layers are unbuilt until build() runs.
        self.norm1_q = keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="norm1_q"
        )

        self.norm1_kv = keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="norm1_kv"
        )

        self.attention = PerceiverAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="attention"
        )

        self.norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="norm2"
        )

        self.mlp_dense1 = keras.layers.Dense(
            self.mlp_hidden_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="mlp_dense1"
        )

        self.mlp_dense2 = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="mlp_dense2"
        )

        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(self.dropout_rate, name="dropout")
        else:
            self.dropout = None

    def build(self, input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]) -> None:
        """Build the layer and all sub-layers for serialization safety.

        :param input_shape: Single shape or list of ``[query_shape, kv_shape]``.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        """
        if self.built:
            return

        # A list whose first element is itself a shape means two separate
        # [query, kv] inputs; Keras passes a stored single-input shape back as a plain list on deserialization, so a bare list of ints is not enough to tell them apart.
        is_multi_input = (
            isinstance(input_shape, (list, tuple))
            and len(input_shape) > 0
            and isinstance(input_shape[0], (list, tuple))
        )
        if is_multi_input:
            # Two separate inputs
            if len(input_shape) != 2:
                raise ValueError(f"Expected 2 inputs, got {len(input_shape)}")
            query_shape, kv_shape = input_shape
        else:
            # Single input shape (will be used for both query and kv)
            query_shape = kv_shape = input_shape

        # Validate shapes
        if len(query_shape) != 3:
            raise ValueError(f"Query input must be 3D, got shape {query_shape}")
        if len(kv_shape) != 3:
            raise ValueError(f"KV input must be 3D, got shape {kv_shape}")

        # Only validate the feature dim when it is statically known -- a None
        # channel dim is legitimate at build time and must not be rejected.
        if query_shape[-1] is not None and query_shape[-1] != self.dim:
            raise ValueError(f"Query last dimension ({query_shape[-1]}) "
                             f"must match dim ({self.dim})")
        if kv_shape[-1] is not None and kv_shape[-1] != self.dim:
            raise ValueError(f"KV last dimension ({kv_shape[-1]}) "
                             f"must match dim ({self.dim})")

        # Build sub-layers in computational order
        self.norm1_q.build(query_shape)
        self.norm1_kv.build(kv_shape)

        # Attention layer needs to be built with proper input shapes
        self.attention.build([query_shape, kv_shape])

        # After attention, shape remains same as query
        attn_output_shape = query_shape

        self.norm2.build(attn_output_shape)
        self.mlp_dense1.build(attn_output_shape)

        # MLP dense1 output shape
        mlp_intermediate_shape = list(attn_output_shape)
        mlp_intermediate_shape[-1] = self.mlp_hidden_dim
        mlp_intermediate_shape = tuple(mlp_intermediate_shape)

        if self.dropout is not None:
            self.dropout.build(mlp_intermediate_shape)

        self.mlp_dense2.build(mlp_intermediate_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            query_input: keras.KerasTensor,
            kv_input: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the Perceiver block.

        :param query_input: Query tensor ``(B, M, dim)``.
        :type query_input: keras.KerasTensor
        :param kv_input: Key/value tensor ``(B, N, dim)``. If ``None``,
            uses ``query_input`` (self-attention mode).
        :type kv_input: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Output tensor ``(B, M, dim)``.
        :rtype: keras.KerasTensor
        """
        if kv_input is None:
            kv_input = query_input

        # First residual connection: Cross-attention
        normed_q = self.norm1_q(query_input, training=training)
        normed_kv = self.norm1_kv(kv_input, training=training)

        attn_output = self.attention(
            normed_q,
            normed_kv,
            training=training
        )

        # Add residual connection
        x = query_input + attn_output

        # Second residual connection: MLP
        normed_x = self.norm2(x, training=training)

        # MLP forward pass
        mlp_output = self.mlp_dense1(normed_x, training=training)

        if self.dropout is not None:
            mlp_output = self.dropout(mlp_output, training=training)

        mlp_output = self.mlp_dense2(mlp_output, training=training)

        # Add residual connection
        output = x + mlp_output

        return output

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape (same as query input shape).

        :param input_shape: Single shape or list of shapes.
        :type input_shape: Union[Tuple, List[Tuple]]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        if isinstance(input_shape, list):
            return input_shape[0]  # Same as query input shape
        else:
            return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout_rate": self.dropout_rate,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config
