"""
Differential Multi-Head Attention Implementation.

This module implements Differential Multi-Head Attention as described in the paper
"DIFFERENTIAL TRANSFORMER: Amplifying attention to the relevant context while canceling noise".

The Differential Attention mechanism employs two parallel scaled dot-product attention
streams and computes a weighted difference between them:
``Attention_diff = SDPA1(x) - lambda * SDPA2(x)``.
This design amplifies relevant context signals while attenuating noise, resulting
in more focused attention patterns.

The adaptive lambda parameter is computed as:
``lambda(l) = (0.8 - 0.6 * exp(-0.3 * (l - 1))) * lambda_learned``
and bounded to ``[0.1, 0.9]`` for training stability.

This implementation uses **manual scaled dot-product attention** rather than two
``keras.layers.MultiHeadAttention`` instances. This makes the per-stream attention
probability normalization customizable via :class:`ProbabilityOutput` and exposes an
optional QK-norm hook applied independently to each stream's Q/K projections.

References:
    Ye, T., Dong, L., Xia, Y., Sun, Y., Zhu, Y., Huang, G., & Wei, F.
    "DIFFERENTIAL TRANSFORMER: Amplifying attention to the relevant context
    while canceling noise"
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from ..activations import ProbabilityOutput
from ..norms.factory import create_normalization_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DifferentialMultiHeadAttention(keras.layers.Layer):
    """
    Differential multi-head attention mechanism with customizable probability
    normalization and optional QK-norm.

    This layer implements differential attention using two parallel manual
    scaled-dot-product-attention (SDPA) streams. It computes their weighted
    difference to amplify relevant context while canceling noise. The key
    innovation is the learnable lambda parameter that balances the contribution
    of the two attention mechanisms.

    The differential attention is computed as:
    ``Attention_diff = SDPA1(x) - lambda * SDPA2(x)``
    where SDPA1 captures primary patterns, SDPA2 identifies noise, and lambda
    controls the noise cancellation strength.

    Each stream's softmax is replaced by an instance of :class:`ProbabilityOutput`
    (``self.attn_prob_1`` / ``self.attn_prob_2``), enabling arbitrary probability
    types (softmax / sparsemax / threshmax / adaptive). Two separate instances are
    used so that per-site debugging and weight inspection remains
    straightforward.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────┐
        │              DifferentialMultiHeadAttention             │
        │                                                         │
        │  Input [B, L, D]                                        │
        │         │                                               │
        │         ▼                                               │
        │   QKV projection -> Q1, K1, Q2, K2, V (shared)          │
        │         │                                               │
        │   ┌─────┴───────────────┬──────────────────┐            │
        │   ▼                     ▼                  ▼            │
        │  Q1,K1               Q2,K2                 V            │
        │   │                     │                  │            │
        │   ▼                     ▼                  │            │
        │ optional q/k_norm_1  optional q/k_norm_2   │            │
        │   │                     │                  │            │
        │   ▼                     ▼                  │            │
        │ scores1 = Q1@K1^T/√d   scores2 = Q2@K2^T/√d│            │
        │   │                     │                  │            │
        │   ▼                     ▼                  │            │
        │ attn_prob_1          attn_prob_2           │            │
        │   │                     │                  │            │
        │   └──── @V ──┐   ┌── @V ┘                  │            │
        │              ▼   ▼                                      │
        │           out1   out2                                   │
        │              │   │                                      │
        │              └── - lambda * ──┐                         │
        │                               │                         │
        │                               ▼                         │
        │                     Differential Output                 │
        │                               │                         │
        │                               ▼                         │
        │                        Output Projection                │
        │                               │                         │
        │                               ▼                         │
        │                            Dropout                      │
        │                               │                         │
        │                               ▼                         │
        │                       Output [B, L, D]                  │
        └─────────────────────────────────────────────────────────┘

    :param dim: Integer, input and output dimension. Must be positive and should be
        divisible by num_heads for optimal performance.
    :type dim: int
    :param num_heads: Integer, number of attention heads for both attention streams.
        Must be positive.
    :type num_heads: int
    :param head_dim: Integer, dimension of each attention head. Must be positive.
    :type head_dim: int
    :param dropout_rate: Float, output dropout rate applied after projection.
        Must be between 0 and 1. Defaults to 0.0.
    :type dropout_rate: float
    :param attention_dropout_rate: Float, dropout rate applied to attention weights in
        both streams. Must be between 0 and 1. Defaults to 0.0.
    :type attention_dropout_rate: float
    :param lambda_init: Float, initial value for the lambda parameter controlling the
        balance between attention streams. Should be between 0 and 1.
        Defaults to 0.8.
    :type lambda_init: float
    :param probability_type: String identifier for the per-stream probability
        normalization strategy. Forwarded to :class:`ProbabilityOutput`. Both streams
        share the same type. Defaults to ``"softmax"``.
    :type probability_type: str
    :param probability_config: Optional dict of strategy-specific arguments forwarded
        to :class:`ProbabilityOutput`. Both streams share the same config.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied to each stream's
        per-head Q and K projections before computing attention scores (QK-norm).
        Forwarded to :func:`create_normalization_layer`. ``None`` disables QK-norm.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        :func:`create_normalization_layer` when constructing per-stream Q/K norms.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param kernel_initializer: String or Initializer, initializer for kernel weights.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional Regularizer, regularizer applied to kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_initializer: String or Initializer, initializer for bias weights.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param bias_regularizer: Optional Regularizer, regularizer applied to bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param activity_regularizer: Optional Regularizer, regularizer applied to layer output.
    :type activity_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments passed to Layer base class.

    :raises ValueError: If dim is not positive.
    :raises ValueError: If num_heads is not positive.
    :raises ValueError: If head_dim is not positive.
    :raises ValueError: If dropout rates are not between 0 and 1.
    :raises ValueError: If lambda_init is not between 0 and 1.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        lambda_init: float = 0.8,
        probability_type: str = "softmax",
        probability_config: Optional[Dict[str, Any]] = None,
        qk_norm_type: Optional[str] = None,
        qk_norm_kwargs: Optional[Dict[str, Any]] = None,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the differential multi-head attention layer."""
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout_rate}")
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(
                f"attention_dropout_rate must be between 0 and 1, got {attention_dropout_rate}"
            )
        if not (0.0 <= lambda_init <= 1.0):
            raise ValueError(f"lambda_init must be between 0 and 1, got {lambda_init}")

        # Reject routing/hierarchical probability types: they require an
        # output_dim and consume features rather than score logits, which is
        # incompatible with attention scores whose last dimension is the
        # dynamic kv sequence length.
        _ptype_lower = probability_type.lower()
        if _ptype_lower in (
            "routing",
            "deterministic_routing",
            "hierarchical",
            "hierarchical_routing",
        ):
            raise ValueError(
                f"probability_type='{probability_type}' is not supported in "
                "DifferentialMultiHeadAttention: routing/hierarchical strategies "
                "require a fixed output_dim and consume features rather than "
                "score logits. Use one of: 'softmax', 'sparsemax', 'threshmax', "
                "'adaptive'."
            )

        # Store configuration - ALL __init__ parameters must be stored
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.lambda_init = lambda_init
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        # Store serialized initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Per-head projection width (each stream has num_heads * head_dim
        # features for Q and K; V is shared between streams).
        self._proj_dim = self.num_heads * self.head_dim

        # Scale factor for scaled dot-product attention.
        self.scale = 1.0 / ops.sqrt(float(self.head_dim))

        # CREATE all sub-layers in __init__ following modern Keras 3 pattern
        try:
            dense_kwargs = {
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
                "kernel_regularizer": self.kernel_regularizer,
                "bias_regularizer": self.bias_regularizer,
            }

            # Five separate projection Dense layers (one fused per stream's
            # Q/K is also possible but five separate layers keeps debugging
            # trivial and matches the per-site pattern documented above).
            self.q1_dense = keras.layers.Dense(self._proj_dim, name="q1", **dense_kwargs)
            self.k1_dense = keras.layers.Dense(self._proj_dim, name="k1", **dense_kwargs)
            self.q2_dense = keras.layers.Dense(self._proj_dim, name="q2", **dense_kwargs)
            self.k2_dense = keras.layers.Dense(self._proj_dim, name="k2", **dense_kwargs)
            self.v_dense = keras.layers.Dense(self._proj_dim, name="v", **dense_kwargs)

            # Output projection layer
            self.proj = keras.layers.Dense(
                self.dim,
                name='proj',
                **dense_kwargs,
            )

            # Output dropout layer
            self.dropout_layer = keras.layers.Dropout(self.dropout_rate, name='dropout')

            # Per-stream attention-weight dropout (matches the
            # ``attention_dropout_rate`` of the original MHA-based version).
            if self.attention_dropout_rate > 0.0:
                self.attn_dropout_1 = keras.layers.Dropout(
                    self.attention_dropout_rate, name="attn_dropout_1"
                )
                self.attn_dropout_2 = keras.layers.Dropout(
                    self.attention_dropout_rate, name="attn_dropout_2"
                )
            else:
                self.attn_dropout_1 = None
                self.attn_dropout_2 = None

            # Per-stream probability normalization layers (two instances,
            # sharing the same probability_type / probability_config).
            self.attn_prob_1 = ProbabilityOutput(
                probability_type=self.probability_type,
                type_config=self.probability_config,
                name="attn_prob_1",
            )
            self.attn_prob_2 = ProbabilityOutput(
                probability_type=self.probability_type,
                type_config=self.probability_config,
                name="attn_prob_2",
            )

            # Optional per-stream QK-norm. Each stream gets its own pair of
            # Q/K normalization layers so they remain independent.
            if self.qk_norm_type is not None:
                _qk_kwargs = self.qk_norm_kwargs or {}
                self.q_norm_1 = create_normalization_layer(
                    self.qk_norm_type, name="q_norm_1", **_qk_kwargs
                )
                self.k_norm_1 = create_normalization_layer(
                    self.qk_norm_type, name="k_norm_1", **_qk_kwargs
                )
                self.q_norm_2 = create_normalization_layer(
                    self.qk_norm_type, name="q_norm_2", **_qk_kwargs
                )
                self.k_norm_2 = create_normalization_layer(
                    self.qk_norm_type, name="k_norm_2", **_qk_kwargs
                )
            else:
                self.q_norm_1 = None
                self.k_norm_1 = None
                self.q_norm_2 = None
                self.k_norm_2 = None

        except Exception as e:
            logger.error(f"Failed to create DifferentialMultiHeadAttention sub-layers: {e}")
            raise ValueError(
                f"Failed to create DifferentialMultiHeadAttention sub-layers. "
                f"This might be due to invalid configuration parameters. "
                f"Original error: {e}"
            )

        # Weight attributes - created in build()
        self.lambda_param = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and create the lambda parameter weight.

        Creates the learnable lambda parameter and explicitly builds all sub-layers
        for robust serialization following modern Keras 3 patterns.

        :param input_shape: Shape tuple of the input tensor, expected as
            ``(batch_size, seq_len, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch_size, seq_len, dim), got shape: {input_shape}"
            )

        input_dim = input_shape[-1]
        if input_dim != self.dim:
            raise ValueError(
                f"Input dimension {input_dim} doesn't match expected dimension {self.dim}"
            )

        # Create the layer's own weights - lambda parameter.
        # The lambda-init schedule (preserved exactly from the previous
        # implementation) is: lambda = clip(layer_dep_init * lambda_param, 0.1, 0.9)
        # where layer_dep_init = 0.8 - 0.6 * exp(-0.3 * max(layer_idx - 1, 0)).
        self.lambda_param = self.add_weight(
            name="lambda_param",
            shape=(1,),
            initializer=keras.initializers.Constant(self.lambda_init),
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        # Build projection layers explicitly for serialization.
        self.q1_dense.build(input_shape)
        self.k1_dense.build(input_shape)
        self.q2_dense.build(input_shape)
        self.k2_dense.build(input_shape)
        self.v_dense.build(input_shape)

        # Output projection consumes (B, L, num_heads*head_dim) and produces (B, L, dim).
        proj_input_shape = (input_shape[0], input_shape[1], self._proj_dim)
        self.proj.build(proj_input_shape)
        self.dropout_layer.build(input_shape)

        # Build per-stream probability layers with the attention-score shape.
        attn_shape = (input_shape[0], self.num_heads, input_shape[1], input_shape[1])
        self.attn_prob_1.build(attn_shape)
        self.attn_prob_2.build(attn_shape)

        if self.attn_dropout_1 is not None:
            self.attn_dropout_1.build(attn_shape)
            self.attn_dropout_2.build(attn_shape)

        # Build per-stream QK-norm layers with the per-head Q/K shape.
        if self.q_norm_1 is not None:
            qk_shape = (input_shape[0], self.num_heads, input_shape[1], self.head_dim)
            self.q_norm_1.build(qk_shape)
            self.k_norm_1.build(qk_shape)
            self.q_norm_2.build(qk_shape)
            self.k_norm_2.build(qk_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def get_lambda(self, layer_idx: int = 0) -> keras.KerasTensor:
        """
        Compute the lambda value with layer-dependent adaptation.

        The lambda parameter is adapted based on layer depth following the paper's
        initialization strategy: ``lambda = 0.8 - 0.6 * exp(-0.3 * (layer_idx - 1))``.
        The learned ``lambda_param`` is then applied as a multiplicative factor.

        :param layer_idx: Integer, index of the layer in the network stack (0-based).
            Used to compute layer-dependent lambda initialization.
        :type layer_idx: int
        :return: Tensor containing the computed lambda value, bounded between 0.1 and 0.9.
        :rtype: keras.KerasTensor
        """
        # Layer-dependent initialization following the paper
        # lambda_init = 0.8 - 0.6*exp(-0.3*(layer_idx - 1))
        layer_factor = ops.cast(layer_idx, dtype="float32")
        exp_term = ops.exp(-0.3 * ops.maximum(layer_factor - 1.0, 0.0))
        layer_dependent_init = 0.8 - 0.6 * exp_term

        # Apply learned lambda parameter as multiplicative factor
        # Clip to ensure training stability
        lambda_val = ops.clip(
            layer_dependent_init * self.lambda_param[0],
            0.1,
            0.9,
        )

        return lambda_val

    def _apply_attention_mask(
        self,
        scores: keras.KerasTensor,
        attention_mask: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """
        Apply attention mask to scores tensor.

        :param scores: Attention scores of shape ``(batch, num_heads, q_seq, kv_seq)``.
        :type scores: keras.KerasTensor
        :param attention_mask: Attention mask. Supported shapes: ``(batch, kv_seq)``
            (padding mask), ``(batch, q_seq, kv_seq)`` (full mask), or
            ``(batch, num_heads, q_seq, kv_seq)``.
        :type attention_mask: keras.KerasTensor
        :return: Masked scores tensor with same shape as input scores.
        :rtype: keras.KerasTensor
        """
        attention_mask = ops.cast(attention_mask, scores.dtype)
        if len(attention_mask.shape) == 2:
            attention_mask = ops.expand_dims(ops.expand_dims(attention_mask, 1), 1)
        elif len(attention_mask.shape) == 3:
            attention_mask = ops.expand_dims(attention_mask, 1)
        mask_value = -1e9
        return scores + (1.0 - attention_mask) * mask_value

    def _project_to_heads(
        self,
        x: keras.KerasTensor,
        batch_size: keras.KerasTensor,
        seq_len: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Reshape a projected tensor ``(B, L, H*D_h)`` to ``(B, H, L, D_h)``."""
        x = ops.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def _stream(
        self,
        q: keras.KerasTensor,
        k: keras.KerasTensor,
        v: keras.KerasTensor,
        q_norm: Optional[keras.layers.Layer],
        k_norm: Optional[keras.layers.Layer],
        attn_prob: ProbabilityOutput,
        attn_dropout: Optional[keras.layers.Dropout],
        attention_mask: Optional[keras.KerasTensor],
        training: Optional[bool],
    ) -> keras.KerasTensor:
        """
        Run a single SDPA stream and return ``(B, H, L, D_h)`` context.

        Applies optional QK-norm, computes scaled dot-product scores, optional
        mask, calls the supplied ``ProbabilityOutput`` to normalize the scores,
        applies optional attention-weight dropout, and returns ``attn @ v``.
        """
        if q_norm is not None:
            q = q_norm(q, training=training)
        if k_norm is not None:
            k = k_norm(k, training=training)

        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        scores = scores * ops.cast(self.scale, q.dtype)

        if attention_mask is not None:
            scores = self._apply_attention_mask(scores, attention_mask)

        attn = attn_prob(scores, training=training)

        if attn_dropout is not None:
            attn = attn_dropout(attn, training=training)

        return ops.matmul(attn, v)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        layer_idx: int = 0,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Apply differential attention mechanism.

        Computes the differential attention as:
        ``Attention_diff = SDPA1(x) - lambda * SDPA2(x)``
        where SDPA1 captures primary attention patterns, SDPA2 identifies noise
        patterns, and lambda controls the balance between them.

        :param inputs: Input tensor of shape ``(batch_size, sequence_length, dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional attention mask tensor. Can be 2D, 3D, or 4D
            for different masking strategies.
        :type attention_mask: Optional[keras.KerasTensor]
        :param layer_idx: Integer, index of the layer in the network stack (0-based).
            Used for layer-dependent lambda computation. Defaults to 0.
        :type layer_idx: int
        :param training: Optional boolean indicating whether in training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, sequence_length, dim)`` after
            applying differential attention and output projection.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Project to Q1, K1, Q2, K2, V and reshape to per-head format.
        q1 = self._project_to_heads(self.q1_dense(inputs), batch_size, seq_len)
        k1 = self._project_to_heads(self.k1_dense(inputs), batch_size, seq_len)
        q2 = self._project_to_heads(self.q2_dense(inputs), batch_size, seq_len)
        k2 = self._project_to_heads(self.k2_dense(inputs), batch_size, seq_len)
        v = self._project_to_heads(self.v_dense(inputs), batch_size, seq_len)

        # Two parallel SDPA streams (V is shared, lambda-init schedule
        # combines them post-hoc).
        out1 = self._stream(
            q1, k1, v,
            self.q_norm_1, self.k_norm_1,
            self.attn_prob_1, self.attn_dropout_1,
            attention_mask, training,
        )
        out2 = self._stream(
            q2, k2, v,
            self.q_norm_2, self.k_norm_2,
            self.attn_prob_2, self.attn_dropout_2,
            attention_mask, training,
        )

        # Compute layer-dependent lambda value (same schedule as original).
        lambda_val = self.get_lambda(layer_idx)

        # Differential attention: SDPA1 - lambda*SDPA2
        diff = out1 - lambda_val * out2

        # Merge heads: (B, H, L, D_h) -> (B, L, H*D_h)
        diff = ops.transpose(diff, (0, 2, 1, 3))
        diff = ops.reshape(diff, (batch_size, seq_len, self._proj_dim))

        # Apply output projection and dropout
        output = self.proj(diff, training=training)
        output = self.dropout_layer(output, training=training)

        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape, same as input shape for attention layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization, includes all constructor parameters.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'lambda_init': self.lambda_init,
            'probability_type': self.probability_type,
            'probability_config': self.probability_config,
            'qk_norm_type': self.qk_norm_type,
            'qk_norm_kwargs': self.qk_norm_kwargs,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
