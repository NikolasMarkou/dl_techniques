import math
import string
from typing import Optional, Union, Tuple, List, Dict, Any

import numpy as np
from keras.src import ops
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.layers.layer import Layer
from keras.src.api_export import keras_export
from keras.src.layers.core.einsum_dense import EinsumDense
from keras.src.layers.regularization.dropout import Dropout
from keras.src.backend.config import is_flash_attention_enabled


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..adaptive_softmax import AdaptiveTemperatureSoftmax

# ---------------------------------------------------------------------


@keras_export("keras.layers.AdaptiveMultiHeadAttention")
class AdaptiveMultiHeadAttention(Layer):
    """MultiHeadAttention layer with AdaptiveTemperatureSoftmax.

    This is an implementation of multi-headed attention as described in the
    paper "Attention is all you Need" [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762),
    but using AdaptiveTemperatureSoftmax instead of standard softmax for attention score normalization.

    The adaptive temperature softmax dynamically adjusts its temperature parameter based on
    the entropy of the input distribution, helping maintain sharpness in the output
    probabilities even as the input size grows.

    Args:
        num_heads: Number of attention heads.
        key_dim: Size of each attention head for query and key.
        value_dim: Size of each attention head for value.
        dropout: Dropout probability.
        use_bias: Boolean, whether the dense layers use bias vectors/matrices.
        output_shape: The expected shape of an output tensor, besides the batch
            and sequence dims. If not specified, projects back to the query
            feature dim (the query input's last dimension).
        attention_axes: axes over which the attention is applied. `None` means
            attention over all axes, but batch, heads, and features.
        flash_attention: If `None`, the layer attempts to use flash
            attention for faster and more memory-efficient attention
            computations when possible.
        kernel_initializer: Initializer for dense layer kernels.
        bias_initializer: Initializer for dense layer biases.
        kernel_regularizer: Regularizer for dense layer kernels.
        bias_regularizer: Regularizer for dense layer biases.
        activity_regularizer: Regularizer for dense layer activity.
        kernel_constraint: Constraint for dense layer kernels.
        bias_constraint: Constraint for dense layer kernels.
        seed: Optional integer to seed the dropout layer.
        min_temp: Minimum temperature for AdaptiveTemperatureSoftmax.
        max_temp: Maximum temperature for AdaptiveTemperatureSoftmax.
        entropy_threshold: Entropy threshold for AdaptiveTemperatureSoftmax.
        polynomial_coeffs: Optional coefficients for the polynomial temperature function.

    Call arguments:
        query: Query tensor of shape `(B, T, dim)`, where `B` is the batch size,
            `T` is the target sequence length, and dim is the feature dimension.
        value: Value tensor of shape `(B, S, dim)`, where `B` is the batch size,
            `S` is the source sequence length, and dim is the feature dimension.
        key: Optional key tensor of shape `(B, S, dim)`. If not given, will
            use `value` for both `key` and `value`, which is the most common
            case.
        attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions. The boolean mask specifies which
            query elements can attend to which key elements, 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.
        return_attention_scores: A boolean to indicate whether the output should
            be `(attention_output, attention_scores)` if `True`, or
            `attention_output` if `False`. Defaults to `False`.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
        use_causal_mask: A boolean to indicate whether to apply a causal mask to
            prevent tokens from attending to future tokens (e.g., used in a
            decoder Transformer).

    Returns:
        attention_output: The result of the computation, of shape `(B, T, E)`,
            where `T` is for target sequence shapes and `E` is the query input
            last dimension if `output_shape` is `None`. Otherwise, the
            multi-head outputs are projected to the shape specified by
            `output_shape`.
        attention_scores: (Optional) multi-head attention coefficients over
            attention axes.
    """

    def __init__(
            self,
            num_heads: int,
            key_dim: int,
            value_dim: Optional[int] = None,
            dropout: float = 0.0,
            use_bias: bool = True,
            output_shape: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
            attention_axes: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
            flash_attention: Optional[bool] = None,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            activity_regularizer: Optional[regularizers.Regularizer] = None,
            kernel_constraint: Optional[constraints.Constraint] = None,
            bias_constraint: Optional[constraints.Constraint] = None,
            seed: Optional[int] = None,
            # AdaptiveTemperatureSoftmax parameters
            min_temp: float = 0.1,
            max_temp: float = 1.0,
            entropy_threshold: float = 0.5,
            polynomial_coeffs: Optional[List[float]] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._dropout = dropout
        self._use_bias = use_bias
        if output_shape:
            if isinstance(output_shape, int):
                output_shape = (output_shape,)
            try:
                output_shape = tuple(output_shape)
            except:
                raise ValueError(
                    f"Invalid `output_shape`: {output_shape}. When "
                    "specified, the `output_shape` should be of type tuple, "
                    "list, or int."
                )
        self._output_shape = output_shape
        self._flash_attention = flash_attention or is_flash_attention_enabled()
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)
        if isinstance(attention_axes, int):
            attention_axes = (attention_axes,)
        elif attention_axes and not isinstance(attention_axes, (list, tuple)):
            raise ValueError(
                "`attention_axes` must be an int, list, or tuple."
                f"Received: attention_axes={attention_axes}"
            )
        self._attention_axes = attention_axes
        self.seed = seed

        # AdaptiveTemperatureSoftmax parameters
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._entropy_threshold = entropy_threshold
        self._polynomial_coeffs = polynomial_coeffs

        self._inverse_sqrt_key_dim = 1.0 / math.sqrt(float(self._key_dim))
        self._return_attention_scores = False

        # Check for flash attention constraints
        if self._flash_attention and self._dropout > 0.0:
            raise ValueError(
                "Dropout is not supported when flash attention is enabled. "
                "Please set dropout to 0.0 to use flash attention."
            )

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def key_dim(self) -> int:
        return self._key_dim

    @property
    def value_dim(self) -> int:
        return self._value_dim

    @property
    def dropout(self) -> float:
        return self._dropout

    @property
    def use_bias(self) -> bool:
        return self._use_bias

    @property
    def attention_axes(self) -> Optional[Tuple[int, ...]]:
        return self._attention_axes

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        config = {
            "num_heads": self._num_heads,
            "key_dim": self._key_dim,
            "value_dim": self._value_dim,
            "dropout": self._dropout,
            "use_bias": self._use_bias,
            "output_shape": self._output_shape,
            "attention_axes": self._attention_axes,
            "kernel_initializer": initializers.serialize(
                self._kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self._bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self._kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self._bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self._activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self._kernel_constraint),
            "bias_constraint": constraints.serialize(self._bias_constraint),
            "seed": self.seed,
            # AdaptiveTemperatureSoftmax params
            "min_temp": self._min_temp,
            "max_temp": self._max_temp,
            "entropy_threshold": self._entropy_threshold,
            "polynomial_coeffs": self._polynomial_coeffs,
        }
        return {**base_config, **config}

    def build(
            self,
            query_shape: Tuple[int, ...],
            value_shape: Tuple[int, ...],
            key_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Builds layers and variables.

        Args:
            query_shape: Shape of the `query` tensor.
            value_shape: Shape of the `value` tensor.
            key_shape: Optional shape of the `key` tensor.
        """
        key_shape = value_shape if key_shape is None else key_shape

        if value_shape[1:-1] != key_shape[1:-1]:
            raise ValueError(
                "All dimensions of `value` and `key`, except the last one, "
                f"must be equal. Received: value_shape={value_shape} and "
                f"key_shape={key_shape}"
            )

        query_rank = len(query_shape)
        value_rank = len(value_shape)
        key_rank = len(key_shape)
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            query_rank - 1, bound_dims=1, output_dims=2
        )
        self._query_dense = EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="query",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._query_dense.build(query_shape)
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            key_rank - 1, bound_dims=1, output_dims=2
        )
        self._key_dense = EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="key",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._key_dense.build(key_shape)
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            value_rank - 1, bound_dims=1, output_dims=2
        )
        self._value_dense = EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._value_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="value",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._value_dense.build(value_shape)

        # Builds the attention computations for multi-head dot product
        # attention.  These computations could be wrapped into the keras
        # attention layer once it supports multi-head einsum computations.
        self._build_attention(output_rank)
        self._output_dense = self._make_output_dense(
            query_shape,
            self._get_common_kwargs_for_sublayer(),
            "attention_output",
        )
        output_dense_input_shape = list(
            self._query_dense.compute_output_shape(query_shape)
        )
        output_dense_input_shape[-1] = self._value_dim
        self._output_dense.build(tuple(output_dense_input_shape))
        self.built = True

    @property
    def query_dense(self) -> EinsumDense:
        return self._query_dense

    @property
    def key_dense(self) -> EinsumDense:
        return self._key_dense

    @property
    def value_dense(self) -> EinsumDense:
        return self._value_dense

    @property
    def output_dense(self) -> EinsumDense:
        return self._output_dense

    def _get_common_kwargs_for_sublayer(self) -> Dict[str, Any]:
        common_kwargs = dict(
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            dtype=self.dtype_policy,
        )
        # Create new clone of kernel/bias initializer, so that we don't reuse
        # the initializer instance, which could lead to same init value since
        # initializer is stateless.
        kernel_initializer = self._kernel_initializer.__class__.from_config(
            self._kernel_initializer.get_config()
        )
        bias_initializer = self._bias_initializer.__class__.from_config(
            self._bias_initializer.get_config()
        )
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return common_kwargs

    def _make_output_dense(
            self,
            query_shape: Tuple[int, ...],
            common_kwargs: Dict[str, Any],
            name: Optional[str] = None
    ) -> EinsumDense:
        """Builds the output projection matrix.

        Args:
            query_shape: Shape of the query tensor.
            common_kwargs: Common keyword arguments for einsum layer.
            name: Name for the projection layer.

        Returns:
            Projection layer.
        """
        query_rank = len(query_shape)
        if self._output_shape:
            output_shape = self._output_shape
        else:
            output_shape = [query_shape[-1]]
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            query_rank - 1, bound_dims=2, output_dims=len(output_shape)
        )
        return EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(output_rank - 1, output_shape),
            bias_axes=bias_axes if self._use_bias else None,
            name=name,
            **common_kwargs,
        )

    def _build_attention(self, rank: int) -> None:
        """Builds multi-head dot-product attention computations.

        This function builds attributes necessary for `_compute_attention` to
        customize attention computation to replace the default dot-product
        attention with AdaptiveTemperatureSoftmax.

        Args:
            rank: the rank of query, key, value tensors.
        """
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)
        (
            self._dot_product_equation,
            self._combine_equation,
            attn_scores_rank,
        ) = _build_attention_equation(rank, attn_axes=self._attention_axes)
        norm_axes = tuple(
            range(
                attn_scores_rank - len(self._attention_axes), attn_scores_rank
            )
        )

        # Replace standard softmax with AdaptiveTemperatureSoftmax
        kwargs = {}
        if self._polynomial_coeffs is not None:
            kwargs["polynomial_coeffs"] = self._polynomial_coeffs

        self._adaptive_softmax = AdaptiveTemperatureSoftmax(
            min_temp=self._min_temp,
            max_temp=self._max_temp,
            entropy_threshold=self._entropy_threshold,
            dtype=self.dtype_policy,
            **kwargs
        )

        self._dropout_layer = Dropout(
            rate=self._dropout, dtype=self.dtype_policy, seed=self.seed
        )

    def _masked_softmax(
            self,
            attention_scores: backend.KerasTensor,
            attention_mask: Optional[backend.KerasTensor] = None
    ) -> backend.KerasTensor:
        """Apply masked AdaptiveTemperatureSoftmax to attention scores.

        Since AdaptiveTemperatureSoftmax doesn't handle masks directly in the same way as
        standard Softmax, we need to apply the mask to the logits before passing them to
        AdaptiveTemperatureSoftmax. This is done by setting masked positions to a large
        negative value, which will effectively be zero after softmax.

        Args:
            attention_scores: Raw attention scores tensor.
            attention_mask: Boolean mask to apply to attention scores.

        Returns:
            Normalized attention probabilities tensor.
        """
        # Normalize the attention scores to probabilities.
        # attention_scores = [B, N, T, S]
        if attention_mask is not None:
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims,
            # key_attention_dims>)
            mask_expansion_axis = -len(self._attention_axes) * 2 - 1
            for _ in range(
                    len(attention_scores.shape) - len(attention_mask.shape)
            ):
                attention_mask = ops.expand_dims(
                    attention_mask, axis=mask_expansion_axis
                )

            # Apply mask to the attention scores
            # Set masked positions to a large negative value
            attention_scores = ops.where(
                attention_mask,
                attention_scores,
                ops.constant(-1e9, dtype=attention_scores.dtype)
            )

        # Apply adaptive temperature softmax
        return self._adaptive_softmax(attention_scores)

    def _compute_attention(
            self,
            query: backend.KerasTensor,
            key: backend.KerasTensor,
            value: backend.KerasTensor,
            attention_mask: Optional[backend.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> Tuple[backend.KerasTensor, Optional[backend.KerasTensor]]:
        """Applies Dot-product attention with query, key, value tensors.

        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Uses AdaptiveTemperatureSoftmax for attention score
        normalization.

        Args:
            query: Projected query tensor of shape `(B, T, N, key_dim)`.
            key: Projected key tensor of shape `(B, S, N, key_dim)`.
            value: Projected value tensor of shape `(B, S, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions. It is generally not needed if
                the `query` and `value` (and/or `key`) are masked.
            training: Python boolean indicating whether the layer should behave
                in training mode (adding dropout) or in inference mode (doing
                nothing).

        Returns:
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
        """
        # Check for flash attention constraints
        if self._flash_attention and self._return_attention_scores:
            raise ValueError(
                "Returning attention scores is not supported when flash "
                "attention is enabled. Please disable flash attention to access"
                " attention scores."
            )

        # Determine whether to use dot-product attention
        use_dot_product_attention = not (
                self._dropout > 0.0
                or self._return_attention_scores
                or (len(query.shape) != 4)
        )

        if use_dot_product_attention:
            # Flash attention doesn't work with adaptive softmax, so we disable it here
            use_dot_product_attention = False

        if use_dot_product_attention:
            if attention_mask is not None:
                # Ensure attention_mask has the correct shape for broadcasting
                # Expected shape: [batch_size, num_heads, query_seq_len,
                # key_seq_len].
                mask_expansion_axis = -len(self._attention_axes) * 2 - 1
                len_attention_scores_shape = 4  # Only accepts 4D inputs
                for _ in range(
                        len_attention_scores_shape - len(attention_mask.shape)
                ):
                    attention_mask = ops.expand_dims(
                        attention_mask, axis=mask_expansion_axis
                    )
                attention_mask = ops.cast(attention_mask, dtype="bool")
            # Directly compute the attention output using dot-product attention
            attention_output = ops.dot_product_attention(
                query=query,
                key=key,
                value=value,
                bias=None,
                mask=attention_mask,
                scale=self._inverse_sqrt_key_dim,
                is_causal=False,
                flash_attention=self._flash_attention,
            )
            return attention_output, None

        # Scale query using inverse square root of key_dim
        query = ops.multiply(
            query, ops.cast(self._inverse_sqrt_key_dim, query.dtype)
        )

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = ops.einsum(self._dot_product_equation, key, query)

        # Apply the mask using the custom masked softmax with AdaptiveTemperatureSoftmax
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )

        # Apply dropout to the attention scores if needed
        if self._dropout > 0.0:
            final_attn_scores = self._dropout_layer(
                attention_scores, training=training
            )
        else:
            final_attn_scores = attention_scores

        # `context_layer` = [B, T, N, H]
        attention_output = ops.einsum(
            self._combine_equation, final_attn_scores, value
        )
        return attention_output, attention_scores

    def call(
            self,
            query: backend.KerasTensor,
            value: backend.KerasTensor,
            key: Optional[backend.KerasTensor] = None,
            query_mask: Optional[backend.KerasTensor] = None,
            value_mask: Optional[backend.KerasTensor] = None,
            key_mask: Optional[backend.KerasTensor] = None,
            attention_mask: Optional[backend.KerasTensor] = None,
            return_attention_scores: bool = False,
            training: Optional[bool] = None,
            use_causal_mask: bool = False,
    ) -> Union[backend.KerasTensor, Tuple[backend.KerasTensor, backend.KerasTensor]]:
        """Calls the layer with the given inputs.

        Args:
            query: Query tensor of shape `(B, T, dim)`.
            value: Value tensor of shape `(B, S, dim)`.
            key: Optional key tensor of shape `(B, S, dim)`.
            query_mask: Optional mask tensor for query.
            value_mask: Optional mask tensor for value.
            key_mask: Optional mask tensor for key.
            attention_mask: Optional mask tensor for attention scores.
            return_attention_scores: Whether to return attention scores.
            training: Training mode flag.
            use_causal_mask: Whether to use causal masking.

        Returns:
            attention_output: Output tensor.
            attention_scores: Optional attention scores if requested.
        """
        self._return_attention_scores = return_attention_scores
        if key is None:
            key = value

        # Delete the masks because the masks are handled at the level of the
        # layer
        query_mask = backend.get_keras_mask(query)
        backend.set_keras_mask(query, None)
        backend.set_keras_mask(value, None)
        backend.set_keras_mask(key, None)

        attention_mask = self._compute_attention_mask(
            query,
            value,
            query_mask=query_mask,
            value_mask=value_mask,
            key_mask=key_mask,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        # `query` = [B, T, N, H]
        query = self._query_dense(query)

        # `key` = [B, S, N, H]
        key = self._key_dense(key)

        # `value` = [B, S, N, H]
        value = self._value_dense(value)
        attention_output, attention_scores = self._compute_attention(
            query,
            key,
            value,
            attention_mask,
            training,
        )
        attention_output = self._output_dense(attention_output)

        # Set mask on output if needed
        if query_mask is not None:
            backend.set_keras_mask(attention_output, query_mask)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

    def _compute_attention_mask(
            self,
            query: backend.KerasTensor,
            value: backend.KerasTensor,
            query_mask: Optional[backend.KerasTensor] = None,
            value_mask: Optional[backend.KerasTensor] = None,
            key_mask: Optional[backend.KerasTensor] = None,
            attention_mask: Optional[backend.KerasTensor] = None,
            use_causal_mask: bool = False,
    ) -> Optional[backend.KerasTensor]:
        """Computes the attention mask, using the Keras masks of the inputs.

        * The `query`'s mask is reshaped from [B, T] to [B, T, 1].
        * The `value`'s mask is reshaped from [B, S] to [B, 1, S].
        * The `key`'s mask is reshaped from [B, S] to [B, 1, S]. The `key`'s
          mask is ignored if `key` is `None` or if `key is value`.
        * If `use_causal_mask=True`, then the causal mask is computed. Its shape
          is [1, T, S].

        All defined masks are merged using a logical AND operation (`&`).

        Args:
            query: Query tensor.
            value: Value tensor.
            query_mask: Mask for query.
            value_mask: Mask for value.
            key_mask: Mask for key.
            attention_mask: Explicit attention mask.
            use_causal_mask: Whether to use causal masking.

        Returns:
            Final attention mask or None if no masks specified.
        """
        auto_mask = None
        if query_mask is not None:
            query_mask = ops.cast(query_mask, "bool")  # defensive casting
            # B = batch size, T = max query length
            auto_mask = ops.expand_dims(query_mask, -1)  # shape is [B, T, 1]
        if value_mask is not None:
            value_mask = ops.cast(value_mask, "bool")  # defensive casting
            # B = batch size, S == max value length
            mask = ops.expand_dims(value_mask, -2)  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if key_mask is not None:
            key_mask = ops.cast(key_mask, "bool")  # defensive casting
            # B == batch size, S == max key length == max value length
            mask = ops.expand_dims(key_mask, -2)  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if use_causal_mask:
            # the shape of the causal mask is [1, T, S]
            mask = self._compute_causal_mask(query, value)
            auto_mask = mask if auto_mask is None else auto_mask & mask

        if attention_mask is not None:
            attention_mask = ops.cast(attention_mask, "bool")
        if auto_mask is not None:
            # merge attention_mask & automatic mask, to shape [B, T, S]
            attention_mask = (
                auto_mask
                if attention_mask is None
                else attention_mask & auto_mask
            )
        return attention_mask

    def _compute_causal_mask(
            self,
            query: backend.KerasTensor,
            value: Optional[backend.KerasTensor] = None
    ) -> backend.KerasTensor:
        """Computes a causal mask (e.g., for masked self-attention layers).

        For example, if query and value both contain sequences of length 4,
        this function returns a boolean tensor equal to:

        ```
        [[[True,  False, False, False],
          [True,  True,  False, False],
          [True,  True,  True,  False],
          [True,  True,  True,  True]]]
        ```

        Args:
            query: query tensor of shape `(B, T, ...)`.
            value: value tensor of shape `(B, S, ...)` (optional, defaults to
                query).

        Returns:
            mask: a boolean tensor of shape `(1, T, S)` containing a lower
                triangular matrix of shape `(T, S)`.
        """
        q_seq_length = ops.shape(query)[1]
        v_seq_length = q_seq_length if value is None else ops.shape(value)[1]
        ones_mask = ops.ones((1, q_seq_length, v_seq_length), dtype="int32")
        row_index = ops.cumsum(ones_mask, axis=-2)
        col_index = ops.cumsum(ones_mask, axis=-1)
        return ops.greater_equal(row_index, col_index)

    def compute_output_shape(
            self,
            query_shape: Tuple[int, ...],
            value_shape: Tuple[int, ...],
            key_shape: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[int, ...]:
        """Computes the output shape based on input shapes.

        Args:
            query_shape: Shape of query tensor.
            value_shape: Shape of value tensor.
            key_shape: Optional shape of key tensor.

        Returns:
            Output tensor shape.

        Raises:
            ValueError: If value and key shapes don't match.
        """
        query_shape = tuple(query_shape)
        value_shape = tuple(value_shape)
        if key_shape is None:
            key_shape = value_shape
        else:
            key_shape = tuple(key_shape)

        if value_shape[1:-1] != key_shape[1:-1]:
            raise ValueError(
                "All dimensions of `value` and `key`, except the last one, "
                f"must be equal. Received: value_shape={value_shape} and "
                f"key_shape={key_shape}"
            )
        if self._output_shape:
            query_shape = query_shape[:-1] + self._output_shape
        return query_shape

    def compute_output_spec(
            self,
            query: backend.KerasTensor,
            value: backend.KerasTensor,
            key: Optional[backend.KerasTensor] = None,
            query_mask: Optional[backend.KerasTensor] = None,
            value_mask: Optional[backend.KerasTensor] = None,
            key_mask: Optional[backend.KerasTensor] = None,
            attention_mask: Optional[backend.KerasTensor] = None,
            return_attention_scores: bool = False,
            training: Optional[bool] = None,
            use_causal_mask: bool = False,
    ) -> Union[backend.KerasTensor, Tuple[backend.KerasTensor, backend.KerasTensor]]:
        """Computes the output tensor spec.

        Args:
            query: Query tensor.
            value: Value tensor.
            key: Optional key tensor.
            query_mask: Optional mask for query.
            value_mask: Optional mask for value.
            key_mask: Optional mask for key.
            attention_mask: Optional attention mask.
            return_attention_scores: Whether to return attention scores.
            training: Training mode flag.
            use_causal_mask: Whether to use causal masking.

        Returns:
            Output tensor spec or tuple of output and attention score specs.
        """
        if key is not None:
            key_shape = key.shape
        else:
            key_shape = None
        output_shape = self.compute_output_shape(
            query.shape, value.shape, key_shape
        )
        output_spec = backend.KerasTensor(
            output_shape, dtype=self.compute_dtype
        )
        if return_attention_scores:
            length = query.shape[1]
            attention_shape = (query.shape[0], self.num_heads, length, length)
            return output_spec, backend.KerasTensor(
                attention_shape, dtype=self.compute_dtype
            )
        return output_spec


# Utility functions needed by MultiHeadAttention
def _index_to_einsum_variable(i: int) -> str:
    """Converts an index to a einsum variable name.

    Args:
        i: Index to convert.

    Returns:
        String corresponding to the index in einsum notation.
    """
    return string.ascii_lowercase[i]


def _build_attention_equation(
        rank: int, attn_axes: Tuple[int, ...]
) -> Tuple[str, str, int]:
    """Builds einsum equations for the attention computation.

    Args:
        rank: Rank of query, key, value tensors.
        attn_axes: List/tuple of axes to apply attention over.

    Returns:
        Tuple of dot product equation, combination equation, and attention scores rank.
    """
    target_notation = ""
    for i in range(rank):
        target_notation += _index_to_einsum_variable(i)
    # `batch_dims` includes the head dim.
    batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
    letter_offset = rank
    source_notation = ""
    for i in range(rank):
        if i in batch_dims or i == rank - 1:
            source_notation += target_notation[i]
        else:
            source_notation += _index_to_einsum_variable(letter_offset)
            letter_offset += 1

    product_notation = "".join(
        [target_notation[i] for i in batch_dims]
        + [target_notation[i] for i in attn_axes]
        + [source_notation[i] for i in attn_axes]
    )
    dot_product_equation = "%s,%s->%s" % (
        source_notation,
        target_notation,
        product_notation,
    )
    attn_scores_rank = len(product_notation)
    combine_equation = "%s,%s->%s" % (
        product_notation,
        source_notation,
        target_notation,
    )
    return dot_product_equation, combine_equation, attn_scores_rank


def _build_proj_equation(
        free_dims: int, bound_dims: int, output_dims: int
) -> Tuple[str, str, int]:
    """Builds an einsum equation for projections inside multi-head attention.

    Args:
        free_dims: Number of free dimensions.
        bound_dims: Number of bound dimensions.
        output_dims: Number of output dimensions.

    Returns:
        Tuple of equation string, bias axes string, and output rank.
    """
    input_str = ""
    kernel_str = ""
    output_str = ""
    bias_axes = ""
    letter_offset = 0
    for i in range(free_dims):
        char = _index_to_einsum_variable(i + letter_offset)
        input_str += char
        output_str += char

    letter_offset += free_dims
    for i in range(bound_dims):
        char = _index_to_einsum_variable(i + letter_offset)
        input_str += char
        kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
        char = _index_to_einsum_variable(i + letter_offset)
        kernel_str += char
        output_str += char
        bias_axes += char
    equation = f"{input_str},{kernel_str}->{output_str}"

    return equation, bias_axes, len(output_str)


def _get_output_shape(
        output_rank: int, known_last_dims: List[int]
) -> List[Optional[int]]:
    """Gets the output shape for a given rank and known dimensions.

    Args:
        output_rank: Rank of output tensor.
        known_last_dims: List of known dimensions at the end.

    Returns:
        List representing output shape with None for unknown dimensions.
    """
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)

# ---------------------------------------------------------------------