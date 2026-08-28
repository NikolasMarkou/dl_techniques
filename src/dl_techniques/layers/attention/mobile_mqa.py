"""
Multi-query attention for mobile and edge vision models.

MobileMQA is a subclass of Grouped Query Attention with the group count pinned
to one. All query heads share ONE key/value head. That is the whole idea: K and
V are what a decoder must re-read every step, so shrinking them to a single
head is where the memory bandwidth goes.

Architecture:
    Four differences from the parent GQA:

    1.  **One shared K/V head.** ``num_kv_heads`` is forced to 1 in
        ``__init__`` and is not a constructor argument. All ``num_heads``
        query heads read the same K and the same V.
    2.  **Optional spatial downsampling.** A stride-2 depthwise convolution
        runs on the K and V feature maps before attention. Q stays at full
        resolution, so the score matrix is ``(N, M)`` with ``M`` about
        ``N / 4`` instead of ``(N, N)``.
    3.  **Learnable residual.** The output is ``x + lambda * Attention(x)``
        with a trainable scalar ``lambda`` initialized to 1.0, not a plain
        skip connection.
    4.  **No RoPE.** ``rope_percentage`` is forced to 0.0. MobileMQA relies on
        explicit positional embeddings or on CNN-induced locality instead.

    Everything else is inherited from ``GroupedQueryAttention``: the four
    ``Dense`` projections, the precomputed attention scale, the
    ``ProbabilityOutput`` normalizer, the attention dropout and
    ``compute_output_shape()``. See the ``[REUSE]`` note on the class below.

Foundational Mathematics:
    With ``num_kv_heads = 1`` the grouped-query form collapses to multi-query
    attention: one K,V head broadcast to all ``num_heads`` query heads::

        Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V
        output             = x + lambda * W_o( Attention(x) )

    Optional stride-2 depthwise downsampling shortens the key/value sequence
    from ``N = H*W`` to ``M = ceil(H/2)*ceil(W/2)``, roughly ``N / 4``.

References:
    - Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need."
    - Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models."
"""

# ---------------------------------------------------------------------

import keras
from typing import Tuple, Optional, Any, Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .group_query_attention import GroupedQueryAttention

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MobileMQA(GroupedQueryAttention):
    """
    Mobile Multi-Query Attention block with optional spatial downsampling and learnable residual.

    A specialized subclass of ``GroupedQueryAttention`` that enforces
    Multi-Query Attention (single KV head), supports optional spatial
    downsampling for Key/Value projections via depthwise convolution, and
    uses a learnable lambda-scaled residual connection
    ``output = input + lambda * Attention(input)``.

    **[REUSE] This class has NO local ``compute_output_shape()``, on purpose.**

    It inherits ``GroupedQueryAttention.compute_output_shape()``, which returns
    ``tuple(input_shape)``. That is right here, and it is not an accident of
    omission. The output is ``inputs + lambda * attention_output``, an addition
    against ``inputs``, so the shapes must already match: K/V downsampling
    shortens only the key/value sequence, never the query sequence, and ``w_o``
    projects back to ``dim``.

    WHAT NOT TO DO: don't add an override that re-derives
    ``tuple(input_shape)``. A second copy is a second thing to keep in step
    with the parent for no behavioral gain, and if the parent's shape contract
    changes, the silently-shadowing override is what breaks.

    Also inherited rather than re-created: ``w_q``/``w_k``/``w_v``/``w_o``,
    ``self.scale`` (the precomputed Python-float attention scale),
    ``self.attn_prob``, ``self.dropout`` and the optional ``q_norm``/``k_norm``.
    Only ``call()`` is overridden, plus the ``downsample`` conv and the
    ``lambda`` weight added here.

    **Architecture Overview:**

    .. code-block:: text

                 inputs  [B, H, W, C]  (a spatial map)
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
          ┌─────────────┐        ┌────────────────────┐
          │ w_q         │        │ w_k  and  w_v      │
          │ Dense, from │        │ Dense, from GQA.   │
          │ GQA         │        │ num_kv_heads == 1, │
          │             │        │ so each emits ONE  │
          │             │        │ head's worth       │
          └──────┬──────┘        └─────────┬──────────┘
                 │                         ▼
                 │              ┌────────────────────┐
                 │              │ downsample         │
                 │              │ DepthwiseConv2D,   │
                 │              │ stride 2 (optional)│
                 │              │ K and V ONLY       │
                 │              └─────────┬──────────┘
                 ▼                        ▼
          q [B, heads, N, d]      k, v [B, 1, M, d]
          N = H * W               M = N, or about N/4
                 │                        │
                 ▼                        ▼
          q_norm (optional)        k_norm (optional)
                 │                        │
                 │                        ▼
                 │              ┌────────────────────┐
                 │              │ SHARED K/V:        │
                 │              │ ops.repeat over    │
                 │              │ axis 1, num_heads  │
                 │              │ times. One head's  │
                 │              │ weights serve      │
                 │              │ every query head.  │
                 │              └─────────┬──────────┘
                 └────────────┬───────────┘
                              ▼
                    S = q . k^T * scale     [B, heads, N, M]
                              ▼
                    attn_prob(S) -> dropout -> A . v
                              ▼
                    transpose, reshape  [B, H, W, C]
                              ▼
                             w_o
                              ▼
                inputs + lambda_param * w_o(...)
                a TRAINABLE scalar residual, lambda init 1.0
                              ▼
                    output  [B, H, W, C]

        attention_mask is accepted and NEVER applied. There is no mask
        code on this path, so padding keeps its full weight. The
        parameter note carries the measured damage.
        return_attention_weights=True returns (output, A), with A of
        shape [B, heads, N, M].


    :param dim: Input/output dimension. Must be positive and divisible
        by ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads. Defaults to 8.
    :type num_heads: int
    :param use_downsampling: Whether to use spatial downsampling
        (stride-2 ``DepthwiseConv2D``) for keys and values.
        Defaults to ``False``.
    :type use_downsampling: bool
    :param kernel_initializer: Initializer for kernels.
        Defaults to ``'he_normal'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernels.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kwargs: Additional arguments passed to
        ``GroupedQueryAttention``. Note that ``num_kv_heads`` and
        ``rope_percentage`` are overwritten unconditionally (to ``1`` and ``0.0``)
        and ``use_bias`` defaults to ``True`` here, not ``False`` as in the parent.

    :raises ValueError: Propagated from ``GroupedQueryAttention.__init__`` if
        ``dim`` or ``num_heads`` is not positive, or if ``dim`` is not divisible
        by ``num_heads``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        use_downsampling: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        """Pin the MQA configuration, then hand everything to the parent.

        :param dim: Input/output dimension. Must be positive and divisible by
            ``num_heads``.
        :type dim: int
        :param num_heads: Number of query heads.
        :type num_heads: int
        :param use_downsampling: Whether to build the stride-2 depthwise
            convolution for keys and values.
        :type use_downsampling: bool
        :param kernel_initializer: Initializer for kernels.
        :type kernel_initializer: str or keras.initializers.Initializer
        :param kernel_regularizer: Optional regularizer for kernels.
        :type kernel_regularizer: keras.regularizers.Regularizer or None
        :param kwargs: Forwarded to ``GroupedQueryAttention``.
        :type kwargs: Any

        :raises ValueError: Propagated from the parent's validation if ``dim``
            or ``num_heads`` is not positive, or if ``dim`` is not divisible by
            ``num_heads``.
        """
        # This `__init__` raises nothing itself, and that is on purpose. Every
        # constrained argument it takes is validated one frame down, by
        # `GroupedQueryAttention._validate_inputs`, which raises on non-positive
        # `dim`, non-positive `num_heads`, and `dim % num_heads != 0`. The two
        # arguments this subclass adds are `use_downsampling` (a bool, no range
        # to check) and the initializer/regularizer pair, which Keras validates
        # when it resolves them.
        #
        # WHAT NOT TO DO: don't add a local `if dim <= 0: raise ValueError(...)`
        # here. It duplicates the parent's check, and it raises a DIFFERENT
        # message BEFORE the parent's, which silently breaks any
        # `pytest.raises(..., match=...)` pinned on the parent's text. If
        # MobileMQA ever grows a constrained argument of its own, validate THAT
        # argument here and leave the inherited three alone.

        # num_kv_heads=1 is the definition of multi-query attention, and
        # rope_percentage=0.0 disables RoPE. Neither is a constructor argument,
        # so neither can be overridden by a caller.
        kwargs['dim'] = dim
        kwargs['num_heads'] = num_heads
        kwargs['num_kv_heads'] = 1
        kwargs['rope_percentage'] = 0.0
        # bias defaults to True here, matching CNN convention, where the parent
        # defaults to False.
        kwargs['use_bias'] = kwargs.get('use_bias', True)
        kwargs['kernel_initializer'] = kernel_initializer
        kwargs['kernel_regularizer'] = kernel_regularizer

        super().__init__(**kwargs)

        self.use_downsampling = use_downsampling

        # The depthwise conv exists only when downsampling is on.
        if self.use_downsampling:
            self.downsample = keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=2,
                padding="same",
                depthwise_initializer=self.kernel_initializer,
                depthwise_regularizer=self.kernel_regularizer,
                name="downsample"
            )
        else:
            self.downsample = None

        # The residual scalar is a weight, so it is created in build().
        self.lambda_param = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the parent's GQA weights plus this layer's two additions.

        The additions are the trainable residual scalar ``lambda_param`` and,
        when enabled, the K/V downsampling convolution. The parent's
        ``build()`` runs last because it is what finalizes layer state.

        :param input_shape: Shape tuple of the input tensor, expected
            ``(batch, height, width, dim)``.
        :type input_shape: tuple
        """
        if self.built:
            return

        # The learnable residual scalar. Initialized to 1.0, so the layer
        # starts as a plain skip plus full attention.
        self.lambda_param = self.add_weight(
            name="lambda",
            shape=(),
            initializer="ones",
            trainable=True,
            dtype=self.compute_dtype
        )

        # The downsample conv runs on the PROJECTED K/V, not on the input.
        # w_k and w_v emit num_kv_heads * head_dim channels, and num_kv_heads
        # is 1, so their output is (B, H, W, head_dim).
        if self.downsample is not None:
            if len(input_shape) == 4:
                kv_shape = list(input_shape)
                kv_shape[-1] = self.head_dim
                self.downsample.build(tuple(kv_shape))
            else:
                # This layer expects 4D input. A non-4D shape is left for the
                # parent build() and the forward pass to reject.
                pass

        # The parent build() creates w_q, w_k, w_v, w_o and finalizes layer
        # state, so it goes last.
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        attention_mask: Optional[keras.KerasTensor] = None,
        return_attention_weights: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Run multi-query attention over a spatial map, then the scaled residual.

        Project, optionally downsample K and V, flatten the spatial axes,
        broadcast the single K/V head to every query head, attend, project out,
        and add ``lambda_param`` times the result back to the input.

        :param inputs: Input tensor of shape
            ``(batch_size, height, width, dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training or inference mode.
        :type training: bool or None
        :param attention_mask: Accepted and **IGNORED**. It is in the signature
            only for compatibility with the package's attention contract; no
            code on this path reads it. Downsampling K/V changes the key/value
            sequence length, so a general token mask has no unambiguous target
            here.

            "No effect" understates the problem, so here is the measurement,
            taken 2026-08-27. Padding is not merely unmasked, it CONTAMINATES
            real positions. Unit-scale padding moved a real position's output by
            roughly its own magnitude, 2.77 against a 2.72 baseline. Adversarial
            padding moved it by 318 against a baseline scale of 1 to 4. Don't
            feed this layer a padded batch.
        :type attention_mask: keras.KerasTensor or None
        :param return_attention_weights: Whether to return attention
            weights alongside the output.
        :type return_attention_weights: bool
        :return: Output tensor of shape
            ``(batch_size, height, width, dim)``, or tuple of
            ``(output, attention_weights)`` if
            ``return_attention_weights=True``.
        :rtype: keras.KerasTensor or tuple[keras.KerasTensor, keras.KerasTensor]
        """
        input_shape = keras.ops.shape(inputs)
        batch_size = input_shape[0]
        height, width = input_shape[1], input_shape[2]

        # 1. Project Q, K, V with the inherited Dense layers. w_q emits
        # num_heads * head_dim channels; w_k and w_v emit head_dim, because
        # num_kv_heads is 1.
        q = self.w_q(inputs, training=training)
        k = self.w_k(inputs, training=training)
        v = self.w_v(inputs, training=training)

        # 2. Optional stride-2 downsampling, on K and V only. Q keeps its full
        # resolution, so the score matrix becomes (N, M) with M about N/4.
        if self.downsample is not None:
            k = self.downsample(k, training=training)
            v = self.downsample(v, training=training)

            # The K/V sequence length changed, so re-read it.
            kv_shape = keras.ops.shape(k)
            kv_height, kv_width = kv_shape[1], kv_shape[2]
            kv_len = kv_height * kv_width
        else:
            kv_len = height * width

        # 3. Flatten the spatial axes into one sequence axis.
        # Q becomes (B, H*W, num_heads, head_dim).
        q = keras.ops.reshape(q, (batch_size, height * width, self.num_heads, self.head_dim))

        # K and V get a head axis of size 1: MQA has one KV head.
        k = keras.ops.reshape(k, (batch_size, kv_len, 1, self.head_dim))
        v = keras.ops.reshape(v, (batch_size, kv_len, 1, self.head_dim))

        # 4. Move the head axis forward.
        # q -> (B, num_heads, S_q, head_dim); k and v -> (B, 1, S_kv, head_dim).
        q = keras.ops.transpose(q, (0, 2, 1, 3))
        k = keras.ops.transpose(k, (0, 2, 1, 3))
        v = keras.ops.transpose(v, (0, 2, 1, 3))

        # Optional q/k normalization (inherited from GroupedQueryAttention).
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # 5. Broadcast the ONE K/V head to every query head. This is the
        # sharing that makes the layer multi-QUERY: one set of K/V weights
        # serves all num_heads queries. num_kv_heads is 1, so num_groups
        # equals num_heads.
        k = keras.ops.repeat(k, self.num_heads, axis=1)
        v = keras.ops.repeat(v, self.num_heads, axis=1)

        # 6. Scores.
        # (B, heads, S_q, d) @ (B, heads, d, S_kv) -> (B, heads, S_q, S_kv).
        #
        # `self.scale` is the parent's precomputed Python float, made once in
        # GroupedQueryAttention.__init__. Don't recompute it here, and don't
        # reach for keras.ops.sqrt: a backend tensor built in __init__ can leak
        # out of a symbolic scratch graph. The parent's anchor beside
        # `self.scale = compute_attention_scale(...)` records why.
        scale = keras.ops.cast(self.scale, k.dtype)
        scores = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2))) * scale

        # No mask is applied. See the attention_mask parameter note: this layer
        # has no mask code, and the omission is documented rather than fixed
        # because downsampling leaves a token mask with no unambiguous target.

        attn_weights = self.attn_prob(scores)
        attn_weights = self.dropout(attn_weights, training=training)

        # 7. Weighted sum of V -> (B, heads, S_q, head_dim).
        out = keras.ops.matmul(attn_weights, v)

        # 8. Merge the heads and restore the spatial layout.
        # (B, heads, S_q, d) -> (B, S_q, heads, d) -> (B, H, W, dim).
        out = keras.ops.transpose(out, (0, 2, 1, 3))
        out = keras.ops.reshape(out, (batch_size, height, width, self.dim))

        # 9. Output projection, then the scaled residual.
        attention_output = self.w_o(out, training=training)

        # This is the layer's signature: a TRAINABLE scalar on the residual
        # branch, not a plain skip connection.
        output = inputs + self.lambda_param * attention_output

        if return_attention_weights:
            return output, attn_weights
        return output

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        Two of the parent's keys are dropped, for the reason given below.

        :return: Dictionary containing the layer configuration.
        :rtype: dict
        """
        config = super().get_config()

        # Drop the two parent keys this subclass pins in __init__:
        # num_kv_heads=1 and rope_percentage=0.0. Neither is a constructor
        # argument here, so leaving them in the config makes
        # `from_config(get_config())` raise a duplicate-keyword TypeError. The
        # parent receives them through the kwargs dict this class populates, and
        # a config-supplied copy collides with it.
        #
        # WHAT NOT TO DO: don't keep them for transparency. The key SET of this
        # config is part of the frozen serialization surface, and adding a key
        # breaks the round-trip of every existing .keras checkpoint.
        params_to_remove = ['num_kv_heads', 'rope_percentage']
        for param in params_to_remove:
            config.pop(param, None)

        config.update({
            "use_downsampling": self.use_downsampling,
        })
        return config

# ---------------------------------------------------------------------
