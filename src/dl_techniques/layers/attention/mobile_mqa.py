"""Multi-query attention for mobile and edge vision models, the :class:`MobileMQA` layer.

``MobileMQA`` subclasses :class:`GroupedQueryAttention` with the group
count pinned to one, so all query heads share a single key/value head —
the K and V a decoder must re-read every step, and so where the memory
bandwidth goes. Three more differences from the parent: an optional
stride-2 depthwise convolution downsamples K and V before attention
(queries stay full resolution), the residual is a learnable scalar
``x + lambda * Attention(x)`` rather than a plain skip, and RoPE is
disabled (``rope_percentage`` forced to 0.0) in favor of explicit
positional embeddings or CNN-induced locality. Everything else —
projections, attention scale, probability normalizer, dropout, output
shape — is inherited unchanged.

With ``num_kv_heads = 1`` the grouped-query form collapses to
multi-query attention, one K/V head broadcast to every query head::

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    output             = x + lambda * W_o(Attention(x))

References:
    - Shazeer, 2019. Fast Transformer Decoding: One Write-Head is All You
      Need. (https://arxiv.org/abs/1911.02150)
    - Rombach et al., 2022. High-Resolution Image Synthesis with Latent
      Diffusion Models. (https://arxiv.org/abs/2112.10752)
"""

import keras
from typing import Tuple, Optional, Any, Dict, Union

from .group_query_attention import GroupedQueryAttention
from dl_techniques.utils.keras_registration import register_dl_technique

@register_dl_technique("dl_techniques.layers.attention.mobile_mqa")
class MobileMQA(GroupedQueryAttention):
    """
    Mobile Multi-Query Attention block with optional spatial downsampling and learnable residual.

    A specialized subclass of ``GroupedQueryAttention`` that enforces
    Multi-Query Attention (single KV head), supports optional spatial
    downsampling for Key/Value projections via depthwise convolution, and
    uses a learnable lambda-scaled residual connection
    ``output = input + lambda * Attention(input)``.

    This class has no local ``compute_output_shape()``. It inherits
    :meth:`GroupedQueryAttention.compute_output_shape`, which returns
    ``tuple(input_shape)`` — correct here because the output is
    ``inputs + lambda * attention_output``, an addition against
    ``inputs``, so the shapes must already match: K/V downsampling
    shortens only the key/value sequence, never the query sequence, and
    ``w_o`` projects back to ``dim``. Do not add an override that
    re-derives ``tuple(input_shape)`` — a second copy tracks the parent
    for no behavioral gain and silently shadows it if the parent's shape
    contract changes.

    Also inherited rather than re-created: ``w_q``/``w_k``/``w_v``/``w_o``,
    ``self.scale`` (the precomputed Python-float attention scale),
    ``self.attn_prob``, ``self.dropout`` and the optional
    ``q_norm``/``k_norm``. Only ``call()`` is overridden, plus the
    ``downsample`` conv and the ``lambda`` weight added here.

    Architecture:

    .. code-block:: text

        inputs [B, H, W, C]  (a spatial map)
                │
        ┌───────┴────────┐
        ▼                 ▼
        w_q          w_k, w_v (from GQA;
        (from GQA)   num_kv_heads=1, so
        │            each emits one head)
        │                 ▼
        │            downsample (optional)
        │            DepthwiseConv2D stride 2,
        │            on K and V only
        ▼                 ▼
        q [B,heads,N,d]  k,v [B,1,M,d]
        N = H*W          M = N or about N/4
        │                 │
        ▼                 ▼
        q_norm (opt.)   k_norm (opt.)
        │                 ▼
        │            repeat over axis 1,
        │            num_heads times: one
        │            head's weights serve
        │            every query head
        └────────┬────────┘
                  ▼
        S = q . k^T * scale   [B, heads, N, M]
                  ▼
        attn_prob(S) -> dropout -> A . v
                  ▼
        transpose, reshape    [B, H, W, C]
                  ▼
                 w_o
                  ▼
        inputs + lambda_param * w_o(...)
        (lambda is a trainable scalar, init 1.0)
                  ▼
        output [B, H, W, C]

    ``attention_mask`` is accepted but never applied — there is no mask
    code on this path, so padding keeps its full weight; see the
    ``attention_mask`` parameter note below for the measured damage.
    ``return_attention_weights=True`` returns ``(output, A)``, with ``A``
    of shape ``[B, heads, N, M]``.

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
        # Validation happens one frame down, in
        # GroupedQueryAttention._validate_inputs. Do not add a local dim<=0
        # check: it would raise a different message before the parent's and
        # break any test pinned on the parent's text.

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

        # Runs on the projected K/V, not the input: w_k/w_v emit head_dim
        # channels since num_kv_heads is 1.
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
        :param attention_mask: Accepted for compatibility with the
            package's attention contract, but ignored: no code on this
            path reads it, since downsampling K/V changes the key/value
            sequence length and a general token mask has no unambiguous
            target here. Padding is not merely unmasked, it contaminates
            real positions: measured 2026-08-27, unit-scale padding moved
            a real position's output by roughly its own magnitude (2.77
            against a 2.72 baseline), and adversarial padding by 318
            against a baseline scale of 1 to 4. Do not feed this layer a
            padded batch.
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

        # 5. Broadcast the one K/V head to every query head: this is the
        # sharing that makes the layer multi-query.
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

        # A trainable scalar on the residual branch, not a plain skip.
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

        # num_kv_heads and rope_percentage aren't constructor arguments here;
        # leaving them would make from_config(get_config()) raise a
        # duplicate-keyword TypeError, since the parent receives them through
        # the kwargs dict this class populates in __init__.
        params_to_remove = ['num_kv_heads', 'rope_percentage']
        for param in params_to_remove:
            config.pop(param, None)

        config.update({
            "use_downsampling": self.use_downsampling,
        })
        return config
