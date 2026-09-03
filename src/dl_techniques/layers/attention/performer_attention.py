"""Linear-complexity attention, built by ``PerformerAttention``, approximating
the softmax kernel with random features (FAVOR+).

Standard attention is quadratic because softmax sits between the two
matmuls and blocks reassociation. FAVOR+ replaces the exponential kernel
with an inner product of explicit feature maps, so the three matrices
``phi(Q)``, ``phi(K)^T`` and ``V`` contract in the cheaper order: build one
``(F, d)`` summary of the sequence, then let every query read from it. Cost
falls from ``O(N^2 d)`` to ``O(N F d)`` and the ``N x N`` matrix is never
formed. A causal variant uses a running ``cumsum`` prefix summary instead,
at higher memory cost.

``call()`` has no ``attention_mask`` argument; masking is only available
through the ``causal`` flag. ``nb_features`` reduces the estimator's
variance, not its bias: measured error against a softmax reference
plateaus and does not shrink as ``nb_features`` grows. See the class
docstring's Note for the measurements.

References:
    - Choromanski et al., 2020. Rethinking Attention with Performers.
      (https://arxiv.org/abs/2009.14794)
    - Rahimi and Recht, 2007. Random Features for Large-Scale Kernel
      Machines. (the trigonometric feature map this uses)
    - Katharopoulos et al., 2020. Transformers are RNNs: Fast
      Autoregressive Transformers with Linear Attention.
      (https://arxiv.org/abs/2006.16236)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
"""

import math
import keras
from typing import Optional, Union, Tuple, Any, Dict

from .common import (
    compute_attention_scale,
    validate_head_divisibility
)
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.attention.performer_attention")
class PerformerAttention(keras.layers.Layer):
    """Performer attention: linear complexity via FAVOR+ kernel approximation.

    Approximates standard softmax attention in ``O(N)`` time and memory instead of
    ``O(N^2)``. Positive random feature maps ``phi(Q)`` and ``phi(K)`` are
    constructed such that ``exp(q . k) ~ <phi(q), phi(k)>``, after which
    ``phi(Q) (phi(K)^T V) / phi(Q) (phi(K)^T 1_N)`` gives the attention output
    without ever materializing the ``N x N`` matrix. The feature map is
    trigonometric — ``phi(x) = (1/sqrt(r)) [cos(w_i . x), sin(w_i . x)]``, scaled by
    ``exp(-||x||^2 / 2)`` on the query side.

    ``call()`` accepts ``(inputs, training, return_attention_scores)`` and has
    no ``attention_mask`` or ``mask`` parameter, unlike most siblings in this
    package. The attention factory registers this layer for construction only
    and documents the call-signature difference rather than renaming it, so a
    silently-ignored ``attention_mask`` never becomes possible — callers would
    otherwise believe padding was handled when it is not. Causal masking is
    supported only through the ``causal=True`` constructor flag, which selects
    the prefix-``cumsum`` path in ``_linear_attention``.

    The ``dim % num_heads`` check and the ``1 / sqrt(head_dim)`` temperature
    come from :mod:`~dl_techniques.layers.attention.common` rather than being
    re-derived here.

    Why this is O(N), the re-association:

    .. code-block:: text

        standard attention: softmax sits between the two matmuls, so it
        must see the whole N x N score matrix before anything hits V.

            Q [N, d] ─┐
                      ├─► QKᵀ [N, N] ─► softmax ─► @ V ─► out [N, d]
            K [N, d] ─┘        ▲
                               └─ materialized: O(N²·d)

        performer: replace exp(q·k) by <phi(q), phi(k)>. softmax
        disappears, three plain matmuls remain, and the cheaper pair
        contracts first.

            phi(K)ᵀ [F, N] ─┐
                            ├─► KV [F, d]      one summary of the
            V       [N, d] ─┘                  whole sequence
                              │
            phi(Q)  [N, F] ───┴─► @ KV ─► num [N, d]

            phi(K)ᵀ · 1_N ─► k_sum [F]
            phi(Q) · k_sum ─► z [N]  (+ 1e-6)

            out = num / z          cost O(N · F · d); the N x N
                                   matrix is never formed

        the summary is built once, then every query reads from it.

    Architecture:

    .. code-block:: text

        input [B, N, dim]
             │
             ▼
        to_qkv ('qkv_projection'): Dense(3*dim) -> split 3 -> heads
             │  q, k, v  [B, H, N, d_h]
             ▼
        q = q * (1/sqrt(d_h))
             │  scale applied before the feature map, since there is
             │  no score matmul; scale also multiplies the projection
             │  matrix below, so it lands on the query path twice
             ▼
        projection matrix, redrawn every call from
        keras.random.normal(seed=None)   [H, F/2, d_h]
             │  * ortho_scaling, only if ortho_scaling > 0 (default 0,
             │    skipped); a plain scalar multiply, not orthogonalization
             │  * scale, unconditionally
             ▼
        phi(x) = max(0, [cos(xWᵀ), sin(xWᵀ)] * feature_scale)   [B, H, N, F]
             │  phi(q) additionally scaled by exp(-‖q‖² / 2); phi(k) is not
             ▼
        fork on causal (N x N never formed in either branch):
             │
             causal=False   KV = phi(k)ᵀ · v          [B, H, F, d_h]
                             z  = phi(q) · Σ_n phi(k) + 1e-6
                             one summary of the whole sequence
             │
             causal=True    KV = cumsum_n(phi(k) ⊗ v)  prefix state
                             z  = phi(q) · cumsum_n phi(k) + 1e-6
                             a prefix summary per position; rank-5
                             intermediate, more memory-hungry
             ▼
        out = (phi(q) · KV) / z -> merge heads
             ▼
        to_out ('output_projection'): Dense(dim) -> dropout (optional)
             ▼
        output [B, N, dim]
             call() has no mask argument; only masking is the causal
             flag. return_attention_scores=True returns (output, None).

    Complexity:

    .. code-block:: text

        standard attention   O(N² · d)     the (N x N) score matrix
        performer            O(N · F · d)  F = nb_features

        F costs memory proportionally. It does not buy accuracy: the
        feature map is biased and its measured error against a softmax
        reference plateaus from F = 32 upward (see the class Note).
        Raising F narrows the per-redraw variance only.

    :param dim: Model dimensionality. Must be positive and divisible by num_heads.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param nb_features: Number of random features for the kernel approximation.
        Higher values cost proportionally more memory and shrink the per-redraw
        variance, but not the approximation error: this feature map is biased
        and its measured error plateaus near 0.78 (see the class Note). Not a
        quality dial.
    :type nb_features: int
    :param ortho_scaling: Scaling factor applied to the random projection matrix.
        When ``ortho_scaling > 0`` this applies a plain scalar multiplication to
        the random Gaussian projection; it does not perform orthogonal random
        feature construction (no QR / Gram-Schmidt step exists). ``0.0`` disables
        the scaling.
    :type ortho_scaling: float
    :param causal: Whether to use causal (autoregressive) attention masking.
        Selects the prefix-``cumsum`` path, which is still linear in ``N`` but
        materially more memory-hungry than the non-causal one.
    :type causal: bool
    :param dropout_rate: Dropout rate for attention weights, between 0 and 1.
    :type dropout_rate: float
    :param use_bias: Whether to use bias in Q, K, V projections.
    :type use_bias: bool
    :param kernel_initializer: Initializer for projection weight matrices.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for projection bias vectors.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for projection weights.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for projection biases.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional arguments for Layer base class.
    :type kwargs: Any

    :raises ValueError: If ``dim``, ``num_heads`` or ``nb_features`` is not
        positive.
    :raises ValueError: If ``dim`` is not divisible by ``num_heads``.
    :raises ValueError: If ``dropout_rate`` is outside ``[0, 1]``.
    :raises ValueError: From ``build()``, if the input is not 3D or its trailing
        dimension does not equal ``dim``.

    Input shape:
        3D tensor with shape ``(batch_size, seq_len, dim)``. There is no mask
        argument; padding is not handled.

    Output shape:
        3D tensor with shape ``(batch_size, seq_len, dim)`` — unchanged from the
        input. With ``return_attention_scores=True`` the return is
        ``(output, None)``: no attention matrix exists to return.

    Example:
        >>> # Long-sequence encoder attention
        >>> attn = PerformerAttention(dim=512, num_heads=8, nb_features=256)
        >>> x = keras.random.normal((2, 8192, 512))
        >>> y = attn(x, training=False)                  # (2, 8192, 512)
        >>>
        >>> # Autoregressive: the only masking this layer offers
        >>> attn = PerformerAttention(dim=512, num_heads=8, causal=True)
        >>>
        >>> # Interface compatibility; the second entry is always None
        >>> y, scores = attn(x, return_attention_scores=True)

    Note:
        The projection matrix is redrawn on every call, so two forward passes over
        the same input differ by the sampling noise of the kernel approximation.
        This layer is an approximation of softmax attention, not an exact
        reformulation of it — unlike, say, blockwise/online-softmax attention.

        ``nb_features`` controls variance, not accuracy. Measured relative
        error against a softmax reference built from this layer's own
        weights (float32, num_heads=1, batch=1, seq_len=64, head_dim=16,
        mean of 20 projection redraws, score taken before ``to_out``)::

            features m :      8       32      128      512     2048     8192
            rel. error : 0.7773   0.7729   0.7714   0.7715   0.7714   0.7714
            std dev    : 0.0060   0.0019   0.0014   0.0005   0.0003   0.0002

        The per-redraw variance shrinks with ``m`` as a Monte-Carlo estimator
        should; the bias plateau does not move. The plateau level depends on
        the harness (0.577 at num_heads=4, batch=4 instead of 0.771). Neither
        the query-side ``exp(-||x||^2/2)`` factor (paired max delta 7.3e-09 at
        float64) nor the ``ops.maximum(features, 0)`` clamp (removing it
        lowers the measured error slightly) isolates the cause under
        ablation; the clamp is kept because an unclamped feature is signed
        and removing it changes every shipped checkpoint's output. Two
        alternative feature maps were tried and rejected (decisions.md
        D-013): a fixed trigonometric map and a textbook positive FAVOR+ map
        both converge under symmetric ``d^(-1/4)`` scaling, but measure 82.7
        and 2.08 at ``m=128`` against this map's 0.78.

    Attributes:
        to_qkv: Fused Q/K/V projection, ``3 * dim`` wide.
        to_out: Output projection back to ``dim``.
        dropout: Output dropout, or ``None`` at rate 0.
        head_dim: ``dim // num_heads``.
        scale: The ``1 / sqrt(head_dim)`` temperature, a Python float.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            nb_features: int = 256,
            ortho_scaling: float = 0.0,
            causal: bool = False,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create the projections and dropout.

        This layer owns no weights of its own: the random projection is resampled
        per call rather than stored. See the class docstring for the parameter
        reference.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        validate_head_divisibility(dim, num_heads)
        if nb_features <= 0:
            raise ValueError(f"nb_features must be positive, got {nb_features}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.dim = dim
        self.num_heads = num_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.causal = causal
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        # Normalize regularizers via regularizers.get() so str/dict/object/None
        # all round-trip uniformly through regularizers.serialize() in get_config.
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Computed attributes
        self.head_dim = dim // num_heads
        self.scale = compute_attention_scale(self.head_dim)
        self._feature_scale = math.sqrt(2.0 / float(self.nb_features))

        # Q, K, V projection layer
        self.to_qkv = keras.layers.Dense(
            3 * dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='qkv_projection'
        )

        # Output projection layer
        self.to_out = keras.layers.Dense(
            dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='output_projection'
        )

        # DECISION plan-2026-08-27T040114-580f8b63/D-014: Dropout is created
        # unconditionally, gated in call() -- creating it only when dropout_rate > 0
        # made auto-generated sub-layer names depend on the rate. See decisions.md.
        self.dropout = keras.layers.Dropout(dropout_rate, name="dropout")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and its two projection sub-layers.

        Both projections take the same input shape: ``to_out`` consumes
        ``(B, N, dim)`` because heads are merged back before it runs.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not rank 3, or if its last
            dimension does not equal ``dim``.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input, got shape {input_shape}")
        if input_shape[-1] != self.dim:
            raise ValueError(
                f"Last dimension of input ({input_shape[-1]}) must match dim ({self.dim})"
            )

        # Build sub-layers
        self.to_qkv.build(input_shape)
        self.to_out.build(input_shape)

        super().build(input_shape)

    def _create_projection_matrix(self, batch_size: int) -> keras.KerasTensor:
        """Create the random projection matrix for the FAVOR+ feature map.

        Resampled on every call, from an unseeded ``keras.random.normal``, so it is
        not a weight and does not round-trip through a checkpoint. Note that
        ``ortho_scaling`` only rescales here — nothing orthogonalizes.

        :param batch_size: Batch size for broadcasting.
        :type batch_size: int

        :return: Projection matrix of shape
            ``(batch, num_heads, nb_features//2, head_dim)``.
        :rtype: keras.KerasTensor
        """
        # Shape: (num_heads, nb_features//2, head_dim)
        shape = (self.num_heads, self.nb_features // 2, self.head_dim)
        # DECISION plan-2026-08-27T040114-580f8b63/D-014: dtype=self.compute_dtype
        # is required -- without it random.normal follows the global float policy
        # and mixed_float16 crashes the einsum below on a dtype mismatch. See decisions.md.
        projection = keras.random.normal(
            shape=shape,
            mean=0.0,
            stddev=1.0,
            seed=None,
            dtype=self.compute_dtype
        )

        # ortho_scaling rescales the projection; it does not orthogonalize it.
        if self.ortho_scaling > 0:
            projection = projection * self.ortho_scaling

        # Scale by sqrt(head_dim) for proper variance
        projection = projection * self.scale

        # Broadcast to batch dimension: (batch, num_heads, nb_features//2, head_dim)
        projection = keras.ops.expand_dims(projection, axis=0)
        projection = keras.ops.repeat(projection, batch_size, axis=0)

        return projection

    def _create_kernel_features(
            self,
            x: keras.KerasTensor,
            projection_matrix: keras.KerasTensor,
            is_query: bool = True
    ) -> keras.KerasTensor:
        """Map inputs into the trigonometric random feature space.

        ``cos`` and ``sin`` of the projection are concatenated to width
        ``nb_features``, scaled so the inner product approximates the exponential
        kernel in expectation, and clamped non-negative. The
        ``exp(-||x||^2 / 2)`` factor is applied to queries only; it nearly cancels
        between the numerator and denominator downstream, except against the
        ``1e-6`` :meth:`_linear_attention` adds to the normalizer. See the class
        docstring's Note for the paired measurement of what survives.

        :param x: Input tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type x: keras.KerasTensor
        :param projection_matrix: Random projection matrix.
        :type projection_matrix: keras.KerasTensor
        :param is_query: Whether this is for queries (affects normalization).
        :type is_query: bool

        :return: Random features of shape ``(batch, num_heads, seq_len, nb_features)``.
        :rtype: keras.KerasTensor
        """
        # x: (batch, num_heads, seq_len, head_dim)
        # projection_matrix: (batch, num_heads, nb_features//2, head_dim)
        # Result: (batch, num_heads, seq_len, nb_features//2)
        x_projected = keras.ops.einsum('bhnd,bhfd->bhnf', x, projection_matrix)

        features_cos = keras.ops.cos(x_projected)
        features_sin = keras.ops.sin(x_projected)

        # Shape: (batch, num_heads, seq_len, nb_features)
        features = keras.ops.concatenate([features_cos, features_sin], axis=-1)

        # Scaling ensures E[phi(x)^T phi(y)] ~ exp(x^T y)
        features = features * self._feature_scale

        if is_query:
            features = (
                    features *
                    keras.ops.exp(
                        -keras.ops.square(
                            keras.ops.norm(x, axis=-1, keepdims=True)
                        ) / 2.0
                    )
            )

        # Clamp to non-negative: an unclamped feature is signed, and removing
        # this changes every shipped checkpoint's output. See the class Note.
        return keras.ops.maximum(features, 0)

    def _linear_attention(
            self,
            q: keras.KerasTensor,
            k: keras.KerasTensor,
            v: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Contract the feature maps in the order that keeps the cost linear.

        The non-causal branch builds one ``(F, d_h)`` summary of the whole
        sequence; the causal branch builds a prefix summary per position via
        ``cumsum`` over the per-position outer products. Neither branch forms the
        ``N x N`` matrix. Both add ``1e-6`` to the normalizer.

        :param q: Query features of shape ``(batch, num_heads, seq_len, nb_features)``.
        :type q: keras.KerasTensor
        :param k: Key features of shape ``(batch, num_heads, seq_len, nb_features)``.
        :type k: keras.KerasTensor
        :param v: Value tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type v: keras.KerasTensor

        :return: Attention output of shape ``(batch, num_heads, seq_len, head_dim)``.
        :rtype: keras.KerasTensor
        """
        if self.causal:
            # k: (B, h, N, F), v: (B, h, N, D). Per-position outer product
            # k (x) v -> (B, h, N, F, D); cumsum over the sequence axis gives the
            # prefix-summed KV state.
            kv_cumsum = keras.ops.cumsum(
                keras.ops.einsum('bhnf,bhnd->bhnfd', k, v),
                axis=2
            )
            k_cumsum = keras.ops.cumsum(k, axis=2)

            # q:(B,h,N,F) contracts against kv_cumsum:(B,h,N,F,D) -> (B,h,N,D) directly.
            z_causal = keras.ops.einsum('bhnf,bhnf->bhn', q, k_cumsum) + 1e-6
            out = keras.ops.einsum('bhnf,bhnfd->bhnd', q, kv_cumsum)
            out = out / keras.ops.expand_dims(z_causal, axis=-1)
        else:
            # kv: (batch, num_heads, nb_features, head_dim)
            kv = keras.ops.einsum('bhnf,bhnd->bhfd', k, v)

            # k_sum: (batch, num_heads, nb_features)
            k_sum = keras.ops.sum(k, axis=2)

            # z: (batch, num_heads, seq_len)
            z = keras.ops.einsum('bhnf,bhf->bhn', q, k_sum)
            z = z + 1e-6

            # out: (batch, num_heads, seq_len, head_dim)
            out = keras.ops.einsum('bhnf,bhfd->bhnd', q, kv)
            out = out / keras.ops.expand_dims(z, axis=-1)

        return out

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            return_attention_scores: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, None]]:
        """Apply Performer attention to inputs.

        Project, reshape to heads, scale the queries, draw a fresh projection
        matrix, build both feature maps, contract linearly, then merge heads and
        project out. Note the frozen signature: there is no mask argument.

        :param inputs: Input tensor of shape ``(batch_size, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :param return_attention_scores: If ``True``, returns ``(output, None)``
            for compatibility. Performer does not compute explicit attention matrices.
        :type return_attention_scores: bool

        :return: Output tensor of shape ``(batch_size, seq_len, dim)``.
            If return_attention_scores is ``True``, returns ``(output, None)``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, None]]
        """
        batch_size = keras.ops.shape(inputs)[0]
        seq_len = keras.ops.shape(inputs)[1]

        # Shape: (batch, seq_len, 3*dim)
        qkv = self.to_qkv(inputs)

        # Each has shape: (batch, seq_len, dim)
        q, k, v = keras.ops.split(qkv, 3, axis=-1)

        # Reshape to multi-head format: (batch, num_heads, seq_len, head_dim)
        q = keras.ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        q = keras.ops.transpose(q, (0, 2, 1, 3))

        k = keras.ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = keras.ops.transpose(k, (0, 2, 1, 3))

        v = keras.ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = keras.ops.transpose(v, (0, 2, 1, 3))

        q = q * self.scale

        projection_matrix = self._create_projection_matrix(batch_size)

        q_features = self._create_kernel_features(q, projection_matrix, is_query=True)
        k_features = self._create_kernel_features(k, projection_matrix, is_query=False)

        out = self._linear_attention(q_features, k_features, v)

        # Reshape back to (batch, seq_len, dim)
        out = keras.ops.transpose(out, (0, 2, 1, 3))
        out = keras.ops.reshape(out, (batch_size, seq_len, self.dim))

        out = self.to_out(out)

        if self.dropout_rate > 0.0:
            out = self.dropout(out, training=training)

        if return_attention_scores:
            # Performer computes no explicit attention matrix.
            return out, None

        return out

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        The output projection maps ``dim`` back to ``dim``, so the layer is
        shape-preserving.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]

        :return: Shape tuple of the output (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer for serialization.

        :return: Dictionary containing all configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'nb_features': self.nb_features,
            'ortho_scaling': self.ortho_scaling,
            'causal': self.causal,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config
