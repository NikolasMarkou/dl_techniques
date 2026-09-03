"""
RPCAttention, a robust attention layer that decomposes attention scores with
Principal Component Pursuit instead of using them raw.

A single ``Dense(3 * dim)`` produces Q, K and V. The scaled score matrix
``A = Q K^T / sqrt(d_k)`` is split by ``max_pcp_iter`` fixed sweeps of
alternating minimization into a low-rank component ``L`` (global patterns,
via singular value thresholding) and a sparse component ``S`` (localized
outliers, via soft thresholding), solving
``min_{L,S} ||L||_* + lambda ||S||_1  s.t.  A = L + S``. The recombined
``L + S`` is normalized and applied to ``V`` in place of a plain softmax, so
an adversarial perturbation concentrated on a few token pairs is routed into
``S`` instead of distorting the whole distribution.

The iteration count is a Python constant, not a convergence test, and the
loop has no early stop so it unrolls identically under graph tracing. Cost is
``max_pcp_iter`` batched SVDs per forward pass, making this layer far more
expensive than plain attention and unsuitable for long sequences. This layer
has no float16 SVD kernel outside XLA; see the class docstring's warning.
``call()``'s mask parameter is spelled ``mask=``, not ``attention_mask=``,
and is frozen — see the class docstring.

References:
    - Candes et al., 2011. Robust Principal Component Analysis? Journal of
      the ACM.
"""

# ---------------------------------------------------------------------

import keras
from typing import Optional, Union, Tuple, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.activations import ProbabilityOutput
from dl_techniques.layers.norms.factory import create_normalization_layer

from .common import (
    apply_attention_mask,
    compute_attention_scale,
    validate_head_divisibility,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.attention.rpc_attention")
class RPCAttention(keras.layers.Layer):
    """Robust Principal Components Attention via PCP decomposition.

    Implements RPC-Attention which decomposes the raw attention score matrix
    ``A = Q K^T / sqrt(d_k)`` into low-rank ``L`` and sparse ``S`` components
    using iterative alternating minimization (ADMM-style). The low-rank component
    is obtained via singular value thresholding:
    ``L = U diag(max(sigma - tau, 0)) V^H`` where ``(U, sigma, V^H) = SVD(A - S)``.
    The sparse component uses soft thresholding:
    ``S = sign(A - L) max(|A - L| - lambda, 0)``. After ``max_pcp_iter``
    iterations, the robust attention output is ``softmax(L + S) V``.

    ``call()``'s mask parameter is named ``mask``, not ``attention_mask`` as
    in most siblings; this spelling is frozen by the standing anchor
    ``plan_2026-06-14_0c5d4a21/D-007`` in ``factory.py``, since the factory
    and existing callers pass it by keyword. The public attribute
    ``self.attention_scale`` is likewise frozen: it is the only occurrence of
    that name in the package (every sibling uses ``self.scale``), but it is
    reachable from user code, so renaming it is an API change.

    The ``dim % num_heads`` check, the ``1 / sqrt(head_dim)`` temperature and
    the additive mask bias
    (:func:`~dl_techniques.layers.attention.common.apply_attention_mask`,
    which keeps the masked scores in ``>= float32`` so they can reach
    ``ops.svd``) come from :mod:`~dl_techniques.layers.attention.common``;
    score normalization comes from the shared
    :class:`~dl_techniques.layers.activations.ProbabilityOutput`; the optional
    Q/K norms come from
    :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`.

    .. warning::
        This layer cannot run under a ``mixed_float16`` policy on this
        backend: ``ops.svd`` has no float16 kernel outside XLA, so the first
        ``ops.svd`` inside :meth:`_pcp_decomposition` raises
        ``NotFoundError: Could not find device for node: {{node Svd}} =
        Svd[T=DT_HALF, ...]``. It raises with or without a mask; a masked
        fp16 forward happens to survive because the mask bias promotes the
        scores to float32 before the SVD (see the D-005 note below), but the
        unmasked path has nothing to promote it. ``float32`` and ``float64``
        policies are fully supported.

    Architecture:

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────┐
        │     RPCAttention — Robust PCA in place of a softmax     │
        │                                                         │
        │   The score matrix is split into a low-rank L plus a    │
        │   sparse S by iterative Principal Component Pursuit     │
        │   before any probability is taken. Not softmax(Q Kᵀ) V. │
        │                                                         │
        │                   Input  [B, S, dim]                    │
        │                            ▼                            │
        │    to_qkv Dense(3*dim) ► split 3 ► heads [B,H,S,d_h]    │
        │                            ▼                            │
        │                optional q_norm / k_norm                 │
        │                            ▼                            │
        │       A = Q Kᵀ * attention_scale     [B, H, S, S]       │
        │                            ▼                            │
        │   mask (if given): expand rank 2/3, keep = (mask != 0), │
        │   additive bias in float32 (the SVD has no fp16 kernel),│
        │   degenerate rows rescued on the probability axis       │
        │                            ▼                            │
        │  ┌── repeat max_pcp_iter times ──────────────────────┐  │
        │  │  L = SVT(A - S, svd_threshold)   batched ops.svd  │  │
        │  │  S = soft(A - L, lambda_sparse)  shrinkage        │  │
        │  │  (no early stop — fixed count, graph-safe)        │  │
        │  └───────────────────────────────────────────────────┘  │
        │                            ▼                            │
        │         cast (L + S) back to the compute dtype          │
        │                            ▼                            │
        │      attn = attn_prob(L + S)   (ProbabilityOutput)      │
        │                            ▼                            │
        │     out = attn @ V ► merge heads ► to_out ► dropout     │
        │                            ▼                            │
        │                   Output  [B, S, dim]                   │
        │                                                         │
        │   Known defect: return_attention_scores=True re-runs the│
        │   whole PCP on scores that were never masked, so the    │
        │   returned weights need not match the returned output.  │
        └─────────────────────────────────────────────────────────┘

    :param dim: Model dimensionality. Must be positive and divisible by num_heads.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param lambda_sparse: Sparsity regularization parameter for the ``S`` component.
        Higher values create sparser attention.
    :type lambda_sparse: float
    :param max_pcp_iter: Maximum iterations for PCP decomposition.
    :type max_pcp_iter: int
    :param svd_threshold: Threshold for singular value soft-thresholding in low-rank
        approximation.
    :type svd_threshold: float
    :param qkv_bias: Whether to use bias in Q, K, V projections.
    :type qkv_bias: bool
    :param dropout_rate: Dropout rate for output, between 0 and 1.
    :type dropout_rate: float
    :param kernel_initializer: Initializer for projection weight matrices.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for projection bias vectors.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for projection weights.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for projection biases.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param probability_type: String identifier for the attention-score
        normalization strategy, forwarded to
        :class:`~dl_techniques.layers.activations.ProbabilityOutput` as its
        ``probability_type``. Defaults to ``"softmax"``. Routing/hierarchical
        variants are rejected: they consume features rather than logits.
    :type probability_type: str
    :param probability_config: Optional dictionary forwarded to
        :class:`~dl_techniques.layers.activations.ProbabilityOutput` as
        ``type_config``. Defaults to ``None``.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied per-head to Q and K
        before the robust scoring loop, forwarded to
        :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`.
        ``None`` disables QK-norm. Defaults to ``None``.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        :func:`~dl_techniques.layers.norms.factory.create_normalization_layer` for
        both the Q and K norms. Defaults to ``None``.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Additional arguments for Layer base class.
    :type kwargs: Any

    :raises ValueError: If ``dim``, ``num_heads``, ``lambda_sparse``,
        ``max_pcp_iter`` or ``svd_threshold`` is not positive.
    :raises ValueError: If ``dim`` is not divisible by ``num_heads``.
    :raises ValueError: If ``dropout_rate`` is outside ``[0, 1]``.
    :raises ValueError: If ``probability_type`` is a routing/hierarchical variant.
    :raises ValueError: From ``build()``, if the input is not 3D or its trailing
        dimension does not equal ``dim``.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            lambda_sparse: float = 0.1,
            max_pcp_iter: int = 10,
            svd_threshold: float = 1.0,
            qkv_bias: bool = False,
            dropout_rate: float = 0.0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            probability_type: str = "softmax",
            probability_config: Optional[Dict[str, Any]] = None,
            qk_norm_type: Optional[str] = None,
            qk_norm_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create every sub-layer.

        Every argument is documented on the class. All sub-layers are built here
        rather than in :meth:`build`, so a layer that is never called still
        serializes and deserializes to the same object graph.

        :param kwargs: Forwarded to ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: On any invalid argument — see the class docstring for
            the full list.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        # Raises the same message test_invalid_dim_mismatch pins with a regex.
        validate_head_divisibility(dim, num_heads)
        if lambda_sparse <= 0:
            raise ValueError(f"lambda_sparse must be positive, got {lambda_sparse}")
        if max_pcp_iter <= 0:
            raise ValueError(f"max_pcp_iter must be positive, got {max_pcp_iter}")
        if svd_threshold <= 0:
            raise ValueError(f"svd_threshold must be positive, got {svd_threshold}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if probability_type in (
            "routing",
            "deterministic_routing",
            "hierarchical",
            "hierarchical_routing",
        ):
            raise ValueError(
                f"probability_type '{probability_type}' is not supported by "
                f"RPCAttention. Use a non-routing probability type."
            )

        # Store configuration
        self.dim = dim
        self.num_heads = num_heads
        self.lambda_sparse = lambda_sparse
        self.max_pcp_iter = max_pcp_iter
        self.svd_threshold = svd_threshold
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        # Normalize regularizers via regularizers.get() so str/dict/object/None
        # all round-trip uniformly through regularizers.serialize() in get_config.
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        self.head_dim = dim // num_heads
        # Attribute name is frozen as attention_scale, not scale (see class docstring);
        # a Python float computed here, never ops.sqrt in call().
        self.attention_scale = compute_attention_scale(self.head_dim)

        # Create sub-layers in __init__
        # Q, K, V projection layer
        self.to_qkv = keras.layers.Dense(
            3 * dim,
            use_bias=qkv_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='qkv_projection'
        )

        # Output projection layer
        self.to_out = keras.layers.Dense(
            dim,
            use_bias=qkv_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='output_projection'
        )

        # Dropout layer
        if dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(dropout_rate)
        else:
            self.dropout = None

        # Probability activation (shared between standard scoring and
        # the robust attention iteration / return_attention_scores path).
        self.attn_prob = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="attn_prob",
        )

        # Optional Q/K normalization layers
        if self.qk_norm_type is not None:
            self.q_norm = create_normalization_layer(
                self.qk_norm_type,
                name="q_norm",
                **(self.qk_norm_kwargs or {}),
            )
            self.k_norm = create_normalization_layer(
                self.qk_norm_type,
                name="k_norm",
                **(self.qk_norm_kwargs or {}),
            )
        else:
            self.q_norm = None
            self.k_norm = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and its sub-layers.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
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

        # Build attention-score probability activation.
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        score_shape = (batch_size, self.num_heads, seq_len, seq_len)
        self.attn_prob.build(score_shape)

        # Build Q/K normalization layers if configured.
        if self.q_norm is not None:
            qk_shape = (batch_size, self.num_heads, seq_len, self.head_dim)
            self.q_norm.build(qk_shape)
            self.k_norm.build(qk_shape)

        super().build(input_shape)

    def _soft_threshold(self, x: keras.KerasTensor, threshold: float) -> keras.KerasTensor:
        """Apply soft thresholding: ``S_lambda(x) = sign(x) max(|x| - lambda, 0)``.

        :param x: Input tensor.
        :type x: keras.KerasTensor
        :param threshold: Threshold value lambda.
        :type threshold: float

        :return: Soft-thresholded tensor.
        :rtype: keras.KerasTensor
        """
        return keras.ops.sign(x) * keras.ops.maximum(keras.ops.abs(x) - threshold, 0.0)

    def _nuclear_norm_minimization(
            self,
            matrix: keras.KerasTensor,
            threshold: float
    ) -> keras.KerasTensor:
        """Minimize nuclear norm via singular value thresholding.

        :param matrix: Input matrix to be approximated.
        :type matrix: keras.KerasTensor
        :param threshold: Threshold for singular values.
        :type threshold: float

        :return: Low-rank approximation of the input matrix.
        :rtype: keras.KerasTensor
        """
        # Perform SVD
        # Note: keras.ops.svd returns (u, s, v)
        # For TF backend, v corresponds to V^H (adjoint of right singular vectors)
        u, s, v = keras.ops.svd(matrix, full_matrices=False)

        # Apply soft thresholding to singular values
        s_thresholded = keras.ops.maximum(s - threshold, 0.0)

        # Reconstruct low-rank matrix
        # Need to construct a batch of diagonal matrices from s_thresholded
        # s_thresholded is (B, K)
        # We need s_diag to be (B, K, K)

        k = keras.ops.shape(s)[-1]
        eye = keras.ops.eye(k, dtype=s.dtype)

        # Broadcasting: (B, K, 1) * (1, K, K) -> (B, K, K)
        s_diag = keras.ops.expand_dims(s_thresholded, axis=-1) * keras.ops.expand_dims(eye, axis=0)

        # Reconstruct: U @ S @ V^H
        # Since v is already V^H in Keras/TF, we use it directly
        low_rank = keras.ops.matmul(keras.ops.matmul(u, s_diag), v)

        return low_rank

    def _pcp_decomposition(
            self,
            attention_matrix: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Perform Principal Component Pursuit decomposition via alternating minimization.

        :param attention_matrix: Input attention matrix of shape
            ``(batch, num_heads, seq_len, seq_len)``.
        :type attention_matrix: keras.KerasTensor

        :return: Tuple of ``(L, S)`` where ``L`` is the low-rank component and
            ``S`` is the sparse component, both with the same shape as input.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Get shape info
        shape = keras.ops.shape(attention_matrix)
        seq_len = shape[2]

        # Reshape to (Batch * Heads, Seq, Seq) for vectorized SVD
        # We use -1 for the batch dimension to handle symbolic shapes
        flat_matrix = keras.ops.reshape(attention_matrix, (-1, seq_len, seq_len))

        # Initialize components
        L_flat = keras.ops.zeros_like(flat_matrix)
        S_flat = keras.ops.zeros_like(flat_matrix)

        # Alternating minimization
        for _ in range(self.max_pcp_iter):
            # Update L (low-rank component) via nuclear norm minimization
            # L = argmin ||L||_* s.t. A = L + S
            # Solution: L = SVT(A - S) where SVT is singular value thresholding
            residual = flat_matrix - S_flat

            # _nuclear_norm_minimization handles batched inputs correctly
            L_flat = self._nuclear_norm_minimization(residual, self.svd_threshold)

            # Update S (sparse component) via soft thresholding
            # S = argmin ||S||_1 s.t. A = L + S
            # Solution: S = soft_threshold(A - L, λ)
            S_flat = self._soft_threshold(flat_matrix - L_flat, self.lambda_sparse)

            # Note: Early stopping removed for Graph mode compatibility

        # Reshape back to (Batch, Num_Heads, Seq, Seq)
        L = keras.ops.reshape(L_flat, shape)
        S = keras.ops.reshape(S_flat, shape)

        return L, S

    def _compute_attention(
            self,
            q: keras.KerasTensor,
            k: keras.KerasTensor,
            v: keras.KerasTensor,
            mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """Compute robust attention using PCP decomposition.

        :param q: Query tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type q: keras.KerasTensor
        :param k: Key tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type k: keras.KerasTensor
        :param v: Value tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type v: keras.KerasTensor
        :param mask: Optional attention mask.
        :type mask: Optional[keras.KerasTensor]

        :return: Attention output of shape ``(batch, num_heads, seq_len, head_dim)``.
        :rtype: keras.KerasTensor
        """
        # Compute attention scores
        # Shape: (batch, num_heads, seq_len, seq_len)
        attention_scores = keras.ops.matmul(q, keras.ops.transpose(k, axes=[0, 1, 3, 2]))
        attention_scores = attention_scores * self.attention_scale

        # Dtype captured before any mask-driven promotion, restored at the cast-back
        # boundary below; that is what makes the no-mask path a bit-for-bit no-op.
        scores_dtype = getattr(attention_scores.dtype, "name", None) or str(attention_scores.dtype)

        if mask is not None:
            # Case 1: Mask is (batch, seq_len) -> Expand to (batch, 1, 1, seq_len)
            if len(mask.shape) == 2:
                mask = keras.ops.expand_dims(mask, axis=1)
                mask = keras.ops.expand_dims(mask, axis=1)
            # Case 2: Mask is (batch, seq_len, seq_len) -> Expand to (batch, 1, seq_len, seq_len)
            elif len(mask.shape) == 3:
                mask = keras.ops.expand_dims(mask, axis=1)

            # DECISION plan-2026-07-27T183600-b4ef45f0/D-009: rescue_axis stays at its
            # default (degenerate-row rescue on) — before the rescue, one blanked query row gave 64/4096 non-finite under mixed_float16. See decisions.md.
            #
            # DECISION plan-2026-07-27T183600-b4ef45f0/D-017: softmax axis is derived
            # from probability_config, not a bare -1 — at {"axis": -2} a dead key column gave 8192/8192 non-finite. See decisions.md.
            attention_scores = apply_attention_mask(
                attention_scores,
                keras.ops.not_equal(mask, 0),
                rescue_axis=(self.probability_config or {}).get("axis", -1),
            )

        L, S = self._pcp_decomposition(attention_scores)

        # DECISION plan-2026-07-27T183600-b4ef45f0/D-005: cast back to scores_dtype
        # only here, after the SVD — moving it above the PCP loop makes ops.svd die under mixed_float16. See decisions.md.
        robust_attention_scores = keras.ops.cast(L + S, scores_dtype)

        # Apply probability activation to get attention weights
        attention_weights = self.attn_prob(robust_attention_scores)

        # Apply attention weights to values
        # Shape: (batch, num_heads, seq_len, head_dim)
        attention_output = keras.ops.matmul(attention_weights, v)

        return attention_output

    def call(
            self,
            inputs: keras.KerasTensor,
            mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
            return_attention_scores: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Apply RPC attention to inputs.

        :param inputs: Input tensor of shape ``(batch_size, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :param mask: Optional attention mask of shape
            ``(batch_size, seq_len, seq_len)``.
        :type mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :param return_attention_scores: If ``True``, also returns attention weights.
        :type return_attention_scores: bool

        :return: Output tensor of shape ``(batch_size, seq_len, dim)``.
            If return_attention_scores is ``True``, returns
            ``(output, attention_weights)``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        """
        batch_size = keras.ops.shape(inputs)[0]
        seq_len = keras.ops.shape(inputs)[1]

        # Project to Q, K, V
        # Shape: (batch, seq_len, 3*dim)
        qkv = self.to_qkv(inputs)

        # Split into Q, K, V
        # Each has shape: (batch, seq_len, dim)
        q, k, v = keras.ops.split(qkv, 3, axis=-1)

        # Reshape to multi-head format
        # Shape: (batch, num_heads, seq_len, head_dim)
        # Shape: (B, N, dim) -> (B, N, H, head_dim) -> (B, H, N, head_dim)  [q, k, v]
        q = keras.ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        q = keras.ops.transpose(q, (0, 2, 1, 3))

        k = keras.ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = keras.ops.transpose(k, (0, 2, 1, 3))

        v = keras.ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = keras.ops.transpose(v, (0, 2, 1, 3))

        # Q/K normalization runs before the robust scoring loop.
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attention_output = self._compute_attention(q, k, v, mask)

        attention_weights = None
        if return_attention_scores:
            # Known defect: re-runs the whole PCP decomposition without applying mask,
            # so the returned weights need not match the weights that produced output.
            attention_scores = keras.ops.matmul(q, keras.ops.transpose(k, axes=[0, 1, 3, 2]))
            attention_scores = attention_scores * self.attention_scale
            L, S = self._pcp_decomposition(attention_scores)
            attention_weights = self.attn_prob(L + S)

        # Reshape back to (batch, seq_len, dim)
        # Shape: (B, H, N, head_dim) -> (B, N, H, head_dim) -> (B, N, dim)
        attention_output = keras.ops.transpose(attention_output, (0, 2, 1, 3))
        attention_output = keras.ops.reshape(attention_output, (batch_size, seq_len, self.dim))

        # Apply output projection
        # Shape: (B, N, dim) -> (B, N, dim)
        output = self.to_out(attention_output)

        # Apply dropout if specified
        if self.dropout is not None:
            output = self.dropout(output, training=training)

        if return_attention_scores:
            return output, attention_weights

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

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
            'lambda_sparse': self.lambda_sparse,
            'max_pcp_iter': self.max_pcp_iter,
            'svd_threshold': self.svd_threshold,
            'qkv_bias': self.qkv_bias,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'probability_type': self.probability_type,
            'probability_config': self.probability_config,
            'qk_norm_type': self.qk_norm_type,
            'qk_norm_kwargs': self.qk_norm_kwargs,
        })
        return config

# ---------------------------------------------------------------------
