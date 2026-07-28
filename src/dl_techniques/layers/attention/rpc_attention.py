"""
A robust attention mechanism via Principal Component Pursuit.

This layer enhances standard scaled dot-product attention by integrating
Principal Component Pursuit (PCP), a matrix decomposition technique that
separates the attention matrix into a low-rank component ``L`` (global
patterns) and a sparse component ``S`` (localized details/outliers) by
solving ``min_{L,S} ||L||_* + lambda ||S||_1  s.t.  A = L + S``. The
robust attention is then ``softmax(L + S) V``, providing resilience
against noise and adversarial perturbations.

Architecture:
    A single ``Dense(3 * dim)`` produces Q, K and V, reshaped to
    ``(batch, num_heads, seq_len, head_dim)``. The scaled score matrix is then run
    through ``max_pcp_iter`` fixed sweeps of alternating minimization — a singular
    value thresholding step for ``L`` and a soft-thresholding step for ``S`` — and
    the recombined ``L + S`` is normalized by the shared ``ProbabilityOutput``
    layer before being applied to ``V``.

    The iteration count is a Python constant, not a convergence test: early
    stopping was deliberately removed so the loop unrolls identically under graph
    tracing. Cost is therefore ``max_pcp_iter`` batched SVDs of an ``S x S``
    matrix per forward pass, which makes this layer far more expensive than plain
    scaled dot-product attention and unsuitable for long sequences.

    ``call()``'s mask parameter is spelled ``mask=``, not ``attention_mask=``; see
    the ``[FROZEN SIGNATURE]`` note on the class below.

Foundational Mathematics:
    Principal Component Pursuit recovers a low-rank ``L`` and a sparse ``S`` from
    their sum by solving the convex relaxation::

        min_{L,S}  ||L||_*  +  lambda ||S||_1     s.t.  A = L + S

    where ``||.||_*`` is the nuclear norm (sum of singular values). The two
    proximal operators used by the alternating sweeps are::

        L = SVT_tau(A - S) = U diag(max(sigma - tau, 0)) V^H
        S = soft_lambda(A - L) = sign(A - L) max(|A - L| - lambda, 0)

    Applied to attention logits, ``L`` captures the global, low-rank co-occurrence
    structure and ``S`` absorbs localized outliers, so an adversarial perturbation
    concentrated on a few token pairs is routed into ``S`` instead of distorting
    the whole softmax.

References:
    - Candes, E. J., Li, X., Ma, Y., & Wright, J. (2011). "Robust
      Principal Component Analysis?". Journal of the ACM.
"""

# ---------------------------------------------------------------------

import keras
from typing import Optional, Union, Tuple, Any, Dict
from keras import ops, layers, initializers, regularizers

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from .common import (
    apply_attention_mask,
    compute_attention_scale,
    validate_head_divisibility,
)
from ..activations import ProbabilityOutput
from ..norms.factory import create_normalization_layer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
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

    **[FROZEN SIGNATURE — D-007 carve-out]** ``call()``'s mask parameter is named
    ``mask``, not ``attention_mask`` as in most siblings. This inconsistency is
    intentional and is recorded by the standing anchor
    ``plan_2026-06-14_0c5d4a21/D-007`` at ``factory.py:939-944``, which places this
    spelling (and ``performer_attention.PerformerAttention.call``'s complete
    absence of a mask argument) out of scope for normalization passes. **Do NOT
    rename it**: the factory and existing callers pass it by keyword.

    The public attribute ``self.attention_scale`` is likewise frozen. It is the
    only occurrence of that name in the package — every sibling calls the same
    quantity ``self.scale`` — but it is reachable from user code and from
    subclasses, so renaming it is an API change, not a style fix.

    **[REUSE]** The ``dim % num_heads`` check, the ``1 / sqrt(head_dim)``
    temperature and the additive mask bias
    (:func:`~dl_techniques.layers.attention.common.apply_attention_mask`, which
    keeps the masked scores in ``>= float32`` so they can reach ``ops.svd``) all
    come from :mod:`~dl_techniques.layers.attention.common`; score
    normalization comes from the shared
    :class:`~dl_techniques.layers.activations.ProbabilityOutput`; the optional Q/K
    norms come from
    :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`.

    .. warning::
        **This layer cannot run under a ``mixed_float16`` policy on this backend,
        and that is a missing backend kernel rather than a defect in this file.**
        MEASURED on TF 2.18 / CUDA / RTX 4070 (plan-2026-07-27T183600-b4ef45f0,
        steps 2 and 5b): ``ops.svd`` has NO float16 kernel outside XLA, so the
        first ``ops.svd`` inside :meth:`_pcp_decomposition` raises::

            NotFoundError: Could not find device for node:
            {{node Svd}} = Svd[T=DT_HALF, ...]
            All kernels registered for op Svd:
              device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_HALF]
              device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_HALF]
              device='CPU';         T in [DT_FLOAT] / [DT_DOUBLE] / complex
              device='GPU';         T in [DT_DOUBLE] / [DT_FLOAT]

        Specifics, because they are easy to get wrong:

        * It raises **with or without an ``attention_mask``** — this is not a
          masking bug. A MASKED fp16 forward happens to survive, because the mask
          bias is applied in ``mask_dtype(...)`` (see the D-005 boundary below) and
          the promoted scores then have a float32 SVD kernel. The **unmasked** fp16
          forward has nothing to promote it and dies.
        * XLA/``jit_compile=True`` is the ONLY path on which an fp16 ``Svd`` kernel
          exists at all (``XLA_*_JIT`` above).
        * Promoting the UNMASKED path too would change no-mask numerics for every
          existing caller, which is why it was not done here.

        ``float32`` and ``float64`` policies are fully supported. Carried to the
        Tier-4 brief (``research/2026_attention_open_design_questions.md``).

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────┐
        │     RPCAttention — Robust PCA in place of a softmax     │
        │                                                         │
        │   The score matrix is split into a LOW-RANK L plus a    │
        │   SPARSE S by iterative Principal Component Pursuit     │
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
        │   KNOWN DEFECT: return_attention_scores=True re-runs the│
        │   WHOLE PCP on scores that were never masked, so the    │
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
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            probability_type: str = "softmax",
            probability_config: Optional[Dict[str, Any]] = None,
            qk_norm_type: Optional[str] = None,
            qk_norm_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        # R13: adopts the shared validator. `test_rpc_attention.py:82` pins the FULL
        # message with a regex (`dim \(63\) must be divisible by num_heads \(8\)`),
        # and the helper's default `*_name` kwargs reproduce that text
        # character-for-character, so the pinned regex still matches.
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
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        # Normalize regularizers via regularizers.get() so str/dict/object/None
        # all round-trip uniformly through regularizers.serialize() in get_config.
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        # Computed attributes
        self.head_dim = dim // num_heads
        # R13: was `1.0 / np.sqrt(self.head_dim)`. Adoption was gated on an explicit
        # equality probe, not on the two expressions looking alike:
        # `float(1.0/np.sqrt(d)).hex()` matched `compute_attention_scale(d).hex()`
        # for 27 realistic head dims (1..512). (Step 2 of this plan proved the
        # `head_dim ** -0.5` form is NOT bit-identical, so the probe is load-bearing.)
        # The only change is the Python-level type, `np.float64` -> `float`;
        # `np.float64` is a `float` subclass, this value is not a `get_config()` key,
        # and `keras.ops` converts either to the tensor dtype identically.
        #
        # ATTRIBUTE NAME IS FROZEN: it stays `attention_scale`, not `scale`. It is
        # the only such spelling in the package but it is public-ish surface (see
        # the frozen-signature note in the class docstring). The D-002 rule still
        # holds: a Python float computed in `__init__`, never `ops.sqrt` in `call()`.
        self.attention_scale = compute_attention_scale(self.head_dim)

        # Create sub-layers in __init__
        # Q, K, V projection layer
        self.to_qkv = layers.Dense(
            3 * dim,
            use_bias=qkv_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='qkv_projection'
        )

        # Output projection layer
        self.to_out = layers.Dense(
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
            self.dropout = layers.Dropout(dropout_rate)
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
        return ops.sign(x) * ops.maximum(ops.abs(x) - threshold, 0.0)

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
        u, s, v = ops.svd(matrix, full_matrices=False)

        # Apply soft thresholding to singular values
        s_thresholded = ops.maximum(s - threshold, 0.0)

        # Reconstruct low-rank matrix
        # Need to construct a batch of diagonal matrices from s_thresholded
        # s_thresholded is (B, K)
        # We need s_diag to be (B, K, K)

        k = ops.shape(s)[-1]
        eye = ops.eye(k, dtype=s.dtype)

        # Broadcasting: (B, K, 1) * (1, K, K) -> (B, K, K)
        s_diag = ops.expand_dims(s_thresholded, axis=-1) * ops.expand_dims(eye, axis=0)

        # Reconstruct: U @ S @ V^H
        # Since v is already V^H in Keras/TF, we use it directly
        low_rank = ops.matmul(ops.matmul(u, s_diag), v)

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
        shape = ops.shape(attention_matrix)
        seq_len = shape[2]

        # Reshape to (Batch * Heads, Seq, Seq) for vectorized SVD
        # We use -1 for the batch dimension to handle symbolic shapes
        flat_matrix = ops.reshape(attention_matrix, (-1, seq_len, seq_len))

        # Initialize components
        L_flat = ops.zeros_like(flat_matrix)
        S_flat = ops.zeros_like(flat_matrix)

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
        L = ops.reshape(L_flat, shape)
        S = ops.reshape(S_flat, shape)

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
        attention_scores = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2]))
        attention_scores = attention_scores * self.attention_scale

        # The dtype the scores arrive in, captured BEFORE any mask-driven promotion.
        # This is what the cast-back boundary below restores, which is what makes the
        # no-mask path a bit-for-bit no-op (`ops.cast` to the dtype a tensor already
        # has returns the tensor itself).
        scores_dtype = keras.backend.standardize_dtype(attention_scores.dtype)

        # Apply mask if provided
        if mask is not None:
            # Broadcast mask to (batch, num_heads, seq_len, seq_len). The dispatch
            # below reads the STATIC rank (`len(mask.shape)`); a dynamic `ops.shape`
            # call here was left dead by the step-2 rewrite and removed at step 10.

            # Case 1: Mask is (batch, seq_len) -> Expand to (batch, 1, 1, seq_len)
            if len(mask.shape) == 2:
                mask = ops.expand_dims(mask, axis=1)
                mask = ops.expand_dims(mask, axis=1)
            # Case 2: Mask is (batch, seq_len, seq_len) -> Expand to (batch, 1, seq_len, seq_len)
            elif len(mask.shape) == 3:
                mask = ops.expand_dims(mask, axis=1)

            # The keep predicate is spelled `mask != 0` because THIS site spells
            # masking `mask == 0` (every multiply-form sibling instead treats a
            # `1 = keep` float mask). `apply_attention_mask` performs no polarity
            # inference by design, so the polarity lives here, in one visible line.
            # Inverting it would raise nothing, change no shape and stay finite —
            # the layer would simply attend to the padding. `test_rpc_attention.py::
            # TestRPCAttentionMaskPolarity` is the only guard that can see that.
            #
            # `out_dtype` is deliberately LEFT AT ITS DEFAULT, so the biased scores
            # stay in `mask_dtype(...)` (>= float32) all the way into the SVD below.
            # See the D-005 anchor at the cast-back boundary for why that is the
            # whole point of this site's fix.
            #
            # DECISION plan-2026-07-27T183600-b4ef45f0/D-009
            # `rescue_axis` is ALSO left at its default, which since step 4c means the
            # degenerate-row rescue is ON here: a query row that keeps NOTHING is
            # treated as keeping EVERYTHING. Step 4b had deliberately excluded this
            # site to keep it byte-identical; the user removed that hedge on
            # 2026-07-28 ("I care about correctness, not backwards compatibility"),
            # and one uniform semantics across the package is the point.
            #
            # This is NOT cosmetic here, contrary to the "rpc runs in float32 so it
            # was already finite" reading. MEASURED at step 4c with a rank-3 mask that
            # blanks exactly one query row: under `mixed_float16` the pre-4c forward
            # returned 64/4096 NON-FINITE outputs — the all-`-1e9` row survives the
            # SVD but the cast-back boundary below turns it into all-`-inf`, and
            # `self.attn_prob` softmaxes that to NaN. In float32 and float64 it was
            # finite but WRONG-BY-CONVENTION: because `_pcp_decomposition` is a GLOBAL
            # factorization, one all-`-1e9` row shifted the ENTIRE output (measured
            # max deviation 0.710 in float32, 0.0293 in float64 against an all-ones
            # mask that the rescue makes it equivalent to).
            #
            # WHAT NOT TO DO: do NOT pass `rescue_axis=None` here to restore the old
            # numbers. Guarded by
            # `test_rpc_attention.py::TestRPCAttentionFullyMaskedRow`.
            # See decisions.md D-009 (plan-2026-07-27T183600-b4ef45f0).
            #
            # DECISION plan-2026-07-27T183600-b4ef45f0/D-017
            # The axis is DERIVED from this layer's own `probability_config` rather than
            # left to the helper's `-1` default: `ProbabilityOutput` reads its softmax
            # `axis` from `type_config` (`activations/probability_output.py:180`) and
            # this layer forwards `probability_config` VERBATIM, so a caller can move
            # the reduction axis and the pre-step-10 "`-1` is correct because
            # `self.attn_prob` reduces over the KEY axis" claim held only for the
            # DEFAULT config. MEASURED at the sibling `gated_attention` under
            # `mixed_float16` with `probability_config={"axis": -2}` and a dead KEY
            # COLUMN: 8192/8192 non-finite. WHAT NOT TO DO: do NOT restore a bare `-1`,
            # and do NOT read this as the rank/shape INFERENCE the D-009 anchor in
            # `common.py` forbids — this reads the site's own declared config.
            # See decisions.md D-017 (plan-2026-07-27T183600-b4ef45f0).
            attention_scores = apply_attention_mask(
                attention_scores,
                ops.not_equal(mask, 0),
                rescue_axis=(self.probability_config or {}).get("axis", -1),
            )

        # Perform PCP decomposition
        L, S = self._pcp_decomposition(attention_scores)

        # DECISION plan-2026-07-27T183600-b4ef45f0/D-005
        # THE cast-back boundary. The masked score matrix is promoted to float32 by
        # `apply_attention_mask` above and stays there through `_pcp_decomposition`;
        # this single line brings it back to the compute dtype, AFTER the SVD and
        # BEFORE the probability activation.
        #
        # WHAT NOT TO DO, and why:
        #   * Do NOT move this cast ABOVE `_pcp_decomposition` (i.e. do not pass
        #     `out_dtype=scores_dtype` to `apply_attention_mask`) to "keep the loop in
        #     the compute dtype like before". Under `mixed_float16` that re-creates
        #     `-inf` masked entries, and MEASURED on TF 2.18 + CUDA, `ops.svd` then
        #     fails outright: `Could not find device for node: Svd[T=DT_HALF]` — the
        #     op has NO float16 kernel (registered: CPU float/double/complex, GPU
        #     float/double). Even where a half kernel exists (XLA), a single
        #     non-finite entry NaN-poisons the entire decomposition. The promotion is
        #     not a precision nicety; it is what makes a masked forward pass exist at
        #     all under mixed precision.
        #   * Do NOT "simplify" this away as redundant with Keras autocasting. It is
        #     redundant only for the value it produces, not for the invariant it
        #     states: the boundary is named here so the next reader sees exactly where
        #     the float32 region ends, instead of discovering it inside
        #     `ProbabilityOutput.__call__`.
        #   * Do NOT replace `scores_dtype` with `self.compute_dtype`. `scores_dtype`
        #     is captured from the tensor itself before masking, which is what makes
        #     the NO-MASK path provably unchanged: `ops.cast` to a tensor's own dtype
        #     returns that tensor, so an unmasked forward traces the same graph and
        #     produces bit-identical output to the pre-fix implementation (verified).
        #
        # The FULLY-masked query row used to be a known residual here — it became
        # all-`-inf` at exactly this cast under fp16 and its softmax was NaN. That is
        # FIXED as of step 4c, not by this boundary but by the degenerate-row rescue
        # now defaulted on in `apply_attention_mask` above (D-009): the all-`-1e9` row
        # is never formed, so there is nothing for this cast to overflow. Still
        # unchanged: an fp16 forward with NO mask hits the missing `Svd[T=DT_HALF]`
        # kernel, because nothing promotes it.
        # See decisions.md D-005 (plan-2026-07-27T183600-b4ef45f0).
        robust_attention_scores = ops.cast(L + S, scores_dtype)

        # Apply probability activation to get attention weights
        attention_weights = self.attn_prob(robust_attention_scores)

        # Apply attention weights to values
        # Shape: (batch, num_heads, seq_len, head_dim)
        attention_output = ops.matmul(attention_weights, v)

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
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Project to Q, K, V
        # Shape: (batch, seq_len, 3*dim)
        qkv = self.to_qkv(inputs)

        # Split into Q, K, V
        # Each has shape: (batch, seq_len, dim)
        q, k, v = ops.split(qkv, 3, axis=-1)

        # Reshape to multi-head format
        # Shape: (batch, num_heads, seq_len, head_dim)
        # Shape: (B, N, dim) -> (B, N, H, head_dim) -> (B, H, N, head_dim)  [q, k, v]
        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))

        k = ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.transpose(k, (0, 2, 1, 3))

        v = ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Optional Q/K normalization (applied BEFORE the robust scoring loop).
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Compute robust attention with PCP decomposition
        # Shape: 3x (B, H, N, head_dim) -> (B, H, N, head_dim)
        attention_output = self._compute_attention(q, k, v, mask)

        # For returning attention scores if requested
        attention_weights = None
        if return_attention_scores:
            # KNOWN DEFECT (pre-existing, deliberately NOT fixed here — the fix is a
            # numerics change AND a restructuring of `_compute_attention`): this
            # branch re-runs the ENTIRE PCP decomposition, i.e. `max_pcp_iter` more
            # batched SVDs, roughly doubling the cost of the forward pass. Worse, it
            # recomputes the scores WITHOUT applying `mask`, so the weights returned
            # to the caller do not correspond to the weights that produced `output`
            # whenever a mask is supplied. Reported, not fixed, in this plan.
            # Recompute attention scores for output
            # Shape: (B, H, N, head_dim) @ (B, H, head_dim, N) -> (B, H, N, N)
            attention_scores = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2]))
            attention_scores = attention_scores * self.attention_scale
            L, S = self._pcp_decomposition(attention_scores)
            attention_weights = self.attn_prob(L + S)

        # Reshape back to (batch, seq_len, dim)
        # Shape: (B, H, N, head_dim) -> (B, N, H, head_dim) -> (B, N, dim)
        attention_output = ops.transpose(attention_output, (0, 2, 1, 3))
        attention_output = ops.reshape(attention_output, (batch_size, seq_len, self.dim))

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
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'probability_type': self.probability_type,
            'probability_config': self.probability_config,
            'qk_norm_type': self.qk_norm_type,
            'qk_norm_kwargs': self.qk_norm_kwargs,
        })
        return config

# ---------------------------------------------------------------------
