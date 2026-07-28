"""Modern Hopfield Network Layer with Iterative Updates.

This layer implements a Modern Hopfield Network, as described in the paper
"Hopfield Networks is All You Need" [1]. It functions as a content-addressable
associative memory, capable of storing and retrieving a large number of patterns.

The core of the layer is the scaled dot-product attention mechanism from the
Transformer architecture, which serves as the update rule for the network's
state. Unlike a standard attention layer, which performs a single, feed-forward
computation, this layer can apply the attention mechanism iteratively. An
initial query (a "probe" or noisy pattern) is repeatedly refined until it
converges to one of the stored patterns (the "memories"), which are represented
by the Key-Value pairs.

Architecture:
    Projections and head-splitting are exactly a standard multi-head attention
    block; the difference is that the attention step is applied in a bounded
    loop, feeding its own output back in as the next query.

::

    ┌─────────────────────────────────────────────────────────────┐
    │      HopfieldAttention — bounded query-refinement loop      │
    │                                                             │
    │  [Q, K, V] ► Dense x3 ► heads ► optional Q/K norm           │
    │                     ▼                                       │
    │  ┌── repeat update_steps_max + 1 times ────────────────┐    │
    │  │   A   = attn_prob(Q · Kᵀ / sqrt(key_dim))           │    │
    │  │   out = A · V                                       │    │
    │  │   Q   = A · K      ◄── loop-back onto the query;    │    │
    │  │                        skipped on the final step    │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                     ▼                                       │
    │  merge heads ► output_dense ► Output (B, N_q, dim)          │
    │                                                             │
    │  update_steps_max=0 runs the body exactly once and never    │
    │  updates the query — ordinary single-step attention.        │
    │  The loop is a plain bounded Python range(): the count is   │
    │  fixed at trace time and there is no data-dependent exit.   │
    └─────────────────────────────────────────────────────────────┘

Difference from Standard Transformer Attention:
---------------------------------------------
The fundamental difference is the computational flow:

| Feature                | Standard Transformer Attention      | Modern Hopfield Network (This Layer)    |
|:-----------------------|:------------------------------------|:----------------------------------------|
| **Computational Flow** | Single-step, feed-forward           | Iterative, recurrent until convergence  |
| **Primary Goal**       | Contextual information weighting    | Associative pattern retrieval & cleaning|
| **Mechanism**          | `output = attention(Q, K, V)`       | `state_t+1 = attention(state_t, K, V)`  |
| **Query (Q)**          | Static input representation         | Dynamic state vector that evolves       |

In essence, a standard transformer attention layer performs a single update
step (`update_steps_max=0` in this implementation) of a modern Hopfield network.
This layer generalizes that concept by allowing for multiple (`update_steps_max > 0`)
iterative updates, enabling it to function as a true associative memory that
converges to stable fixed-points (attractors).

Iteration Count (there is no convergence test):
----------------------------------------------
The update process runs a **fixed** number of steps. `call()` executes a plain
bounded Python `for update_step in range(update_steps_max + 1)`, so:

1.  **The step count is a Python constant fixed at trace time.** The loop body
    runs `update_steps_max + 1` times on every forward pass — always, for every
    input. `update_steps_max=0` therefore means one ordinary attention step with
    the query never updated.

2.  **There is NO data-dependent early exit.** An earlier version stopped once
    the Frobenius norm of the change in the attention matrix fell below
    `update_steps_eps`; that test was removed because it required a Python
    `bool()` on a traced tensor, which raises under `@tf.function` / jit. See the
    anchored comment in `call()`. `update_steps_eps` is still accepted and
    serialized for API/checkpoint compatibility, but nothing on the forward path
    reads it.

Whether the state has actually settled into a fixed-point attractor is therefore
not observed by the layer; the caller chooses `update_steps_max` accordingly.

Foundational Mathematics:
    The update rule is the gradient-descent step of the modern Hopfield energy
    over continuous states::

        E(xi) = -lse(beta, X^T xi) + 0.5 * xi^T xi + const,   beta = 1/sqrt(d_k)

    whose stationary condition ``dE/dxi = 0`` rearranges into the fixed-point
    iteration this layer implements::

        xi_{t+1} = X softmax(beta * X^T xi_t)

    i.e. exactly ``attention(Q=xi_t, K=V=X)``. Ramsauer et al. show this converges
    in one step for well-separated patterns, which is why a single-step
    transformer attention layer is already a Hopfield retrieval; multiple steps
    matter only for metastable / clustered memories.

    Caveat specific to this implementation: the loop feeds back ``A K`` (keys),
    while the returned output is ``A V`` (values). The two coincide in the
    self-attention case the theory assumes; with distinct K and V the iteration is
    a heuristic generalization, not the energy descent above.

References:
    [1] Ramsauer, H., et al. (2020). "Hopfield Networks is All You Need".
        arXiv:2008.02217.
"""

# ---------------------------------------------------------------------

import math
import keras
from keras import ops
from typing import Optional, Tuple, Union, Any, List, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .common import apply_attention_mask
from ..activations import ProbabilityOutput
from ..norms.factory import create_normalization_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HopfieldAttention(keras.layers.Layer):
    """
    Modern Hopfield Network with iterative attention-based pattern retrieval.

    Implements a content-addressable associative memory using scaled dot-product
    attention as the update rule. When ``update_steps_max=0``, behaves as
    standard single-step attention. When ``update_steps_max > 0``, the query
    state is refined by ``xi_{t+1} = A_t K`` with
    ``A_t = attn_prob(xi_t K^T / sqrt(d_k))``, inside a bounded Python ``for``
    loop over ``update_steps_max + 1`` steps. That loop always runs to
    completion: there is no convergence test and no data-dependent early exit
    (see the ``update_steps_eps`` note below).

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────────────┐
        │                        HopfieldAttention                        │
        │                                                                 │
        │  Input(s) [B, S, D] — one tensor (Q=K=V) for self-attention,    │
        │  or a 2-/3-element list [query, key, value] for cross-attention │
        │                            ▼                                    │
        │  query_dense / key_dense / value_dense ► reshape ► transpose    │
        │  Q (B,H,N_q,key_dim)  K (B,H,N_k,key_dim)  V (B,H,N_k,val_dim)  │
        │                            ▼                                    │
        │  optional q_norm / k_norm  (applied ONCE, before the loop)      │
        │                            ▼                                    │
        │  ┌── for step in range(update_steps_max + 1) ───────────────┐   │
        │  │    S = Q · Kᵀ / sqrt(key_dim)      (a DIVIDE, not a      │   │
        │  │                                     reciprocal multiply) │   │
        │  │    S = apply_attention_mask(S, keep)  — only if a mask   │   │
        │  │        is given; a row that keeps NOTHING is rescued,    │   │
        │  │        on the axis taken from probability_config         │   │
        │  │    A = attn_prob(S)   (ProbabilityOutput: softmax by     │   │
        │  │                        default, but configurable)        │   │
        │  │    A = dropout(A)                                        │   │
        │  │    out = A · V                                           │   │
        │  │    Q = A · K     ◄── the loop-back that makes this a     │   │
        │  │                      Hopfield update; skipped on the     │   │
        │  │                      final step                          │   │
        │  └──────────────────────────────────────────────────────────┘   │
        │                            ▼                                    │
        │  transpose ► reshape to [B, N_q, H*value_dim]                   │
        │                            ▼                                    │
        │  output_dense ► Output [B, N_q, D]                              │
        │  (return_attention_scores=True also returns the LAST A)         │
        │                                                                 │
        │  The iteration count is FIXED at update_steps_max + 1 Python    │
        │  steps. There is no data-dependent early exit: the              │
        │  convergence test was removed because a Python bool() on a      │
        │  traced tensor breaks graph mode.  update_steps_max=0           │
        │  degenerates to ordinary one-shot attention.                    │
        └─────────────────────────────────────────────────────────────────┘

    :param num_heads: Number of attention heads. Must be positive.
    :type num_heads: int
    :param key_dim: Size of each attention head for key and query.
        Must be positive.
    :type key_dim: int
    :param value_dim: Optional size of each attention head for value.
        If ``None``, defaults to ``key_dim``.
    :type value_dim: int or None
    :param dropout_rate: Dropout probability for attention weights.
        Must be in ``[0, 1]``. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether to use bias in the attention projections.
        Defaults to ``True``.
    :type use_bias: bool
    :param kernel_initializer: Initializer for projection matrices.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias vectors.
        Defaults to ``'zeros'``.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for projection matrices.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param bias_regularizer: Optional regularizer for bias vectors.
    :type bias_regularizer: keras.regularizers.Regularizer or None
    :param activity_regularizer: Optional regularizer for layer output.
    :type activity_regularizer: keras.regularizers.Regularizer or None
    :param probability_type: Probability distribution type for attention
        weights. Forwarded to ``ProbabilityOutput``. Defaults to
        ``"softmax"``. Must not be one of ``"routing"``,
        ``"deterministic_routing"``, ``"hierarchical"``, or
        ``"hierarchical_routing"``.
    :type probability_type: str
    :param probability_config: Optional configuration dictionary forwarded
        to ``ProbabilityOutput`` as ``type_config``. Defaults to ``None``.
    :type probability_config: dict or None
    :param qk_norm_type: Optional normalization type applied to projected
        query and key patterns. Forwarded to ``create_normalization_layer``.
        Defaults to ``"layer_norm"``. Pass ``None`` to disable Q/K
        normalization.
    :type qk_norm_type: str or None
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        ``create_normalization_layer`` when building the Q/K norm layers.
        Defaults to ``None``.
    :type qk_norm_kwargs: dict or None
    :param update_steps_max: Maximum number of iterative Hopfield update
        steps. 0 means single-step (standard attention). Must be
        non-negative. Defaults to 0.
    :type update_steps_max: int
    :param update_steps_eps: Convergence threshold for the Frobenius norm
        of attention difference between steps. Must be positive.
        Defaults to ``1e-4``.
    :type update_steps_eps: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.
    :type kwargs: Any

    :raises ValueError: If ``num_heads <= 0`` or ``key_dim <= 0``.
    :raises ValueError: If ``dropout_rate`` is not in ``[0, 1]``.
    :raises ValueError: If ``update_steps_max < 0``.
    :raises ValueError: If ``update_steps_eps <= 0``.
    :raises ValueError: If ``probability_type`` names a routing/hierarchical
        variant.
    :raises ValueError: From ``call()``, if a list/tuple input has a length other
        than 2 or 3.

    .. note::
       ``update_steps_eps`` is retained for API and serialization compatibility
       but is **inert**: the convergence early-exit it controlled was removed as
       graph-unsafe (see the anchored comment in ``call()``). The loop always runs
       the full ``update_steps_max + 1`` steps.
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        update_steps_max: int = 0,
        update_steps_eps: float = 1e-4,
        probability_type: str = "softmax",
        probability_config: Optional[Dict[str, Any]] = None,
        qk_norm_type: Optional[str] = "layer_norm",
        qk_norm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if key_dim <= 0:
            raise ValueError(f"key_dim must be positive, got {key_dim}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        if update_steps_max < 0:
            raise ValueError(f"update_steps_max must be non-negative, got {update_steps_max}")
        if update_steps_eps <= 0:
            raise ValueError(f"update_steps_eps must be positive, got {update_steps_eps}")
        if probability_type in (
            "routing",
            "deterministic_routing",
            "hierarchical",
            "hierarchical_routing",
        ):
            raise ValueError(
                f"probability_type={probability_type!r} is not supported for "
                f"HopfieldAttention; routing/hierarchical variants require "
                f"context not available here."
            )

        # Store ALL configuration parameters
        self.num_heads = num_heads
        self.key_dim = key_dim
        # AF1: keep the RAW constructor arg so get_config() round-trips
        # value_dim=None as None (not the resolved key_dim int). self.value_dim
        # below stays the resolved value used for all internal compute.
        self._value_dim_arg = value_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        # AF2: precompute the static attention scale once. key_dim is a fixed
        # Python int, so math.sqrt(float(key_dim)) is the identical scalar value
        # that ops.sqrt(ops.cast(key_dim, ...)) produced in call() — folding it
        # into __init__ avoids a per-call op while staying byte-identical.
        #
        # R13: deliberately NOT `common.compute_attention_scale`. This layer
        # stores sqrt(key_dim) and DIVIDES by it in `_hopfield_update_step`; the
        # helper returns the reciprocal 1/sqrt(key_dim) for sites that MULTIPLY.
        # Adopting it would turn a divide into a multiply — a real numerics change
        # (measured: the two values differ for 26 of 27 probed dims, as they must,
        # being reciprocals). Same convention split as `wave_field_attention.py`.
        self._sqrt_key_dim = math.sqrt(float(self.key_dim))
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.update_steps_max = update_steps_max
        self.update_steps_eps = update_steps_eps
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        # CREATE all sub-layers in __init__ (they are unbuilt at this point)
        # Following Pattern 2: Composite Layer from the Modern Keras 3 guide

        # Projection layers - will be properly dimensioned in build()
        self.query_dense = keras.layers.Dense(
            self.num_heads * self.key_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="query_dense"
        )

        self.key_dense = keras.layers.Dense(
            self.num_heads * self.key_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="key_dense"
        )

        self.value_dense = keras.layers.Dense(
            self.num_heads * self.value_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="value_dense"
        )

        # Output projection - its output dimension depends on the input feature
        # dim (known only at build()). Created as a None sentinel here and
        # instantiated with the correct units in build(). Do NOT create it as
        # Dense(0) in __init__: a units=0 layer is a malformed placeholder that
        # must be discarded and rebuilt anyway. The None-sentinel + idempotency
        # guard in build() keeps first-build weights/forward exactly as before.
        self.output_dense = None

        # Dropout layer (conditional creation)
        if self.dropout_rate > 0.0:
            self.dropout_layer = keras.layers.Dropout(
                self.dropout_rate,
                name="attention_dropout"
            )
        else:
            self.dropout_layer = None

        # Q/K normalization (conditional creation via factory)
        if self.qk_norm_type is not None:
            self.q_norm = create_normalization_layer(
                self.qk_norm_type, name="q_norm", **(self.qk_norm_kwargs or {})
            )
            self.k_norm = create_normalization_layer(
                self.qk_norm_type, name="k_norm", **(self.qk_norm_kwargs or {})
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # Probability output for attention weights
        self.attn_prob = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="attn_prob",
        )

        logger.info(f"Initialized HopfieldAttention with {num_heads} heads, "
                   f"key_dim={key_dim}, value_dim={self.value_dim}")

    def build(self, input_shape: Union[Tuple, List]) -> None:
        """
        Build the layer and all its sub-layers.

        Explicitly builds each sub-layer for robust serialization following
        the Modern Keras 3 pattern.

        :param input_shape: Shape of input tensor or list of shapes for
            ``[query, key, value]``.
        :type input_shape: tuple or list
        """
        if self.built:
            return

        # Handle different input formats to extract query shape
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0:
            # Check if this is a list of shapes or a single shape
            if isinstance(input_shape[0], (list, tuple)):
                # This is a list of shapes [query_shape, key_shape, value_shape]
                query_shape = input_shape[0]
            else:
                # This is a single shape tuple (None, 32, 512)
                query_shape = input_shape
        else:
            # Single input shape provided
            query_shape = input_shape

        input_dim = query_shape[-1]
        logger.debug(f"Building HopfieldAttention with input_dim={input_dim}")

        # Create output_dense with the correct output dimension now that the
        # input feature dim is known. Idempotency-guarded so a re-build (via
        # from_config / functional reuse) does not clobber an already-created
        # sublayer (which would orphan restored weights).
        if self.output_dense is None:
            self.output_dense = keras.layers.Dense(
                input_dim,  # Output same dimension as input
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer,
                name="output_dense"
            )

        # Build sub-layers explicitly in computational order
        # This ensures all weight variables exist before weight restoration

        # DECISION plan_2026-06-14_077a2a35/D-001: build K/V Dense from actual
        # key/value shapes (cross-attn). Self-attn = flat tuple ->
        # isinstance(input_shape[0], (list,tuple)) False -> falls back to
        # query_shape (byte-identical). Do NOT revert to query_shape; breaks
        # cross-attention with differing K/V feature dim.
        key_shape = (
            input_shape[1]
            if isinstance(input_shape[0], (list, tuple)) and len(input_shape) > 1
            else query_shape
        )
        val_shape = (
            input_shape[2]
            if isinstance(input_shape[0], (list, tuple)) and len(input_shape) > 2
            else key_shape
        )

        # Build projection layers with input shape
        self.query_dense.build(query_shape)
        self.key_dense.build(key_shape)
        self.value_dense.build(val_shape)

        # Calculate intermediate shape for output projection
        projected_shape = list(query_shape)
        projected_shape[-1] = self.num_heads * self.value_dim
        self.output_dense.build(tuple(projected_shape))

        # Build conditional layers
        if self.dropout_layer is not None:
            # Dropout layer doesn't need explicit build as it doesn't have weights
            pass

        # Build Q/K norm layers (each receives (batch, num_heads, seq_len, key_dim))
        norm_shape = (None, None, None, self.key_dim)
        if self.q_norm is not None:
            self.q_norm.build(norm_shape)
        if self.k_norm is not None:
            self.k_norm.build(norm_shape)

        # Build attention probability layer with score shape
        # Scores have shape (batch, num_heads, seq_len_q, seq_len_k)
        score_shape = (None, self.num_heads, None, None)
        self.attn_prob.build(score_shape)

        logger.debug("HopfieldAttention build completed")

        # Always call parent build at the end
        super().build(input_shape)

    def _hopfield_update_step(
        self,
        query: keras.KerasTensor,
        key: keras.KerasTensor,
        value: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Perform one Hopfield update step using scaled dot-product attention.

        :param query: Query tensor of shape
            ``(batch, num_heads, seq_len_q, head_dim)``.
        :type query: keras.KerasTensor
        :param key: Key tensor of shape
            ``(batch, num_heads, seq_len_k, head_dim)``.
        :type key: keras.KerasTensor
        :param value: Value tensor of shape
            ``(batch, num_heads, seq_len_v, value_dim)``.
        :type value: keras.KerasTensor
        :param attention_mask: Optional attention mask.
        :type attention_mask: keras.KerasTensor or None
        :param training: Whether in training mode.
        :type training: bool or None
        :return: Tuple of ``(updated_output, attention_weights)``.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Scaled dot-product attention. AF2: precomputed static scalar (same
        # value as ops.sqrt(ops.cast(key_dim, ...))); a Python float divisor
        # broadcasts against the score tensor unchanged.
        scale = self._sqrt_key_dim
        attention_scores = ops.matmul(query, ops.transpose(key, [0, 1, 3, 2])) / scale

        # Handle mask processing
        actual_mask = None
        if isinstance(attention_mask, (list, tuple)):
            # If Keras passes a list, find the first non-None mask.
            for m in attention_mask:
                if m is not None:
                    actual_mask = m
                    break
        else:
            actual_mask = attention_mask

        if actual_mask is not None:
            mask_tensor = ops.cast(actual_mask, attention_scores.dtype)
            # Add heads dimension if missing for broadcasting.
            # attention_scores shape: (batch, num_heads, seq_len_q, seq_len_k)
            # A common mask shape is (batch, seq_len_q, seq_len_k).
            if len(ops.shape(mask_tensor)) == 3:
                mask_tensor = ops.expand_dims(mask_tensor, axis=1)

            # Add large negative values to masked positions.
            #
            # THIS SITE'S MASK POLARITY, passed through verbatim: `mask_tensor` is a
            # `1 = keep` predicate (already cast to the scores dtype above, on its own
            # untouched line), so it IS the keep predicate `apply_attention_mask`
            # wants. Do NOT "normalize" it into a `> 0` comparison or invert it — the
            # helper performs no polarity inference by design, so an inversion here
            # raises nothing, changes no shape and stays finite; the layer would just
            # attend to the padding. `TestHopfieldAttentionMaskPolarity` is the only
            # guard that can see it.
            #
            # DECISION plan-2026-07-27T183600-b4ef45f0/D-007
            # `out_dtype` is pinned to the SCORES' own dtype, so the biased scores stay
            # in the compute dtype (fp16 under `mixed_float16`), where
            # `MASK_BIAS_VALUE` is `-inf` again. That is deliberate and is NOT the bug
            # being fixed:
            #   * The bug was the ARITHMETIC form this line replaces,
            #     `attention_scores + (1.0 - mask_tensor) * -1e9`. In float16 `-1e9`
            #     is `-inf` (np.float16(-1e9) == -inf) and `(1.0 - mask) == 0` at every
            #     UNMASKED position, so the product is `0 * -inf = NaN` — the NaN
            #     appears where nothing was masked. THIS WAS THE WORST SITE IN THE
            #     PACKAGE: measured at (B=2, N=64, D=64, H=4, key_dim=16) under
            #     `mixed_float16`, an ALL-ONES mask — one that masks NOTHING — gave
            #     8192/8192 NaN, as did the padding and causal masks; float32 gave
            #     0/8192 in every case. `ops.where` inside `mask_dtype(...)` cannot
            #     form that product at all, so the failure mode is removed
            #     structurally rather than numerically.
            #   * Do NOT "improve" this to `out_dtype=None` (stay in float32) hoping to
            #     also rescue a FULLY-MASKED query row. It cannot: the next consumer is
            #     `self.attn_prob`, a Keras layer with autocasting ON, MEASURED to see a
            #     float32 input inside its own `call()` as float16 — the promotion is
            #     silently undone and all that remains is a wider, slower add. Pinned by
            #     `TestHopfieldAttentionMaskHazardIsReal::
            #     test_the_probability_sublayer_autocasts_a_float32_input`.
            #   * A FULLY-MASKED query row is a SEPARATE hazard that no `out_dtype` choice can
            #     touch. It is handled by the rescue below (D-009), not here.
            # See decisions.md D-007 (plan-2026-07-27T183600-b4ef45f0).
            #
            # DECISION plan-2026-07-27T183600-b4ef45f0/D-009
            # The fully-masked-row rescue IS applied here, and it supersedes the "not applied
            # here" note above: a query row that keeps NOTHING is treated as keeping EVERYTHING,
            # so the all-`-inf` row is never FORMED and no NaN gradient is created either. It
            # arrives via `apply_attention_mask`'s DEFAULT `rescue_axis=-1` — step 4c flipped
            # the step-4b opt-in default on the user's direction ("I care about correctness, not
            # backwards compatibility").
            #
            # DECISION plan-2026-07-27T183600-b4ef45f0/D-017
            # The axis is DERIVED from this layer's own `probability_config` rather than
            # left to the helper's `-1` default: `ProbabilityOutput` reads its softmax
            # `axis` from `type_config` (`activations/probability_output.py:180`) and this
            # layer forwards `probability_config` VERBATIM, so a caller can move the
            # reduction axis and the pre-step-10 "checked, not assumed" claim held only for
            # the DEFAULT config. MEASURED at the sibling `gated_attention` under
            # `mixed_float16` with `probability_config={"axis": -2}` and a dead KEY COLUMN:
            # 8192/8192 non-finite. WHAT NOT TO DO: do NOT restore a bare `-1` (correct only
            # while the caller leaves the config alone) and do NOT read this as the
            # rank/shape INFERENCE the D-009 anchor in `common.py` forbids — this reads the
            # site's own declared config. The full argument lives at the D-017 anchors in
            # `common.py` and `gated_attention.py`.
            # See decisions.md D-017 (plan-2026-07-27T183600-b4ef45f0).
            #
            # WHAT NOT TO DO: do NOT pass `rescue_axis=None` to "get the loud NaN back" — the
            # user ruled the finite-garbage semantics package-wide on 2026-07-28, and opting out
            # also restores the NaN GRADIENT on that row; do NOT move the rescue after the
            # softmax (`ops.where(row_keeps, w, 0)` still contributes `0 * NaN` in the backward
            # pass). The full argument lives at the D-009 / D-008 anchors in `common.py`.
            # See decisions.md D-009 and D-008 (plan-2026-07-27T183600-b4ef45f0).
            scores_dtype = keras.backend.standardize_dtype(attention_scores.dtype)
            attention_scores = apply_attention_mask(
                attention_scores,
                mask_tensor,
                out_dtype=scores_dtype,
                rescue_axis=(self.probability_config or {}).get("axis", -1),
            )

        attention_weights = self.attn_prob(attention_scores, training=training)

        # Apply dropout if configured
        if self.dropout_layer is not None:
            attention_weights = self.dropout_layer(attention_weights, training=training)

        output = ops.matmul(attention_weights, value)
        return output, attention_weights

    def call(
        self,
        inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        return_attention_scores: bool = False,
        training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Forward pass of the Hopfield attention layer.

        :param inputs: Input tensor or list of tensors
            ``[query, key, value]``. For self-attention, pass a single
            tensor. For cross-attention, pass ``[query, key, value]`` or
            ``[query, key_value]``.
        :type inputs: keras.KerasTensor or list[keras.KerasTensor]
        :param attention_mask: Optional attention mask tensor.
        :type attention_mask: keras.KerasTensor or None
        :param return_attention_scores: Whether to return attention scores
            along with output.
        :type return_attention_scores: bool
        :param training: Whether in training mode.
        :type training: bool or None
        :return: Output tensor of shape
            ``(batch_size, seq_len_query, input_dim)``, or tuple of
            ``(output, attention_weights)`` if
            ``return_attention_scores=True``.
        :rtype: keras.KerasTensor or tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Handle input formats
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 3:
                query, key, value = inputs
            elif len(inputs) == 2:
                query, key = inputs
                value = key
            else:
                raise ValueError(f"Expected 2 or 3 inputs, got {len(inputs)}")
        else:
            # Self-attention case
            query = key = value = inputs

        # Get shapes using Keras ops
        batch_size = ops.shape(query)[0]
        query_len = ops.shape(query)[1]
        key_len = ops.shape(key)[1]

        # Linear projections using built sub-layers
        # Shape: (B, N_q, D) -> (B, N_q, H*key_dim); key/value -> (B, N_k, H*key_dim)
        #        and (B, N_k, H*value_dim) respectively
        query_proj = self.query_dense(query)
        key_proj = self.key_dense(key)
        value_proj = self.value_dense(value)

        # Reshape to multi-head attention format
        # Shape: (B, N_q, H*key_dim) -> (B, N_q, H, key_dim)  [key/value analogous]
        query_proj = ops.reshape(
            query_proj,
            (batch_size, query_len, self.num_heads, self.key_dim)
        )
        key_proj = ops.reshape(
            key_proj,
            (batch_size, key_len, self.num_heads, self.key_dim)
        )
        value_proj = ops.reshape(
            value_proj,
            (batch_size, key_len, self.num_heads, self.value_dim)
        )

        # Transpose to (batch, num_heads, seq_len, dim)
        # Shape: (B, N, H, d) -> (B, H, N, d)  [all three]
        query_proj = ops.transpose(query_proj, [0, 2, 1, 3])
        key_proj = ops.transpose(key_proj, [0, 2, 1, 3])
        value_proj = ops.transpose(value_proj, [0, 2, 1, 3])

        # Apply Q/K normalization if enabled
        if self.q_norm is not None:
            query_proj = self.q_norm(query_proj, training=training)
        if self.k_norm is not None:
            key_proj = self.k_norm(key_proj, training=training)

        # Hopfield iterative updates.
        #
        # DECISION plan_2026-06-14_0c5d4a21/D-002: bounded Python loop (no Python
        # bool on a traced tensor) for graph-mode iterative Hopfield; early-exit
        # removed as optimization-only. The previous `while True` did
        # `if diff_norm < self.update_steps_eps:` where `diff_norm` is a traced
        # tensor (ops.sqrt result) -> a Python bool() on a symbolic tensor, which
        # raises under @tf.function / jit when update_steps_max > 0. Do NOT
        # reintroduce a data-dependent Python `if` on a traced tensor here (and
        # do NOT reach for keras.ops.cond — a plain bounded loop is simpler). The
        # convergence early-exit was an optimization, not a numeric contract:
        # running the full bounded `update_steps_max + 1` steps yields the same
        # final state when the network has not already converged. The
        # `update_steps_max == 0` path runs exactly one step (range(1)), query
        # never updated — numerically identical to before. See decisions.md D-002.
        current_query = query_proj
        attention_weights = None
        output = None
        for update_step in range(self.update_steps_max + 1):
            # Perform one Hopfield update step
            output, attention_weights = self._hopfield_update_step(
                current_query, key_proj, value_proj, attention_mask, training
            )

            # Update the query for the next iteration (skipped after the final
            # step). This implements the iterative Hopfield dynamics.
            if update_step < self.update_steps_max:
                # Shape: (B, H, N_q, N_k) @ (B, H, N_k, key_dim) -> (B, H, N_q, key_dim)
                # The updated query re-enters the loop in the SAME shape it left, which
                # is what makes the bounded iteration well-formed.
                current_query = ops.matmul(attention_weights, key_proj)

        # Reshape output back to original format
        # Shape: (B, H, N_q, value_dim) -> (B, N_q, H, value_dim)
        output = ops.transpose(output, [0, 2, 1, 3])  # (batch, seq_len, num_heads, value_dim)
        # Shape: (B, N_q, H, value_dim) -> (B, N_q, H*value_dim)
        output = ops.reshape(output, (batch_size, query_len, self.num_heads * self.value_dim))

        # Final output projection
        # Shape: (B, N_q, H*value_dim) -> (B, N_q, output_dim)
        output = self.output_dense(output)

        if return_attention_scores:
            return output, attention_weights
        return output

    def compute_output_shape(self, input_shape: Union[Tuple, List]) -> Tuple:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape of the input or list of input shapes.
        :type input_shape: tuple or list
        :return: Output shape tuple (same as query input shape).
        :rtype: tuple
        """
        # Handle different input formats
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0:
            # Check if this is a list of shapes or a single shape
            if isinstance(input_shape[0], (list, tuple)):
                # This is a list of shapes [query_shape, key_shape, value_shape]
                query_shape = input_shape[0]
            else:
                # This is a single shape tuple (None, 32, 512)
                query_shape = input_shape
        else:
            # Single input shape provided
            query_shape = input_shape

        # Output has same shape as query input
        return tuple(query_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            # AF1: serialize the RAW arg so value_dim=None round-trips as None.
            "value_dim": self._value_dim_arg,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
            "update_steps_max": self.update_steps_max,
            "update_steps_eps": self.update_steps_eps,
            "probability_type": self.probability_type,
            "probability_config": self.probability_config,
            "qk_norm_type": self.qk_norm_type,
            "qk_norm_kwargs": self.qk_norm_kwargs,
        })
        return config

# ---------------------------------------------------------------------
