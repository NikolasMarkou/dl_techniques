"""Capsule layers V2 — single-step attention routing with decoupled length & probability.

This module provides a modernised capsule routing primitive that addresses
several documented limitations of the iterative dynamic-routing scheme used in
`dl_techniques.layers.capsules.RoutingCapsule`:

* **No iterative inner loop.** Routing is computed in a single forward pass via
  attention-style scoring. This makes the layer fully parallel, XLA-fusable, and
  removes the sequential ``b_{i+1} = b_i + agreement(v_i)`` data dependency that
  blocks GPU/TPU utilization in the legacy implementation.
* **Decoupled length and probability.** The capsule output is constructed as
  ``v = sigmoid(prob_head(s)) * (s / ||s||)`` — magnitude is a learned scalar
  per capsule (the detection probability under margin loss), and direction is a
  unit vector encoding the pose. This eliminates the squash function's
  saturation at zero and the conflation of "vector magnitude" with "detection
  probability".
* **Optional Top-K sparsity.** Each routing softmax can be restricted to the
  top-k input capsules, keeping per-output cost sub-quadratic when scaling to
  many input capsules (a la mixture-of-experts gating).
* **Optional auxiliary load-balancing loss.** When enabled, an importance loss
  on the routing assignments is added via ``self.add_loss`` during training to
  discourage coupling collapse (a single output capsule monopolizing the
  routing).

The companion :class:`CapsuleBlockV2` wraps :class:`AttentionRoutingCapsule`
with optional dropout and a length-preserving direction-only normalizer.

References
----------
* Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between
  capsules. Advances in NeurIPS 30.
* Hahn, T., Pyeon, M., & Kim, G. (2019). Self-Routing Capsule Networks.
* Tsai, Y.-H. H., et al. (2020). Capsules with Inverted Dot-Product Attention
  Routing.
* Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts Layer (load-balancing loss).
"""

import keras
from typing import Optional, Union, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.activations.squash import SquashLayer  # noqa: F401  (kept for API parity)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class AttentionRoutingCapsule(keras.layers.Layer):
    """Single-step attention-routing capsule layer.

    A drop-in replacement for the iterative :class:`RoutingCapsule` that uses
    a learned per-output query to score the prediction vectors ``u_hat`` and
    a single softmax to compute the routing weights, then aggregates with a
    decoupled magnitude/direction output.

    Forward pass:

    .. code-block:: text

        Input u: (B, N_in, D_in)
                       │
                       ▼
              u_hat = W @ u_i        # (B, N_in, N_out, D_out)
                       │
                       ▼
              score = (u_hat · q_j) / sqrt(D_out)   # (B, N_in, N_out)
                       │
                       ▼  (optional Top-K mask)
              a = softmax(score, axis=output|input)
                       │
                       ▼
              s = sum_i (a * u_hat) (+ bias)        # (B, N_out, D_out)
                       │
                       ▼  decoupled output
              mag       = sigmoid(prob_head(s))     # (B, N_out, 1)
              direction = s / (||s|| + eps)         # (B, N_out, D_out)
              v         = mag * direction           # (B, N_out, D_out)

    Args:
        num_capsules: Number of output capsules. Must be positive.
        dim_capsules: Dimension of each output capsule vector. Must be
            positive.
        softmax_axis: Either ``"output"`` (each input capsule competes for
            output assignment — matches dynamic-routing semantics) or
            ``"input"`` (each output capsule receives a normalised mixture
            of inputs — like classic attention). Defaults to ``"output"``.
        top_k: If set, restrict each softmax row to the ``top_k`` largest
            scores along the soft-maxed axis (other entries are masked
            with -inf before softmax). ``None`` (default) disables masking.
        use_bias: Whether to add a learned bias to the routing aggregate
            ``s`` before computing the output. Defaults to ``True``.
        use_load_balancing: Whether to attach an auxiliary importance loss
            on the routing assignments via ``self.add_loss`` (only during
            training). Defaults to ``False``.
        load_balancing_weight: Scalar weight on the auxiliary loss when
            ``use_load_balancing=True``. Defaults to ``0.01``.
        eps: Numerical-stability constant for the unit-direction division.
            Defaults to ``1e-7``.
        kernel_initializer: Initializer for transformation matrices ``W``
            and the per-output query ``q``. Defaults to ``'glorot_uniform'``.
        kernel_regularizer: Optional regularizer for transformation matrices.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        num_capsules: int,
        dim_capsules: int,
        softmax_axis: Literal["output", "input"] = "output",
        top_k: Optional[int] = None,
        use_bias: bool = True,
        use_load_balancing: bool = False,
        load_balancing_weight: float = 0.01,
        eps: float = 1e-7,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if num_capsules <= 0:
            raise ValueError(f"num_capsules must be positive, got {num_capsules}")
        if dim_capsules <= 0:
            raise ValueError(f"dim_capsules must be positive, got {dim_capsules}")
        if softmax_axis not in ("output", "input"):
            raise ValueError(
                f"softmax_axis must be 'output' or 'input', got {softmax_axis!r}"
            )
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive or None, got {top_k}")
        if not (0.0 <= load_balancing_weight):
            raise ValueError(
                f"load_balancing_weight must be non-negative, got {load_balancing_weight}"
            )

        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.softmax_axis = softmax_axis
        self.top_k = top_k
        self.use_bias = use_bias
        self.use_load_balancing = use_load_balancing
        self.load_balancing_weight = float(load_balancing_weight)
        self.eps = float(eps)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        # Per-capsule scalar probability head:
        # maps a D_out-dim capsule vector to a scalar probability via Dense(1).
        self.prob_head = keras.layers.Dense(
            units=1,
            activation=None,  # raw logit; sigmoid applied in call()
            use_bias=True,
            kernel_initializer="glorot_uniform",
            name="prob_head",
        )

        # Set in build()
        self.W: Optional[keras.Variable] = None
        self.q: Optional[keras.Variable] = None
        self.bias: Optional[keras.Variable] = None
        self.num_input_capsules: Optional[int] = None
        self.input_dim_capsules: Optional[int] = None

    # ------------------------------------------------------------------
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Expected 3D input shape [batch, num_input_capsules, "
                f"input_dim_capsules], got {input_shape}"
            )

        self.num_input_capsules = input_shape[1]
        self.input_dim_capsules = input_shape[2]

        if self.num_input_capsules is None:
            raise ValueError(
                "AttentionRoutingCapsule requires a known num_input_capsules "
                "dimension at build time."
            )

        # Transformation weights W: shape (N_in, N_out, D_out, D_in).
        # Used via einsum for robust graph-mode behavior (the legacy
        # matmul+squeeze pattern can lose static shape info under
        # tf.function tracing).
        self.W = self.add_weight(
            shape=(
                self.num_input_capsules,
                self.num_capsules,
                self.dim_capsules,
                self.input_dim_capsules,
            ),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="capsule_transformation_weights",
        )

        # Learned query per output capsule.
        # Shape chosen so the dot product u_hat · q is broadcastable.
        self.q = self.add_weight(
            shape=(1, 1, self.num_capsules, self.dim_capsules),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="capsule_routing_query",
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(1, self.num_capsules, self.dim_capsules),
                initializer="zeros",
                trainable=True,
                name="capsule_bias",
            )

        self.prob_head.build((None, self.num_capsules, self.dim_capsules))

        super().build(input_shape)
        logger.info(
            f"Built AttentionRoutingCapsule: {self.num_input_capsules} -> "
            f"{self.num_capsules} capsules, softmax_axis={self.softmax_axis}, "
            f"top_k={self.top_k}, use_load_balancing={self.use_load_balancing}"
        )

    # ------------------------------------------------------------------
    def _apply_top_k_mask(
        self,
        score: keras.KerasTensor,
        axis: int,
    ) -> keras.KerasTensor:
        """Mask all but the top-k entries of ``score`` along ``axis`` to -inf."""
        k = int(self.top_k)
        # Clamp to the axis size — the static shape is known after build.
        if axis == 1:
            axis_size = self.num_input_capsules
        else:
            axis_size = self.num_capsules
        if axis_size is not None:
            k = min(k, axis_size)

        if axis == 1:
            # Move axis 1 to last for keras.ops.top_k.
            score_t = keras.ops.transpose(score, (0, 2, 1))  # (B, N_out, N_in)
        else:
            score_t = score  # (B, N_in, N_out)
        # keras.ops.top_k operates on the last axis.
        topk_values, _ = keras.ops.top_k(score_t, k=k)
        # The k-th value (smallest of the top-k) is the threshold per row.
        threshold = topk_values[..., -1:]
        keep = score_t >= threshold
        if axis == 1:
            keep = keras.ops.transpose(keep, (0, 2, 1))
        # Use a large negative number so post-softmax weight ≈ 0.
        neg_inf = keras.ops.cast(-1e9, score.dtype)
        masked = keras.ops.where(keep, score, neg_inf)
        return masked

    # ------------------------------------------------------------------
    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        # inputs: (B, N_in, D_in)
        # u_hat[b, i, o, d] = sum_e W[i, o, d, e] * inputs[b, i, e]
        # einsum keeps static shapes intact under tf.function tracing.
        u_hat = keras.ops.einsum("iode,bie->biod", self.W, inputs)  # (B, N_in, N_out, D_out)

        # Score: (u_hat · q) / sqrt(D_out)
        # u_hat: (B, N_in, N_out, D_out); q: (1, 1, N_out, D_out)
        score = keras.ops.sum(u_hat * self.q, axis=-1)  # (B, N_in, N_out)
        score = score / float(self.dim_capsules) ** 0.5

        # Optional Top-K masking before softmax.
        if self.top_k is not None:
            if self.softmax_axis == "output":
                score = self._apply_top_k_mask(score, axis=2)
            else:
                score = self._apply_top_k_mask(score, axis=1)

        # Softmax along the chosen axis.
        if self.softmax_axis == "output":
            a = keras.activations.softmax(score, axis=2)  # competition over outputs
        else:
            a = keras.activations.softmax(score, axis=1)  # mixture over inputs

        # Aggregate: s_j = sum_i (a_ij * u_hat_ij)
        # a: (B, N_in, N_out) -> expand to (B, N_in, N_out, 1) for broadcast.
        a_exp = keras.ops.expand_dims(a, axis=-1)
        s = keras.ops.sum(a_exp * u_hat, axis=1)  # (B, N_out, D_out)

        if self.use_bias and self.bias is not None:
            s = s + self.bias

        # Decoupled output: magnitude (sigmoid head) * unit direction.
        # Direction.
        s_norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(s), axis=-1, keepdims=True) + self.eps)
        direction = s / s_norm  # (B, N_out, D_out)

        # Magnitude: sigmoid(prob_head(s)) — scalar per capsule.
        mag_logit = self.prob_head(s)  # (B, N_out, 1)
        mag = keras.activations.sigmoid(mag_logit)

        v = mag * direction  # (B, N_out, D_out), ||v|| ∈ (0, 1) by construction

        # Optional load-balancing auxiliary loss (training only).
        if self.use_load_balancing and training:
            # Importance: mean assignment per output capsule, averaged over
            # batch and input capsules. Penalize variance to encourage
            # uniform usage. Mirrors Shazeer et al. (2017) "importance loss".
            if self.softmax_axis == "output":
                # a normalised over output axis -> usage = mean over (B, N_in)
                usage = keras.ops.mean(a, axis=(0, 1))  # (N_out,)
            else:
                # a normalised over input axis -> usage = mean over (B, N_out)
                usage = keras.ops.mean(a, axis=(0, 2))  # (N_in,)
            # Coefficient of variation squared: cv^2 = var / mean^2.
            mean_u = keras.ops.mean(usage) + self.eps
            var_u = keras.ops.mean(keras.ops.square(usage - mean_u))
            aux = var_u / (keras.ops.square(mean_u) + self.eps)
            self.add_loss(self.load_balancing_weight * aux)

        return v

    # ------------------------------------------------------------------
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_capsules, self.dim_capsules)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_capsules": self.num_capsules,
                "dim_capsules": self.dim_capsules,
                "softmax_axis": self.softmax_axis,
                "top_k": self.top_k,
                "use_bias": self.use_bias,
                "use_load_balancing": self.use_load_balancing,
                "load_balancing_weight": self.load_balancing_weight,
                "eps": self.eps,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            }
        )
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CapsuleBlockV2(keras.layers.Layer):
    """Capsule block built on :class:`AttentionRoutingCapsule`.

    Wraps :class:`AttentionRoutingCapsule` with optional dropout and an
    optional length-preserving direction-only LayerNormalization. The
    direction-only normalizer matches the bug-fixed behavior of the legacy
    :class:`CapsuleBlock` — it normalizes the unit-direction subspace
    without rescaling capsule magnitudes (which encode detection
    probability).

    Args:
        num_capsules: Number of output capsules.
        dim_capsules: Dimension of each capsule vector.
        dropout_rate: Dropout in ``[0.0, 1.0)``. Defaults to ``0.0``.
        direction_only_norm: Whether to apply length-preserving LayerNorm
            after the routing capsule. Defaults to ``False``.
        **routing_kwargs: Forwarded to :class:`AttentionRoutingCapsule`.
    """

    def __init__(
        self,
        num_capsules: int,
        dim_capsules: int,
        dropout_rate: float = 0.0,
        direction_only_norm: bool = False,
        softmax_axis: Literal["output", "input"] = "output",
        top_k: Optional[int] = None,
        use_bias: bool = True,
        use_load_balancing: bool = False,
        load_balancing_weight: float = 0.01,
        eps: float = 1e-7,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")
        if not isinstance(direction_only_norm, bool):
            raise TypeError(
                f"direction_only_norm must be boolean, got {type(direction_only_norm)}"
            )

        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.dropout_rate = float(dropout_rate)
        self.direction_only_norm = direction_only_norm
        self.softmax_axis = softmax_axis
        self.top_k = top_k
        self.use_bias = use_bias
        self.use_load_balancing = use_load_balancing
        self.load_balancing_weight = float(load_balancing_weight)
        self.eps = float(eps)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        self.routing = AttentionRoutingCapsule(
            num_capsules=self.num_capsules,
            dim_capsules=self.dim_capsules,
            softmax_axis=self.softmax_axis,
            top_k=self.top_k,
            use_bias=self.use_bias,
            use_load_balancing=self.use_load_balancing,
            load_balancing_weight=self.load_balancing_weight,
            eps=self.eps,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="block_routing",
        )

        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(self.dropout_rate, name="block_dropout")
        else:
            self.dropout = None

        if self.direction_only_norm:
            self.layer_norm = keras.layers.LayerNormalization(axis=-1, name="block_dir_norm")
        else:
            self.layer_norm = None

    def build(self, input_shape):
        self.routing.build(input_shape)
        out_shape = self.routing.compute_output_shape(input_shape)
        if self.dropout is not None:
            self.dropout.build(out_shape)
        if self.layer_norm is not None:
            self.layer_norm.build(out_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        x = self.routing(inputs, training=training)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        if self.layer_norm is not None:
            # Length-preserving direction-only LN — same recipe as the
            # bug-fixed legacy CapsuleBlock.
            mag = keras.ops.sqrt(keras.ops.sum(keras.ops.square(x), axis=-1, keepdims=True) + self.eps)
            direction = x / mag
            direction_normed = self.layer_norm(direction, training=training)
            dir_mag = keras.ops.sqrt(
                keras.ops.sum(keras.ops.square(direction_normed), axis=-1, keepdims=True) + self.eps
            )
            direction_unit = direction_normed / dir_mag
            x = mag * direction_unit

        return x

    def compute_output_shape(self, input_shape):
        return self.routing.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_capsules": self.num_capsules,
                "dim_capsules": self.dim_capsules,
                "dropout_rate": self.dropout_rate,
                "direction_only_norm": self.direction_only_norm,
                "softmax_axis": self.softmax_axis,
                "top_k": self.top_k,
                "use_bias": self.use_bias,
                "use_load_balancing": self.use_load_balancing,
                "load_balancing_weight": self.load_balancing_weight,
                "eps": self.eps,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            }
        )
        return config


# ---------------------------------------------------------------------
