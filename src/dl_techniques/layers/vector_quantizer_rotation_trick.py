"""
Rotation Trick Vector Quantizer.

This module implements the Rotation Trick VQ-VAE (Fifty et al., ICLR 2025) as a
Keras 3 layer. The Rotation Trick replaces the standard straight-through
estimator with a rotation + scaling that propagates a more informative gradient
from the discrete codebook lookup back through the quantization step. Forward
values are identical to a standard VQ (nearest codebook entry); only the
gradient path differs.

Architecture
------------

::

    Input x : (B, ..., D)
        |
        v
    +--------------------------------------------------+
    |  reshape -> (N, D)     N = B * prod(spatial)     |
    +--------------------------------------------------+
        |
        v   split channels into num_heads, D_h = D / H
    +--------------------------------------------------+
    |  flat : (N, H, D_h)                              |
    +--------------------------------------------------+
        |
        v
    +--------------------------------------------------+
    |  Codebook lookup        E : (H, K, D_h)          |
    |    'euclidean' : argmin ||x - e||                |
    |    'cosine'    : argmax <x_hat, e_hat>           |
    +--------------------------------------------------+
        |                                              |
        | indices (N, H)                               | q_lookup (N,H,D_h)
        |                                              |
        |                  +---------------------------+
        |                  |
        |                  v
        |   +-----------------------------------------+
        |   |  Gradient transform                     |
        |   |   'ste'           : x + sg(q - x)       |
        |   |   'rotation'      : R(x) * scale        |
        |   |   'reflection'    : Ref(x) * scale      |
        |   |   'no_grad_scale' : R(x), scale w/ grad |
        |   +-----------------------------------------+
        |                  |
        |                  v   q_out (N, H, D_h)
        |                  |
        |   +-----------------------------------------+
        |   |  Aux losses (training only)             |
        |   |    commitment : ||x - sg(q)||^2         |
        |   |    codebook   : ||sg(x) - q||^2         |
        |   |                (skipped when use_ema)   |
        |   |    diversity  : -H(p_avg)               |
        |   |    orthogonal : ||E E^T - I||^2         |
        |   +-----------------------------------------+
        |                  |
        |                  v
        |   +-----------------------------------------+
        |   |  EMA + dead-code (training, optional)   |
        |   |    cluster_size  <- decay-EMA           |
        |   |    embed_avg     <- decay-EMA           |
        |   |    reinit codes with hit < tau          |
        |   +-----------------------------------------+
        |                  |
        v                  v
    indices (B, ...)   reshape -> (B, ..., D)   output

``sg`` denotes ``stop_gradient``. Very-efficient rotation form:

::

    u_x = sg(x / ||x||);  u_q = sg(q / ||q||)
    w   = sg((u_x + u_q) / ||u_x + u_q||)
    R(x) = x - 2 (x . w) w + 2 (x . u_x) u_q
    scale = sg(||q|| / ||x||)         # 'rotation' mode
    quantized = R(x) * scale

``'reflection'`` drops the ``+2(x . u_x) u_q`` term. ``'no_grad_scale'`` keeps
``scale`` differentiable.

Beyond the rotation gradient, this implementation is a strict superset of the
existing ``VectorQuantizer`` adding:

- ``gradient_mode``: ``'rotation'`` | ``'reflection'`` | ``'no_grad_scale'`` |
  ``'ste'`` (back-compat).
- ``distance_mode``: ``'euclidean'`` (default) | ``'cosine'``. Cosine mode does
  angular nearest neighbour search and restores magnitude via ``||x||``.
- Multi-head codebooks (``num_heads``) — channel split into independent groups.
- EMA codebook updates (``use_ema=True``) with multi-head factorisation.
- Dead-code expiration (``dead_code_threshold`` + ``dead_code_reinit``).
- K-means warm start (``kmeans_init=True``, deterministic).
- Diversity penalty + orthogonal regularisation (optional aux losses).

Mathematical formulation (very-efficient rotation form per the paper):

.. math::

    \\hat{u}_x = x / \\|x\\|, \\quad \\hat{u}_q = q / \\|q\\|
    w_{unnorm} = \\hat{u}_x + \\hat{u}_q,  \\quad w = w_{unnorm} / \\|w_{unnorm}\\|
    R(x) = x - 2 (x \\cdot w) w + 2 (x \\cdot \\hat{u}_x) \\hat{u}_q
    scale = \\|q\\| / \\|x\\|
    quantized = R(x) \\cdot scale

with ``stop_gradient`` applied to ``w``, ``\\hat{u}_x``, ``\\hat{u}_q`` and
(depending on mode) ``scale``. The reflection variant drops the
``+2(x \\cdot \\hat{u}_x)\\hat{u}_q`` term.

References:
    - Fifty, C., Junkins, R., Duan, D., et al. (2025). Restructuring Vector
      Quantization with the Rotation Trick. ICLR 2025.
    - van den Oord, A., Vinyals, O., Kavukcuoglu, K. (2017). Neural Discrete
      Representation Learning. NeurIPS 2017. arXiv:1711.00937.
    - Razavi, A., van den Oord, A., Vinyals, O. (2019). Generating Diverse
      High-Fidelity Images with VQ-VAE-2. NeurIPS 2019.
    - Yu, J., Li, X., Koh, J. Y., et al. (2022). Vector-quantized Image Modeling
      with Improved VQGAN. ICLR 2022.
"""

from typing import Any, Dict, Optional, Tuple, Union

import keras
import numpy as np
from keras import initializers, layers, ops

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VectorQuantizerRotationTrick(layers.Layer):
    """Vector Quantizer with Rotation Trick gradient + multi-head codebook.

    A strict superset of ``VectorQuantizer``. Setting ``gradient_mode='ste'``
    and ``num_heads=1, distance_mode='euclidean', use_ema=False`` recovers the
    existing layer's behaviour bit-equivalently (atol<=1e-6).

    :param num_embeddings: Codebook size per head (``K``).
    :param embedding_dim: Total channel dim ``D``. With multi-head the codebook
        shape is ``(num_heads, K, D/num_heads)``.
    :param commitment_cost: Weight for commitment loss (``beta``).
    :param gradient_mode: One of ``'rotation'``, ``'reflection'``,
        ``'no_grad_scale'``, ``'ste'``.
    :param distance_mode: ``'euclidean'`` or ``'cosine'``.
    :param initializer: Codebook initializer.
    :param use_ema: Use EMA codebook updates instead of gradient updates.
    :param ema_decay: EMA decay rate.
    :param epsilon: Numerical floor for norms / EMA / cosine.
    :param num_heads: Number of independent codebook heads (channel split).
    :param kmeans_init: Run a one-shot k-means warm start on first training call.
    :param kmeans_init_steps: Number of mini-batches to accumulate for k-means.
    :param kmeans_seed: Deterministic numpy seed for k-means.
    :param dead_code_threshold: Consecutive unused-call count after which a
        code is re-initialised. 0 disables.
    :param diversity_coefficient: Weight for codebook diversity penalty.
    :param orthogonal_reg_coefficient: Weight for SRIP-style orthogonal penalty.
    """

    _GRAD_MODES = ("rotation", "reflection", "no_grad_scale", "ste")
    _DIST_MODES = ("euclidean", "cosine")

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            commitment_cost: float = 0.25,
            gradient_mode: str = "rotation",
            distance_mode: str = "euclidean",
            initializer: Union[str, initializers.Initializer] = "uniform",
            use_ema: bool = False,
            ema_decay: float = 0.99,
            epsilon: float = 1e-5,
            num_heads: int = 1,
            kmeans_init: bool = False,
            kmeans_init_steps: int = 1,
            kmeans_seed: int = 42,
            dead_code_threshold: int = 0,
            diversity_coefficient: float = 0.0,
            orthogonal_reg_coefficient: float = 0.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # ---- Validation ----
        if num_embeddings <= 0:
            raise ValueError(f"num_embeddings must be positive, got {num_embeddings}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if commitment_cost < 0:
            raise ValueError(f"commitment_cost must be non-negative, got {commitment_cost}")
        if gradient_mode not in self._GRAD_MODES:
            raise ValueError(
                f"gradient_mode must be one of {self._GRAD_MODES}, got {gradient_mode!r}"
            )
        if distance_mode not in self._DIST_MODES:
            raise ValueError(
                f"distance_mode must be one of {self._DIST_MODES}, got {distance_mode!r}"
            )
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if use_ema and not (0 < ema_decay < 1):
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if kmeans_init_steps <= 0:
            raise ValueError(f"kmeans_init_steps must be positive, got {kmeans_init_steps}")
        if dead_code_threshold < 0:
            raise ValueError(f"dead_code_threshold must be non-negative, got {dead_code_threshold}")
        if diversity_coefficient < 0:
            raise ValueError(
                f"diversity_coefficient must be non-negative, got {diversity_coefficient}"
            )
        if orthogonal_reg_coefficient < 0:
            raise ValueError(
                f"orthogonal_reg_coefficient must be non-negative, "
                f"got {orthogonal_reg_coefficient}"
            )

        # ---- Configuration ----
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.gradient_mode = gradient_mode
        self.distance_mode = distance_mode
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.kmeans_init = kmeans_init
        self.kmeans_init_steps = kmeans_init_steps
        self.kmeans_seed = kmeans_seed
        self.dead_code_threshold = dead_code_threshold
        self.diversity_coefficient = diversity_coefficient
        self.orthogonal_reg_coefficient = orthogonal_reg_coefficient

        if isinstance(initializer, str):
            self.initializer = initializers.get(initializer)
        else:
            self.initializer = initializer

        # Will be created in build()
        self.embeddings = None
        self.ema_cluster_size = None
        self.ema_embeddings = None
        self.dead_code_unused = None
        self.kmeans_init_done = None
        self._kmeans_accum = []  # python list, NOT a weight

        # K-means availability check (deferred to first use)
        if self.kmeans_init:
            try:
                import sklearn.cluster  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "kmeans_init=True requires scikit-learn. Install scikit-learn "
                    ">= 1.6.1 or set kmeans_init=False."
                ) from exc

    # ------------------------------------------------------------------
    # build / config
    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if input_shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Input last dimension {input_shape[-1]} must match "
                f"embedding_dim {self.embedding_dim}"
            )

        # Codebook shape: (num_heads, K, head_dim)
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.num_heads, self.num_embeddings, self.head_dim),
            initializer=self.initializer,
            trainable=not self.use_ema,
        )

        if self.use_ema:
            self.ema_cluster_size = self.add_weight(
                name="ema_cluster_size",
                shape=(self.num_heads, self.num_embeddings),
                initializer="zeros",
                trainable=False,
            )
            self.ema_embeddings = self.add_weight(
                name="ema_embeddings",
                shape=(self.num_heads, self.num_embeddings, self.head_dim),
                initializer=self.initializer,
                trainable=False,
            )

        if self.dead_code_threshold > 0:
            self.dead_code_unused = self.add_weight(
                name="dead_code_unused",
                shape=(self.num_heads, self.num_embeddings),
                initializer="zeros",
                trainable=False,
            )

        if self.kmeans_init:
            self.kmeans_init_done = self.add_weight(
                name="kmeans_init_done",
                shape=(),
                initializer="zeros",
                trainable=False,
                dtype="float32",
            )

        super().build(input_shape)

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_embeddings": self.num_embeddings,
                "embedding_dim": self.embedding_dim,
                "commitment_cost": self.commitment_cost,
                "gradient_mode": self.gradient_mode,
                "distance_mode": self.distance_mode,
                "initializer": initializers.serialize(self.initializer),
                "use_ema": self.use_ema,
                "ema_decay": self.ema_decay,
                "epsilon": self.epsilon,
                "num_heads": self.num_heads,
                "kmeans_init": self.kmeans_init,
                "kmeans_init_steps": self.kmeans_init_steps,
                "kmeans_seed": self.kmeans_seed,
                "dead_code_threshold": self.dead_code_threshold,
                "diversity_coefficient": self.diversity_coefficient,
                "orthogonal_reg_coefficient": self.orthogonal_reg_coefficient,
            }
        )
        return config

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        input_shape = ops.shape(inputs)

        # Flatten everything but channels: (..., D) -> (N, D)
        flat_inputs = ops.reshape(inputs, (-1, self.embedding_dim))

        # K-means warm start (Python side; one-shot)
        if (
                self.kmeans_init
                and training is True
                and float(ops.convert_to_numpy(self.kmeans_init_done)) < 0.5
        ):
            self._maybe_kmeans_init(flat_inputs)

        # Reshape to (N, H, head_dim)
        flat_heads = ops.reshape(flat_inputs, (-1, self.num_heads, self.head_dim))

        # Per-head argmin / argmax
        encoding_indices, quantized_heads = self._lookup(flat_heads)
        # encoding_indices: (N, H) int
        # quantized_heads: (N, H, head_dim) float

        # EMA + dead-code update (training only)
        if training is True:
            if self.use_ema:
                self._update_ema(flat_heads, encoding_indices)
            if self.dead_code_threshold > 0:
                self._update_dead_codes(flat_heads, encoding_indices)

        # Auxiliary losses (training-gated for diversity/ortho; commitment/codebook always)
        # Reshape quantized back to (N, D)
        quantized_flat = ops.reshape(quantized_heads, (-1, self.embedding_dim))

        # Codebook + commitment losses
        codebook_loss = ops.mean(
            ops.square(ops.stop_gradient(flat_inputs) - quantized_flat)
        )
        commitment_loss = self.commitment_cost * ops.mean(
            ops.square(flat_inputs - ops.stop_gradient(quantized_flat))
        )
        self.add_loss(codebook_loss)
        self.add_loss(commitment_loss)

        # Optional aux losses
        if training is True and self.diversity_coefficient > 0:
            self.add_loss(self.diversity_coefficient * self._diversity_loss())
        if training is True and self.orthogonal_reg_coefficient > 0:
            self.add_loss(self.orthogonal_reg_coefficient * self._orthogonal_loss())

        # Gradient transform
        transformed_flat = self._apply_gradient_transform(flat_inputs, quantized_flat)

        # Restore shape
        output = ops.reshape(transformed_flat, input_shape)
        return output

    # ------------------------------------------------------------------
    # lookup helpers
    # ------------------------------------------------------------------

    def _lookup(
            self, flat_heads: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Return (indices, quantized_heads) per head.

        :param flat_heads: ``(N, H, head_dim)``.
        :return: indices ``(N, H)`` int, quantized ``(N, H, head_dim)``.
        """
        # codebook: (H, K, head_dim)
        codebook = self.embeddings

        if self.distance_mode == "euclidean":
            # Squared distance per head:
            # ||x||^2 (N,H,1) + ||e||^2 (H,1,K) - 2 x.e (N,H,K)
            x_sq = ops.sum(ops.square(flat_heads), axis=-1, keepdims=True)  # (N,H,1)
            e_sq = ops.sum(ops.square(codebook), axis=-1)  # (H,K)
            e_sq = ops.expand_dims(e_sq, axis=0)  # (1,H,K)
            # x . e: einsum over head_dim
            # flat_heads (N,H,D) x codebook (H,K,D) -> (N,H,K)
            xe = ops.einsum("nhd,hkd->nhk", flat_heads, codebook)
            distances = x_sq + e_sq - 2.0 * xe
            indices = ops.argmin(distances, axis=-1)  # (N,H)
        else:  # cosine
            # L2-normalise both
            x_norm = ops.sqrt(
                ops.sum(ops.square(flat_heads), axis=-1, keepdims=True) + self.epsilon
            )
            unit_x = flat_heads / x_norm  # (N,H,D)
            e_norm = ops.sqrt(
                ops.sum(ops.square(codebook), axis=-1, keepdims=True) + self.epsilon
            )
            unit_e = codebook / e_norm  # (H,K,D)
            sim = ops.einsum("nhd,hkd->nhk", unit_x, unit_e)
            indices = ops.argmax(sim, axis=-1)  # (N,H)

        # Gather quantized vectors per head.
        # one_hot indices to (N,H,K), then matmul against codebook (H,K,D)
        encodings = ops.one_hot(indices, self.num_embeddings)  # (N,H,K)
        # quantized = sum_k encodings * codebook[h,k] -> (N,H,D)
        quantized = ops.einsum("nhk,hkd->nhd", encodings, codebook)

        if self.distance_mode == "cosine":
            # Restore magnitude
            x_mag = ops.sqrt(
                ops.sum(ops.square(flat_heads), axis=-1, keepdims=True) + self.epsilon
            )
            q_mag = ops.sqrt(
                ops.sum(ops.square(quantized), axis=-1, keepdims=True) + self.epsilon
            )
            quantized = quantized * (x_mag / q_mag)

        return indices, quantized

    # ------------------------------------------------------------------
    # gradient transform
    # ------------------------------------------------------------------

    def _apply_gradient_transform(
            self, x: keras.KerasTensor, q: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Apply selected gradient transform on flat tensors ``(N, D)``."""
        mode = self.gradient_mode

        if mode == "ste":
            return x + ops.stop_gradient(q - x)

        # Promote to fp32 for norms.
        x_dtype = x.dtype
        x32 = ops.cast(x, "float32")
        q32 = ops.cast(q, "float32")

        eps = self.epsilon

        # Per the paper: the geometric anchors (unit_x, unit_q, w, and scale unless
        # 'no_grad_scale') are computed with stop_gradient so that the rotation
        # matrix R becomes a *constant* w.r.t. backprop. The gradient w.r.t. x
        # then flows through R @ x as a constant linear transform — preserving
        # the curvature/direction information that pure STE discards.
        x_norm = ops.sqrt(ops.sum(ops.square(x32), axis=-1, keepdims=True) + eps)
        q_norm = ops.sqrt(ops.sum(ops.square(q32), axis=-1, keepdims=True) + eps)

        # Detached unit vectors and w direction.
        unit_x_sg = ops.stop_gradient(x32 / x_norm)
        unit_q_sg = ops.stop_gradient(q32 / q_norm)
        w_unnorm = unit_x_sg + unit_q_sg
        w_norm = ops.sqrt(ops.sum(ops.square(w_unnorm), axis=-1, keepdims=True) + eps)
        w_sg = ops.stop_gradient(w_unnorm / w_norm)

        # x · w and x · unit_x — gradient WILL flow through x here (the whole point).
        x_dot_w = ops.sum(x32 * w_sg, axis=-1, keepdims=True)
        x_dot_ux = ops.sum(x32 * unit_x_sg, axis=-1, keepdims=True)

        if mode == "rotation":
            rotated = x32 - 2.0 * x_dot_w * w_sg + 2.0 * x_dot_ux * unit_q_sg
        elif mode == "reflection":
            rotated = x32 - 2.0 * x_dot_w * w_sg
        elif mode == "no_grad_scale":
            rotated = x32 - 2.0 * x_dot_w * w_sg + 2.0 * x_dot_ux * unit_q_sg
        else:  # pragma: no cover (validated in __init__)
            raise ValueError(f"Unknown gradient_mode: {mode}")

        # Scale: q_norm/x_norm — by default detached so it does not perturb the
        # gradient direction; 'no_grad_scale' lets scale's gradient flow.
        if mode == "no_grad_scale":
            scale_eff = q_norm / x_norm
        else:
            scale_eff = ops.stop_gradient(q_norm / x_norm)

        out32 = rotated * scale_eff
        return ops.cast(out32, x_dtype)

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def _update_ema(
            self,
            flat_heads: keras.KerasTensor,
            indices: keras.KerasTensor,
    ) -> None:
        """EMA update per head.

        :param flat_heads: ``(N, H, head_dim)``.
        :param indices: ``(N, H)`` int.
        """
        encodings = ops.one_hot(indices, self.num_embeddings)  # (N,H,K)
        # cluster_size: per (H,K) -> sum over N
        cluster_size = ops.sum(encodings, axis=0)  # (H,K)
        # embed sums: (H,K,D) = sum_n encodings[n,h,k] * flat_heads[n,h,d]
        embed_sums = ops.einsum("nhk,nhd->hkd", encodings, flat_heads)

        new_cluster = (
                self.ema_decay * self.ema_cluster_size
                + (1.0 - self.ema_decay) * cluster_size
        )
        self.ema_cluster_size.assign(new_cluster)

        new_embed = (
                self.ema_decay * self.ema_embeddings
                + (1.0 - self.ema_decay) * embed_sums
        )
        self.ema_embeddings.assign(new_embed)

        normalised = self.ema_embeddings / ops.expand_dims(
            self.ema_cluster_size + self.epsilon, axis=-1
        )
        self.embeddings.assign(normalised)

    # ------------------------------------------------------------------
    # dead-code expiration
    # ------------------------------------------------------------------

    def _update_dead_codes(
            self,
            flat_heads: keras.KerasTensor,
            indices: keras.KerasTensor,
    ) -> None:
        """Track unused codes and re-init expired ones from current batch."""
        encodings = ops.one_hot(indices, self.num_embeddings)  # (N,H,K)
        used_this_call = ops.cast(ops.sum(encodings, axis=0) > 0, "float32")  # (H,K)

        # Increment unused counter for codes not used; reset for codes used.
        new_unused = (1.0 - used_this_call) * (self.dead_code_unused + 1.0)
        self.dead_code_unused.assign(new_unused)

        # Find dead codes: unused > threshold.
        # We then replace each dead code with a random encoder vector from this batch
        # (per head). For correctness in pure Keras ops we sample via shuffle.
        dead_mask = ops.cast(
            self.dead_code_unused > float(self.dead_code_threshold), "float32"
        )  # (H, K)

        # Sample replacement vectors per head from flat_heads (N, H, head_dim).
        n = ops.shape(flat_heads)[0]
        # Random indices in [0, N), shape (H, K) — pick one batch vector per dead slot.
        rand_uniform = keras.random.uniform(
            shape=(self.num_heads, self.num_embeddings),
            minval=0.0,
            maxval=1.0,
        )
        rand_idx = ops.cast(rand_uniform * ops.cast(n, "float32"), "int32")
        rand_idx = ops.clip(rand_idx, 0, n - 1)  # (H, K)

        # Gather: replacements[h, k, :] = flat_heads[rand_idx[h, k], h, :]
        # flat_heads is (N, H, D); we want (H, K, D).
        # Build with take + per-head indexing via vectorisation.
        # take along axis=0 with indices (H,K) -> result (H,K,H,D); we need diagonal in H.
        # Simpler: transpose flat_heads to (H, N, D) then gather along axis=1 per head.
        heads_first = ops.transpose(flat_heads, (1, 0, 2))  # (H, N, D)
        # gather indices rand_idx (H, K) along axis=1.
        replacements = ops.take_along_axis(
            heads_first,
            ops.expand_dims(rand_idx, axis=-1),  # (H,K,1)
            axis=1,
        )  # (H, K, D)  -- ops.take_along_axis broadcasts the last dim

        # Blend: new_codebook = dead_mask * replacements + (1 - dead_mask) * embeddings
        dead_mask_exp = ops.expand_dims(dead_mask, axis=-1)  # (H,K,1)
        new_codebook = (
                dead_mask_exp * replacements + (1.0 - dead_mask_exp) * self.embeddings
        )
        self.embeddings.assign(new_codebook)

        # Reset unused counter for revived codes.
        revived_unused = (1.0 - dead_mask) * self.dead_code_unused
        self.dead_code_unused.assign(revived_unused)

    # ------------------------------------------------------------------
    # k-means warm start
    # ------------------------------------------------------------------

    def _maybe_kmeans_init(self, flat_inputs: keras.KerasTensor) -> None:
        """Accumulate batches then run MiniBatchKMeans once."""
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as exc:  # defensive — already checked in __init__
            raise RuntimeError(
                "kmeans_init=True requires scikit-learn."
            ) from exc

        # Accumulate this batch's encoder vectors as a numpy array.
        # flat_inputs shape (N, D); reshape per head to (N*H, head_dim).
        np_inputs = np.asarray(ops.convert_to_numpy(flat_inputs))
        np_heads = np_inputs.reshape(-1, self.num_heads, self.head_dim)
        self._kmeans_accum.append(np_heads)

        if len(self._kmeans_accum) < self.kmeans_init_steps:
            return  # need more batches

        all_batches = np.concatenate(self._kmeans_accum, axis=0)  # (N_total, H, D_h)
        new_codebook = np.zeros(
            (self.num_heads, self.num_embeddings, self.head_dim), dtype=np.float32
        )
        for h in range(self.num_heads):
            head_vectors = all_batches[:, h, :]  # (N_total, D_h)
            if head_vectors.shape[0] < self.num_embeddings:
                logger.warning(
                    f"kmeans_init: head {h} has only "
                    f"{head_vectors.shape[0]} samples for "
                    f"{self.num_embeddings} clusters; falling back to "
                    "current codebook for missing centroids."
                )
                km = MiniBatchKMeans(
                    n_clusters=max(2, min(self.num_embeddings, head_vectors.shape[0])),
                    random_state=self.kmeans_seed + h,
                    n_init=3,
                )
                km.fit(head_vectors.astype(np.float32))
                centroids = km.cluster_centers_
                # pad with existing codebook entries
                existing = np.asarray(
                    ops.convert_to_numpy(self.embeddings)
                )[h]  # (K, D_h)
                pad = self.num_embeddings - centroids.shape[0]
                centroids = np.concatenate([centroids, existing[-pad:]], axis=0)
            else:
                km = MiniBatchKMeans(
                    n_clusters=self.num_embeddings,
                    random_state=self.kmeans_seed + h,
                    n_init=3,
                )
                km.fit(head_vectors.astype(np.float32))
                centroids = km.cluster_centers_
            new_codebook[h] = centroids.astype(np.float32)

        self.embeddings.assign(new_codebook)
        self.kmeans_init_done.assign(1.0)
        self._kmeans_accum = []
        logger.info(
            f"kmeans_init: codebook initialised from "
            f"{all_batches.shape[0]} samples across {self.num_heads} head(s)."
        )

    # ------------------------------------------------------------------
    # aux losses
    # ------------------------------------------------------------------

    def _diversity_loss(self) -> keras.KerasTensor:
        """Penalise mean off-diagonal of unit-codebook gram matrix per head."""
        e = self.embeddings  # (H, K, D)
        norm = ops.sqrt(ops.sum(ops.square(e), axis=-1, keepdims=True) + self.epsilon)
        unit = e / norm  # (H, K, D)
        gram = ops.einsum("hkd,hjd->hkj", unit, unit)  # (H, K, K)
        eye = ops.eye(self.num_embeddings)
        eye = ops.expand_dims(eye, axis=0)  # (1, K, K)
        off_diag = gram - eye
        loss = ops.mean(ops.square(off_diag))
        return loss

    def _orthogonal_loss(self) -> keras.KerasTensor:
        """SRIP-style ``||E E^T - I||_F^2`` summed across heads."""
        e = self.embeddings  # (H, K, D)
        gram = ops.einsum("hkd,hjd->hkj", e, e)
        eye = ops.expand_dims(ops.eye(self.num_embeddings), axis=0)
        diff = gram - eye
        return ops.mean(ops.sum(ops.square(diff), axis=(-1, -2)))

    # ------------------------------------------------------------------
    # public API parity with VectorQuantizer
    # ------------------------------------------------------------------

    def get_codebook_indices(
            self, inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Return discrete indices of nearest codebook entries per head.

        :param inputs: ``(B, ..., D)``.
        :return: ``(B, ..., num_heads)`` int. When ``num_heads==1`` the trailing
            head dim is squeezed for parity with ``VectorQuantizer`` (which
            returns ``(B, ...)``).
        """
        if not self.built:
            self.build(inputs.shape)

        input_shape = ops.shape(inputs)
        spatial_shape = input_shape[:-1]

        flat = ops.reshape(inputs, (-1, self.embedding_dim))
        flat_heads = ops.reshape(flat, (-1, self.num_heads, self.head_dim))
        indices, _ = self._lookup(flat_heads)  # (N, H)

        if self.num_heads == 1:
            indices_out = ops.reshape(indices[:, 0], spatial_shape)
            return indices_out

        # Reshape (N, H) back to (B, ..., H).
        spatial_shape_i32 = ops.cast(spatial_shape, "int32")
        h_tensor = ops.convert_to_tensor([self.num_heads], dtype="int32")
        out_shape = ops.concatenate([spatial_shape_i32, h_tensor], axis=0)
        return ops.reshape(indices, out_shape)

    def quantize_from_indices(
            self, indices: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Convert indices back to embedding vectors.

        :param indices: ``(B, ..., num_heads)`` int, or ``(B, ...)`` when
            ``num_heads==1`` (parity with ``VectorQuantizer``).
        """
        if not self.built:
            raise ValueError("Layer must be built before calling quantize_from_indices")

        idx_shape = ops.shape(indices)

        if self.num_heads == 1:
            flat_indices = ops.reshape(indices, (-1,))  # (N,)
            flat_indices = ops.expand_dims(flat_indices, axis=-1)  # (N, 1)
            spatial_shape_i32 = ops.cast(idx_shape, "int32")
        else:
            flat_indices = ops.reshape(indices, (-1, self.num_heads))  # (N, H)
            # Spatial shape is idx_shape[:-1] (the last axis is heads).
            spatial_shape_i32 = ops.cast(idx_shape[:-1], "int32")

        encodings = ops.one_hot(flat_indices, self.num_embeddings)  # (N, H, K)
        quantized = ops.einsum("nhk,hkd->nhd", encodings, self.embeddings)  # (N,H,D)
        flat_q = ops.reshape(quantized, (-1, self.embedding_dim))  # (N, D)

        d_tensor = ops.convert_to_tensor([self.embedding_dim], dtype="int32")
        out_shape = ops.concatenate([spatial_shape_i32, d_tensor], axis=0)
        return ops.reshape(flat_q, out_shape)
