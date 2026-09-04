"""
VectorQuantizerRotationTrick, a vector quantizer with the Rotation Trick
gradient estimator.

The forward pass is a standard nearest-neighbour lookup in a learned
codebook. The backward pass differs from the usual straight-through
estimator: instead of copying the reconstruction gradient from the quantized
output straight onto the encoder output, the layer treats the map from
encoder output to codebook vector as a rotation plus a rescale, and applies
that same linear operator to the gradient. This carries directional and
curvature information through the discrete bottleneck instead of discarding
it, at the cost of two extra unit-vector computations per call. Multi-head
factorization splits the channel dimension into independent codebooks,
raising the effective vocabulary to ``K^num_heads`` at linear memory cost.

Setting ``gradient_mode='ste'`` with ``num_heads=1``, ``distance_mode='euclidean'``,
and ``use_ema=False`` reproduces a classical vector quantizer's behaviour to
within floating point tolerance. ``warm_start_codebook`` must be called
eagerly before training, never from inside ``call()`` — it runs scikit-learn
and assigns a variable directly, neither of which is graph-safe.

References:
    - Fifty et al., 2025. Restructuring Vector Quantization with the Rotation
      Trick. ICLR 2025. (https://arxiv.org/abs/2410.06424)
    - van den Oord et al., 2017. Neural Discrete Representation Learning.
      (https://arxiv.org/abs/1711.00937)
    - Razavi et al., 2019. Generating Diverse High-Fidelity Images with
      VQ-VAE-2. (https://arxiv.org/abs/1906.00446)
    - Yu et al., 2022. Vector-quantized Image Modeling with Improved VQGAN.
      (https://arxiv.org/abs/2110.04627)
    - Bengio et al., 2013. Estimating or Propagating Gradients Through
      Stochastic Neurons for Conditional Computation.
      (https://arxiv.org/abs/1308.3432)
"""


import keras
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.generative.vector_quantizer_rotation_trick")
class VectorQuantizerRotationTrick(keras.layers.Layer):
    """Vector quantizer with Rotation Trick gradient and multi-head codebook.

    A strict superset of ``VectorQuantizer``. Setting ``gradient_mode='ste'``
    and ``num_heads=1, distance_mode='euclidean', use_ema=False`` recovers the
    existing layer's behaviour bit-equivalently (atol<=1e-6).

    Architecture:

    .. code-block:: text

        ┌────────────────────────────────────────┐
        │  Input z_e [B, ..., D]                 │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Reshape → [N, D]  → split into heads  │
        │  flat: [N, H, D_h]   D_h = D / H       │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Codebook lookup  E: [H, K, D_h]       │
        │    'euclidean': k* = argmin ||z - e||  │
        │    'cosine'   : k* = argmax <ẑ, ê>     │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Gradient transform                    │
        │    'ste'          : z_e + sg(z_q - z_e)│
        │    'rotation'     : R(z_e) * scale     │
        │    'reflection'   : Ref(z_e) * scale   │
        │    'no_grad_scale': R(z_e)             │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Aux losses (training only)            │
        │    commitment : ||z_e - sg(z_q)||²     │
        │    codebook   : ||sg(z_e) - z_q||²     │
        │                  (skipped if use_ema)  │
        │    diversity  : -H(p_avg)              │
        │    orthogonal : ||E Eᵀ - I||²          │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  EMA + dead-code (training, optional)  │
        │    cluster_size ← decay·EMA            │
        │    embed_avg    ← decay·EMA            │
        │    reinit codes with hits < τ          │
        └──────────────┬─────────────────────────┘
                       ▼
        ┌────────────────────────────────────────┐
        │  Reshape to [B, ..., D]                │
        └────────────────────────────────────────┘

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
    :param kmeans_init: Enable the one-shot k-means warm start. This flag does
        NOT trigger it: the warm start runs scikit-learn and assigns a variable,
        neither of which is graph-safe, so it is performed by
        :meth:`warm_start_codebook`, called eagerly before training.
        :class:`~dl_techniques.models.vision.vq_vae_rotation.model.VQVAERotationTrick`
        calls it for you from its ``fit`` override.
    :param kmeans_init_steps: Number of mini-batches the caller should collect
        before invoking :meth:`warm_start_codebook`.
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
            initializer: Union[str, keras.initializers.Initializer] = "uniform",
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
            self.initializer = keras.initializers.get(initializer)
        else:
            self.initializer = initializer

        # Will be created in build()
        self.embeddings = None
        self.ema_cluster_size = None
        self.ema_embeddings = None
        self.ema_step = None
        self.dead_code_unused = None
        self.kmeans_init_done = None

        # K-means availability check (deferred to first use)
        if self.kmeans_init:
            try:
                import sklearn.cluster
            except ImportError as exc:
                raise RuntimeError(
                    "kmeans_init=True requires scikit-learn. Install scikit-learn "
                    ">= 1.6.1 or set kmeans_init=False."
                ) from exc


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
            # DECISION plan-2026-08-19T163559-499b6f0e/D-011: EMA accumulators are
            # float32 state, not autocast activations -- float16 loses assignment counts past 2048.
            # Mixing dtypes here raised AddV2 InvalidArgumentError under mixed_float16. See decisions.md.
            self.ema_cluster_size = self.add_weight(
                name="ema_cluster_size",
                shape=(self.num_heads, self.num_embeddings),
                initializer="zeros",
                trainable=False,
                dtype="float32",
                autocast=False,
            )
            # DECISION plan-2026-08-18T140459-7991552f/D-012: this accumulator starts
            # at zero, not at `self.initializer`, since `_update_ema`'s bias correction assumes a zero-start EMA numerator.
            # DECISION plan-2026-08-19T163559-499b6f0e/D-011 (see `ema_cluster_size`).
            self.ema_embeddings = self.add_weight(
                name="ema_embeddings",
                shape=(self.num_heads, self.num_embeddings, self.head_dim),
                initializer="zeros",
                trainable=False,
                dtype="float32",
                autocast=False,
            )

            # DECISION plan-2026-08-18T140459-7991552f/D-012: ema_step is read only by
            # the bias correction in `_update_ema`; keep it float32, not int32, or TF places it on CPU and jit-compiled training dies crossing devices.
            # DECISION plan-2026-08-19T163559-499b6f0e/D-011 (see `ema_cluster_size`).
            self.ema_step = self.add_weight(
                name="ema_step",
                shape=(),
                initializer="zeros",
                trainable=False,
                dtype="float32",
                autocast=False,
            )

        if self.dead_code_threshold > 0:
            # DECISION plan-2026-08-19T163559-499b6f0e/D-011 (see `ema_cluster_size`).
            self.dead_code_unused = self.add_weight(
                name="dead_code_unused",
                shape=(self.num_heads, self.num_embeddings),
                initializer="zeros",
                trainable=False,
                dtype="float32",
                autocast=False,
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
                "initializer": keras.initializers.serialize(self.initializer),
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


    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        input_shape = keras.ops.shape(inputs)

        # Flatten everything but channels: (..., D) -> (N, D)
        flat_inputs = keras.ops.reshape(inputs, (-1, self.embedding_dim))

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-040: no k-means work in `call()` --
        # it is traced, and a graph-tensor read plus a plain-list accumulator here broke under `model.fit()`. See decisions.md.

        # Reshape to (N, H, head_dim).
        flat_heads = keras.ops.reshape(flat_inputs, (-1, self.num_heads, self.head_dim))

        # encoding_indices: (N, H) int; quantized_heads: (N, H, head_dim) float.
        encoding_indices, quantized_heads = self._lookup(flat_heads)

        # EMA + dead-code update (training only)
        if training is True:
            if self.use_ema:
                self._update_ema(flat_heads, encoding_indices)
            if self.dead_code_threshold > 0:
                self._update_dead_codes(flat_heads, encoding_indices)

        # Auxiliary losses (training-gated for diversity/ortho; commitment/codebook always)
        # Reshape quantized back to (N, D)
        quantized_flat = keras.ops.reshape(quantized_heads, (-1, self.embedding_dim))

        # Codebook + commitment losses
        codebook_loss = keras.ops.mean(
            keras.ops.square(keras.ops.stop_gradient(flat_inputs) - quantized_flat)
        )
        commitment_loss = self.commitment_cost * keras.ops.mean(
            keras.ops.square(flat_inputs - keras.ops.stop_gradient(quantized_flat))
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
        output = keras.ops.reshape(transformed_flat, input_shape)
        return output


    def _lookup(
            self, flat_heads: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Return (indices, quantized_heads) per head.

        :param flat_heads: ``(N, H, head_dim)``.
        :return: indices ``(N, H)`` int, quantized ``(N, H, head_dim)``.
        """
        # codebook: (H, K, head_dim).
        codebook = self.embeddings

        if self.distance_mode == "euclidean":
            # Squared distance per head: ||x||^2 + ||e||^2 - 2 x.e.
            x_sq = keras.ops.sum(keras.ops.square(flat_heads), axis=-1, keepdims=True)
            e_sq = keras.ops.sum(keras.ops.square(codebook), axis=-1)
            e_sq = keras.ops.expand_dims(e_sq, axis=0)
            xe = keras.ops.einsum("nhd,hkd->nhk", flat_heads, codebook)
            distances = x_sq + e_sq - 2.0 * xe
            indices = keras.ops.argmin(distances, axis=-1)
        else:  # cosine
            # L2-normalise both.
            x_norm = keras.ops.sqrt(
                keras.ops.sum(keras.ops.square(flat_heads), axis=-1, keepdims=True) + self.epsilon
            )
            unit_x = flat_heads / x_norm
            e_norm = keras.ops.sqrt(
                keras.ops.sum(keras.ops.square(codebook), axis=-1, keepdims=True) + self.epsilon
            )
            unit_e = codebook / e_norm
            sim = keras.ops.einsum("nhd,hkd->nhk", unit_x, unit_e)
            indices = keras.ops.argmax(sim, axis=-1)

        # Gather quantized vectors per head: one_hot indices matmul codebook.
        # DECISION plan-2026-08-19T163559-499b6f0e/D-011: `dtype=self.compute_dtype`
        # avoids a float32 one_hot promoting the einsum, which raised under mixed_float16. See decisions.md.
        encodings = keras.ops.one_hot(
            indices, self.num_embeddings, dtype=self.compute_dtype
        )
        quantized = keras.ops.einsum("nhk,hkd->nhd", encodings, codebook)

        # DECISION plan-2026-08-17T183311-79c63e38/D-030: cosine mode returns the raw
        # codebook row like euclidean mode, never rescaled by x_mag/q_mag -- rescaling left codebook magnitudes untrained. See decisions.md.

        return indices, quantized


    def _apply_gradient_transform(
            self, x: keras.KerasTensor, q: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Apply selected gradient transform on flat tensors ``(N, D)``."""
        mode = self.gradient_mode

        if mode == "ste":
            return x + keras.ops.stop_gradient(q - x)

        # Promote to fp32 for norms.
        x_dtype = x.dtype
        x32 = keras.ops.cast(x, "float32")
        q32 = keras.ops.cast(q, "float32")

        eps = self.epsilon
        eps_sq = eps * eps

        # DECISION plan-2026-08-18T140459-7991552f/D-025: norms are floored
        # (`sqrt(max(sum(x^2), eps^2))`), not eps-regularised -- regularising inflates every norm and leaked a 4% magnitude error at small scale. See decisions.md.

        # The geometric anchors are detached with stop_gradient so the rotation
        # is a constant linear operator; gradient flows through x as R @ x.
        x_norm = keras.ops.sqrt(keras.ops.maximum(
            keras.ops.sum(keras.ops.square(x32), axis=-1, keepdims=True), eps_sq))
        q_norm = keras.ops.sqrt(keras.ops.maximum(
            keras.ops.sum(keras.ops.square(q32), axis=-1, keepdims=True), eps_sq))

        # Detached unit vectors and w direction.
        unit_x_sg = keras.ops.stop_gradient(x32 / x_norm)
        unit_q_sg = keras.ops.stop_gradient(q32 / q_norm)
        w_unnorm = unit_x_sg + unit_q_sg
        w_norm = keras.ops.sqrt(keras.ops.maximum(
            keras.ops.sum(keras.ops.square(w_unnorm), axis=-1, keepdims=True), eps_sq))
        w_sg = keras.ops.stop_gradient(w_unnorm / w_norm)

        # Gradient flows through x here, via these two dot products.
        x_dot_w = keras.ops.sum(x32 * w_sg, axis=-1, keepdims=True)
        x_dot_ux = keras.ops.sum(x32 * unit_x_sg, axis=-1, keepdims=True)

        if mode == "rotation":
            rotated = x32 - 2.0 * x_dot_w * w_sg + 2.0 * x_dot_ux * unit_q_sg
        elif mode == "reflection":
            # Negated Householder reflection about (u + v), mapping u -> +v.
            # Reflecting about (u - v) instead is equivalent but loses precision when x is near its codebook vector.
            rotated = 2.0 * x_dot_w * w_sg - x32
        elif mode == "no_grad_scale":
            rotated = x32 - 2.0 * x_dot_w * w_sg + 2.0 * x_dot_ux * unit_q_sg
        else:  # pragma: no cover (validated in __init__)
            raise ValueError(f"Unknown gradient_mode: {mode}")

        # Scale: q_norm/x_norm — by default detached so it does not perturb the
        # gradient direction; 'no_grad_scale' lets scale's gradient flow.
        if mode == "no_grad_scale":
            scale_eff = q_norm / x_norm
        else:
            scale_eff = keras.ops.stop_gradient(q_norm / x_norm)

        out32 = rotated * scale_eff
        return keras.ops.cast(out32, x_dtype)


    def _update_ema(
            self,
            flat_heads: keras.KerasTensor,
            indices: keras.KerasTensor,
    ) -> None:
        """EMA update per head.

        :param flat_heads: ``(N, H, head_dim)``.
        :param indices: ``(N, H)`` int.
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-011: this one_hot stays at the
        # float32 default (unlike `_lookup`'s) since EMA accumulators are float32 `autocast=False` state. See decisions.md.
        flat_heads = keras.ops.cast(flat_heads, "float32")
        encodings = keras.ops.one_hot(indices, self.num_embeddings)
        # cluster_size: per (H,K), summed over N.
        cluster_size = keras.ops.sum(encodings, axis=0)
        # embed_sums: (H,K,D) = sum_n encodings[n,h,k] * flat_heads[n,h,d].
        embed_sums = keras.ops.einsum("nhk,nhd->hkd", encodings, flat_heads)

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

        # DECISION plan-2026-08-18T140459-7991552f/D-012: keep both the bias
        # correction and the Laplace smoothing below -- without them a code with zero assignments blows up ~1e5 at step 1 (sibling test: max|codebook| 4521 -> 0.264). See decisions.md.
        self.ema_step.assign_add(1.0)
        bias_correction = 1.0 - keras.ops.power(
            keras.ops.cast(self.ema_decay, self.ema_step.dtype), self.ema_step
        )

        debiased_cluster_size = self.ema_cluster_size / bias_correction
        debiased_embeddings = self.ema_embeddings / bias_correction

        # Laplace smoothing of the counts; per-head total mass preserved.
        # total_count: (H, 1).
        total_count = keras.ops.sum(
            debiased_cluster_size, axis=-1, keepdims=True
        )
        smoothed_cluster_size = (
                (debiased_cluster_size + self.epsilon)
                / (total_count + self.num_embeddings * self.epsilon)
                * total_count
        )

        normalised = debiased_embeddings / keras.ops.expand_dims(
            smoothed_cluster_size, axis=-1
        )
        self.embeddings.assign(normalised)


    def _update_dead_codes(
            self,
            flat_heads: keras.KerasTensor,
            indices: keras.KerasTensor,
    ) -> None:
        """Track unused codes and re-init expired ones from current batch."""
        # DECISION plan-2026-08-19T163559-499b6f0e/D-011: whole routine runs in
        # float32 since `dead_code_unused` is `autocast=False` state -- an fp16 counter stops incrementing at 2048. See decisions.md.
        flat_heads = keras.ops.cast(flat_heads, "float32")
        encodings = keras.ops.one_hot(indices, self.num_embeddings)
        used_this_call = keras.ops.cast(keras.ops.sum(encodings, axis=0) > 0, "float32")

        # Increment unused counter for codes not used; reset for codes used.
        new_unused = (1.0 - used_this_call) * (self.dead_code_unused + 1.0)
        self.dead_code_unused.assign(new_unused)

        # Dead codes (unused > threshold) get replaced by a random batch vector.
        dead_mask = keras.ops.cast(
            self.dead_code_unused > float(self.dead_code_threshold), "float32"
        )

        # Sample replacement vectors per head from flat_heads (N, H, head_dim).
        n = keras.ops.shape(flat_heads)[0]
        rand_uniform = keras.random.uniform(
            shape=(self.num_heads, self.num_embeddings),
            minval=0.0,
            maxval=1.0,
        )
        rand_idx = keras.ops.cast(rand_uniform * keras.ops.cast(n, "float32"), "int32")
        rand_idx = keras.ops.clip(rand_idx, 0, n - 1)

        # Transpose to (H, N, D) and gather rand_idx (H, K) along axis=1.
        heads_first = keras.ops.transpose(flat_heads, (1, 0, 2))
        replacements = keras.ops.take_along_axis(
            heads_first,
            keras.ops.expand_dims(rand_idx, axis=-1),
            axis=1,
        )

        # new_codebook = dead_mask * replacements + (1 - dead_mask) * embeddings.
        dead_mask_exp = keras.ops.expand_dims(dead_mask, axis=-1)
        new_codebook = (
                dead_mask_exp * replacements
                + (1.0 - dead_mask_exp)
                * keras.ops.cast(self.embeddings, "float32")
        )
        self.embeddings.assign(new_codebook)

        # Reset unused counter for revived codes.
        revived_unused = (1.0 - dead_mask) * self.dead_code_unused
        self.dead_code_unused.assign(revived_unused)


    def warm_start_codebook(self, batches: Any) -> None:
        """Initialise the codebook from encoder outputs by MiniBatchKMeans.

        Public entry point, called from
        :meth:`dl_techniques.models.vision.vq_vae_rotation.model.VQVAERotationTrick.warm_start_codebook`
        and usable directly for a standalone quantizer. Call it eagerly, before
        training: it runs scikit-learn and assigns a variable, neither of
        which is graph-safe, so it must never run from inside ``call()``.
        ``batches`` are the encoder outputs the quantizer will see, already
        flattened over every non-channel axis. A head with fewer samples than
        clusters logs a warning and pads from the current codebook rather
        than failing.

        :param batches: Encoder outputs, ``(N, embedding_dim)`` or a sequence
            of such arrays.
        :type batches: Any
        :raises RuntimeError: If scikit-learn is not installed.
        :raises ValueError: If the layer is unbuilt, ``kmeans_init`` is False,
            or the feature dimension does not match ``embedding_dim``.
        """
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as exc:  # defensive — already checked in __init__
            raise RuntimeError(
                "kmeans_init=True requires scikit-learn."
            ) from exc

        if not self.built:
            raise ValueError(
                "warm_start_codebook requires a built layer: the codebook "
                "variable does not exist yet. Call the layer once, or build "
                "it explicitly, first."
            )
        if not self.kmeans_init:
            raise ValueError(
                "warm_start_codebook requires kmeans_init=True; with "
                "kmeans_init=False there is no kmeans_init_done variable to "
                "mark, and the codebook keeps its initializer's values."
            )

        if isinstance(batches, (list, tuple)):
            arrays = [np.asarray(keras.ops.convert_to_numpy(b)) for b in batches]
        else:
            arrays = [np.asarray(keras.ops.convert_to_numpy(batches))]

        flat = [a.reshape(-1, a.shape[-1]) for a in arrays]
        for a in flat:
            if a.shape[-1] != self.embedding_dim:
                raise ValueError(
                    f"warm_start_codebook expects vectors of width "
                    f"embedding_dim={self.embedding_dim}, got {a.shape[-1]}."
                )

        # all_batches: (N_total, H, D_h).
        all_batches = np.concatenate(
            [a.reshape(-1, self.num_heads, self.head_dim) for a in flat],
            axis=0,
        )
        new_codebook = np.zeros(
            (self.num_heads, self.num_embeddings, self.head_dim), dtype=np.float32
        )
        for h in range(self.num_heads):
            head_vectors = all_batches[:, h, :]
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
                # existing: (K, D_h), padded onto centroids below.
                existing = np.asarray(
                    keras.ops.convert_to_numpy(self.embeddings)
                )[h]
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
        logger.info(
            f"kmeans_init: codebook initialised from "
            f"{all_batches.shape[0]} samples across {self.num_heads} head(s)."
        )

    @property
    def is_codebook_warm_started(self) -> bool:
        """True once :meth:`warm_start_codebook` has run.

        Reads the ``kmeans_init_done`` variable eagerly. Never call this from
        inside ``call()``.
        """
        if not self.kmeans_init or self.kmeans_init_done is None:
            return False
        return float(keras.ops.convert_to_numpy(self.kmeans_init_done)) >= 0.5


    def _diversity_loss(self) -> keras.KerasTensor:
        """Penalise mean off-diagonal of unit-codebook gram matrix per head."""
        e = self.embeddings  # (H, K, D)
        norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(e), axis=-1, keepdims=True) + self.epsilon)
        unit = e / norm
        gram = keras.ops.einsum("hkd,hjd->hkj", unit, unit)
        eye = keras.ops.eye(self.num_embeddings)
        eye = keras.ops.expand_dims(eye, axis=0)
        off_diag = gram - eye
        loss = keras.ops.mean(keras.ops.square(off_diag))
        return loss

    def _orthogonal_loss(self) -> keras.KerasTensor:
        """SRIP-style ``||E E^T - I||_F^2`` summed across heads."""
        e = self.embeddings
        gram = keras.ops.einsum("hkd,hjd->hkj", e, e)
        eye = keras.ops.expand_dims(keras.ops.eye(self.num_embeddings), axis=0)
        diff = gram - eye
        return keras.ops.mean(keras.ops.sum(keras.ops.square(diff), axis=(-1, -2)))


    def get_codebook_indices(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Return discrete indices of nearest codebook entries per head.

        :param inputs: ``(B, ..., D)``.
        :return: ``(B, ..., num_heads)`` int. When ``num_heads==1`` the trailing
            head dim is squeezed for parity with ``VectorQuantizer`` (which
            returns ``(B, ...)``).
        """
        if not self.built:
            self.build(inputs.shape)

        input_shape = keras.ops.shape(inputs)
        spatial_shape = input_shape[:-1]

        flat = keras.ops.reshape(inputs, (-1, self.embedding_dim))
        flat_heads = keras.ops.reshape(flat, (-1, self.num_heads, self.head_dim))
        # indices: (N, H).
        indices, _ = self._lookup(flat_heads)

        if self.num_heads == 1:
            indices_out = keras.ops.reshape(indices[:, 0], spatial_shape)
            return indices_out

        # Reshape (N, H) back to (B, ..., H).
        spatial_shape_i32 = keras.ops.cast(spatial_shape, "int32")
        h_tensor = keras.ops.convert_to_tensor([self.num_heads], dtype="int32")
        out_shape = keras.ops.concatenate([spatial_shape_i32, h_tensor], axis=0)
        return keras.ops.reshape(indices, out_shape)

    def quantize_from_indices(
            self,
            indices: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Convert indices back to embedding vectors.

        :param indices: ``(B, ..., num_heads)`` int, or ``(B, ...)`` when
            ``num_heads==1`` (parity with ``VectorQuantizer``).
        """
        if not self.built:
            raise ValueError("Layer must be built before calling quantize_from_indices")

        idx_shape = keras.ops.shape(indices)

        if self.num_heads == 1:
            flat_indices = keras.ops.reshape(indices, (-1,))
            flat_indices = keras.ops.expand_dims(flat_indices, axis=-1)
            spatial_shape_i32 = keras.ops.cast(idx_shape, "int32")
        else:
            flat_indices = keras.ops.reshape(indices, (-1, self.num_heads))
            # Spatial shape is idx_shape[:-1] (the last axis is heads).
            spatial_shape_i32 = keras.ops.cast(idx_shape[:-1], "int32")

        # DECISION plan-2026-08-19T163559-499b6f0e/D-011 (see `_lookup`).
        encodings = keras.ops.one_hot(
            flat_indices, self.num_embeddings, dtype=self.compute_dtype
        )
        quantized = keras.ops.einsum("nhk,hkd->nhd", encodings, self.embeddings)
        flat_q = keras.ops.reshape(quantized, (-1, self.embedding_dim))

        d_tensor = keras.ops.convert_to_tensor([self.embedding_dim], dtype="int32")
        out_shape = keras.ops.concatenate([spatial_shape_i32, d_tensor], axis=0)
        return keras.ops.reshape(flat_q, out_shape)

