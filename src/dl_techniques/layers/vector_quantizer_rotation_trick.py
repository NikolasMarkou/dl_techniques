"""
Vector quantization with the Rotation Trick gradient estimator.

This layer embodies the principle of gradient reshaping at a discrete
bottleneck, a design paradigm that decouples the forward quantization
semantics from the backward gradient path. The core idea is to leave the
forward computation identical to standard vector quantization, a nearest
neighbour lookup in a learned codebook, while replacing the coarse
straight-through estimator with a transformation that carries the geometric
relationship between the encoder output and its assigned code back into the
encoder's gradient.

The motivation is a known deficiency of the straight-through estimator. Because
`argmin` has zero derivative almost everywhere, the standard formulation copies
the reconstruction gradient at `z_q` directly onto `z_e`:

`z_q = z_e + stop_gradient(e_k* - z_e)`

Every point inside a Voronoi cell therefore receives the same gradient
regardless of where it sits relative to its centroid. Directional and
curvature information about the quantization error is discarded, which
degrades codebook utilization and encoder conditioning.

The Rotation Trick instead treats the map from `z_e` to `z_q` as a rotation
composed with a rescaling, and applies that same linear operator to the
incoming gradient. Writing `x` for the encoder output and `q` for its assigned
code, the very-efficient Householder form used here is:

`u_x = x / ||x||`
`u_q = q / ||q||`
`w   = (u_x + u_q) / ||u_x + u_q||`
`R(x) = x - 2 (x . w) w + 2 (x . u_x) u_q`
`scale = ||q|| / ||x||`
`output = R(x) * scale`

The geometric anchors `u_x`, `u_q`, `w`, and by default `scale` are wrapped in
`stop_gradient`, which makes `R` a constant linear operator with respect to
backpropagation. Gradients therefore flow through `R(x)` as a fixed rotation of
the upstream gradient rather than as an identity, preserving the angular
relationship that the straight-through estimator collapses. Forward values are
unchanged: applying `R` and then `scale` to `x` reproduces `q` exactly.

Architecturally, a forward pass proceeds through five stages:
1.  The input is flattened across all non-channel dimensions and split into
    `num_heads` independent channel groups, giving `[N, H, D/H]` against a
    codebook of shape `[H, K, D/H]`. Multi-head factorization raises the
    effective vocabulary to `K^H` at linear cost in memory.
2.  A per-head nearest neighbour search selects one code per group, using
    either squared Euclidean distance or cosine similarity. In cosine mode the
    lookup is purely angular, so the input's magnitude affects neither which code
    is selected nor what is emitted: the quantized vector is the stored codebook
    row itself, as in euclidean mode. Codebook magnitudes are therefore trained by
    the codebook loss in both modes.
3.  The selected codes are combined with the encoder output by the chosen
    gradient transform: `'rotation'` for the full form above, `'reflection'`
    for the Householder reflection alone, `'no_grad_scale'` to let the scale
    factor remain differentiable, or `'ste'` to recover classical
    straight-through behaviour.
4.  Auxiliary objectives are accumulated. The codebook and commitment terms
    follow the original VQ-VAE formulation; optional diversity and orthogonal
    penalties act directly on the codebook gram matrix to discourage
    collinear or redundant entries.
5.  The result is reshaped back to the input geometry, so the layer is a
    shape-preserving drop-in at any point in a network.

Codebook maintenance is handled by three optional mechanisms, each addressing
a distinct failure mode of discrete bottlenecks. Exponential moving average
updates treat the codebook as an online k-means problem, tracking per-code
assignment counts `N` and assigned-vector sums `m` with decay `gamma` and
setting `e = m / (N + eps)`, which removes codebook adaptation from the
optimizer's learning rate and momentum state. Dead-code expiration counts
consecutive calls in which a code receives no assignments and reinitializes
entries past a threshold from vectors in the current batch, recovering capacity
lost to codebook collapse. A one-shot k-means warm start seeds the codebook
from accumulated encoder statistics, avoiding the large initial mismatch between
a randomly initialized codebook and the encoder distribution. It runs in
`warm_start_codebook`, eagerly and before training, NOT inside `call`: until
2026-08-15 it lived in `call`, where reading its own `kmeans_init_done` flag
meant `np.asarray` on a graph tensor -- which raises the moment anyone calls
`model.fit()` -- and where its accumulator was a plain Python list appended to
inside a traced function, so `kmeans_init_steps > 1` collected one trace's worth
of data rather than N batches. Every k-means test called the layer eagerly, the
one regime in which neither failure is visible.

This implementation is a strict superset of a standard vector quantizer.
Setting `gradient_mode='ste'` with `num_heads=1`, `distance_mode='euclidean'`,
and `use_ema=False` reproduces the classical layer's behaviour to within
floating point tolerance.

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

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VectorQuantizerRotationTrick(keras.layers.Layer):
    """Vector Quantizer with Rotation Trick gradient + multi-head codebook.

    A strict superset of ``VectorQuantizer``. Setting ``gradient_mode='ste'``
    and ``num_heads=1, distance_mode='euclidean', use_ema=False`` recovers the
    existing layer's behaviour bit-equivalently (atol<=1e-6).

    **Architecture Overview:**

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
        :class:`~dl_techniques.models.vq_vae_rotation.model.VQVAERotationTrick`
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
            # DECISION plan-2026-08-18T140459-7991552f/D-012
            # This accumulator MUST start at ZERO, not at `self.initializer`.
            # It is the numerator of an EMA that `_update_ema` debiases by
            # `1 - decay**t`; a non-zero start is a bias the debias step
            # AMPLIFIES by `decay**t / (1 - decay**t)` = 99x at t=1. MEASURED
            # on the sibling `vector_quantizer.py` (same defect, same fixture):
            # `max|codebook|` after 5 epochs was 47283 with this initializer
            # kept and the debias added, versus 4521 with neither correction
            # and 0.264 with both. See decisions.md D-012.
            self.ema_embeddings = self.add_weight(
                name="ema_embeddings",
                shape=(self.num_heads, self.num_embeddings, self.head_dim),
                initializer="zeros",
                trainable=False,
            )

            # DECISION plan-2026-08-18T140459-7991552f/D-012
            # EMA step counter, read ONLY by `_update_ema`'s bias correction.
            # Do NOT delete it as an unused scalar and do NOT give it
            # `dtype="int32"`: TF places an int32 variable on the CPU and the
            # resulting CPU/GPU split raises `Trying to access resource
            # .../ema_step ... from device GPU:0` inside the jit-compiled train
            # function (MEASURED on the sibling layer -- `fit()` died).
            # It changes the `use_ema=True` weight set from 3 to 4.
            self.ema_step = self.add_weight(
                name="ema_step",
                shape=(),
                initializer="zeros",
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

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        input_shape = keras.ops.shape(inputs)

        # Flatten everything but channels: (..., D) -> (N, D)
        flat_inputs = keras.ops.reshape(inputs, (-1, self.embedding_dim))

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-040
        # NO k-means work here. `call()` is traced: it ran
        # `float(keras.ops.convert_to_numpy(self.kmeans_init_done))`, which on
        # the TF backend is `np.asarray(graph_tensor)` and RAISES under
        # `model.fit()`, and it appended to a plain Python list, which
        # accumulates once per TRACE rather than once per batch, so
        # `kmeans_init_steps > 1` silently saw one batch. The warm start now
        # lives in `warm_start_codebook`, called eagerly before training.
        # Do NOT move it back inside `call` in any form. See decisions.md D-040.

        # Reshape to (N, H, head_dim)
        flat_heads = keras.ops.reshape(flat_inputs, (-1, self.num_heads, self.head_dim))

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
            x_sq = keras.ops.sum(keras.ops.square(flat_heads), axis=-1, keepdims=True)  # (N,H,1)
            e_sq = keras.ops.sum(keras.ops.square(codebook), axis=-1)  # (H,K)
            e_sq = keras.ops.expand_dims(e_sq, axis=0)  # (1,H,K)
            # x . e: einsum over head_dim
            # flat_heads (N,H,D) x codebook (H,K,D) -> (N,H,K)
            xe = keras.ops.einsum("nhd,hkd->nhk", flat_heads, codebook)
            distances = x_sq + e_sq - 2.0 * xe
            indices = keras.ops.argmin(distances, axis=-1)  # (N,H)
        else:  # cosine
            # L2-normalise both
            x_norm = keras.ops.sqrt(
                keras.ops.sum(keras.ops.square(flat_heads), axis=-1, keepdims=True) + self.epsilon
            )
            unit_x = flat_heads / x_norm  # (N,H,D)
            e_norm = keras.ops.sqrt(
                keras.ops.sum(keras.ops.square(codebook), axis=-1, keepdims=True) + self.epsilon
            )
            unit_e = codebook / e_norm  # (H,K,D)
            sim = keras.ops.einsum("nhd,hkd->nhk", unit_x, unit_e)
            indices = keras.ops.argmax(sim, axis=-1)  # (N,H)

        # Gather quantized vectors per head.
        # one_hot indices to (N,H,K), then matmul against codebook (H,K,D)
        encodings = keras.ops.one_hot(indices, self.num_embeddings)  # (N,H,K)
        # quantized = sum_k encodings * codebook[h,k] -> (N,H,D)
        quantized = keras.ops.einsum("nhk,hkd->nhd", encodings, codebook)

        # DECISION plan-2026-08-17T183311-79c63e38/D-030: cosine mode returns the RAW
        # codebook row, exactly like euclidean mode. This branch used to "restore
        # magnitude" by scaling the row by `x_mag / q_mag`, and it must NOT be put
        # back. `quantize_from_indices` — the only other producer of a quantized
        # vector, and the one `models/vq_vae_rotation`'s
        # `encode_to_indices -> quantize_from_indices -> decode` pair uses — returns
        # the raw row in every mode and CANNOT reproduce the rescale: an index has
        # already discarded x_mag by design in cosine mode (that is what cosine
        # similarity means), so carrying it would require changing
        # `encode_to_indices`' public return signature. So the rescale, not the raw
        # row, was the outlier. It also made the "discrete" bottleneck leak a
        # continuous per-token magnitude channel, leaving codebook MAGNITUDES
        # untrained (`||sg[x] - (||x||/||e||)e||^2` is direction-only). This is the
        # same defect class as the reflection-sign bug fixed in
        # `_apply_gradient_transform` below, for magnitude instead of sign, and the
        # invariant documented there is the one it violated. See decisions.md D-030.

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
            return x + keras.ops.stop_gradient(q - x)

        # Promote to fp32 for norms.
        x_dtype = x.dtype
        x32 = keras.ops.cast(x, "float32")
        q32 = keras.ops.cast(q, "float32")

        eps = self.epsilon
        eps_sq = eps * eps

        # DECISION plan-2026-08-18T140459-7991552f/D-025: the three norms below are
        # FLOORED (`sqrt(max(sum(x^2), eps^2))` == `max(||x||, eps)`), NOT
        # eps-regularised. Do NOT put `sqrt(sum(x^2) + eps)` back. That form is not a
        # floor: it inflates every norm, and since `scale_eff = q_norm / x_norm`
        # multiplies the output, the inflation does not cancel -- `call()` emits
        # `(1 + O(eps/||x||^2)) * q` instead of `q`, so the "discrete" bottleneck
        # leaks a CONTINUOUS per-token magnitude channel and disagrees with
        # `encode_to_indices -> quantize_from_indices -> decode`, which returns the
        # raw codebook row. The leak grows as the encoder's scale shrinks, which is
        # exactly the regime a fresh `Conv2D(embedding_dim, 1)` head produces.
        # MEASURED at HEAD (num_embeddings=16, embedding_dim=8, seeded N(0,1) input
        # scaled by s), max|call() - quantize_from_indices()|, all three rotation
        # modes vs the exact `ste` path:
        #     s=1.00  (||x||^2~8.2e+00): 5.35e-05  (rel 1.1e-03)   ste 2.98e-08
        #     s=0.10  (||x||^2~8.2e-02): 6.88e-05  (rel 1.4e-03)   ste 7.45e-09
        #     s=0.01  (||x||^2~8.2e-04): 1.92e-03  (rel 4.0e-02)   ste 1.86e-09
        # i.e. a 4% magnitude leak at the small-norm end. With the floor, the same
        # three measurements are 2.61e-08 / 2.61e-08 / 1.86e-08 -- the float32
        # arithmetic floor, indistinguishable from the exact `ste` path and no longer
        # scale dependent. The floor is written on the SQUARED side deliberately: below the
        # floor the sqrt argument is a constant, so the `no_grad_scale` mode (the one
        # branch where gradient flows through these norms) cannot see `d/dx sqrt(x)`
        # blow up at an exactly-zero row. See decisions.md D-025.
        #
        # Per the paper: the geometric anchors (unit_x, unit_q, w, and scale unless
        # 'no_grad_scale') are computed with stop_gradient so that the rotation
        # matrix R becomes a *constant* w.r.t. backprop. The gradient w.r.t. x
        # then flows through R @ x as a constant linear transform — preserving
        # the curvature/direction information that pure STE discards.
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


        # x · w and x · unit_x — gradient WILL flow through x here (the whole point).
        x_dot_w = keras.ops.sum(x32 * w_sg, axis=-1, keepdims=True)
        x_dot_ux = keras.ops.sum(x32 * unit_x_sg, axis=-1, keepdims=True)

        if mode == "rotation":
            rotated = x32 - 2.0 * x_dot_w * w_sg + 2.0 * x_dot_ux * unit_q_sg
        elif mode == "reflection":
            # A Householder reflection about the hyperplane with normal
            # (u + v) maps u -> -v, NOT u -> +v. This branch used to return
            # that reflection unnegated, so its forward output was exactly -q
            # -- verified numerically -- contradicting the module's contract
            # that the forward pass emits the codebook vector, and making
            # call() disagree in SIGN with
            # encode_to_indices -> quantize_from_indices -> decode.
            #
            # Negating recovers u -> +v while keeping the (u + v) normal.
            # The alternative -- reflecting about (u - v), which maps u -> +v
            # directly -- is mathematically equivalent but numerically far
            # worse HERE: (u - v) vanishes exactly when x is close to its
            # codebook vector, which is the common case, and normalizing a
            # vanishing normal loses precision (measured ~1e-4 absolute error
            # on unit-scale codebook entries, versus ~1e-7 for this form).
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
        encodings = keras.ops.one_hot(indices, self.num_embeddings)  # (N,H,K)
        # cluster_size: per (H,K) -> sum over N
        cluster_size = keras.ops.sum(encodings, axis=0)  # (H,K)
        # embed sums: (H,K,D) = sum_n encodings[n,h,k] * flat_heads[n,h,d]
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

        # DECISION plan-2026-08-18T140459-7991552f/D-012
        # Do NOT "simplify" the block below back to
        #     self.ema_embeddings / (self.ema_cluster_size + self.epsilon)
        # It looks equivalent and is not. Two corrections, both load-bearing:
        #
        # (1) BIAS CORRECTION. Both accumulators start at zero, so at step t
        #     they carry a factor `1 - decay**t` (0.01 at t=1). The counts and
        #     the sums are on a common scale only after dividing BOTH by it;
        #     without it a code that received no assignments evaluates to
        #     `numerator / 1e-5`, a step-1 blow-up of ~1e5 that collapses every
        #     input onto a single code. No amount of extra training undoes it.
        # (2) LAPLACE SMOOTHING. Bare `+ epsilon` is not a stabiliser but an
        #     unbounded gain: as the count -> 0 the ratio -> numerator / 1e-5.
        #     Smoothing the counts toward a uniform prior while preserving
        #     their total keeps every denominator O(N/K).
        #
        # AXIS: this quantizer is PER-HEAD, so `debiased_cluster_size` is
        # `(H, K)` and the Laplace total is summed over the CODEBOOK axis K
        # with `keepdims=True` -- shape `(H, 1)`, one total per head. The
        # sibling `vector_quantizer.py` has a 1-D `(K,)` count and a scalar
        # total; this is the one place the two copies of this fix differ.
        #
        # Honest note on how much that choice matters, MEASURED rather than
        # asserted: `cluster_size = sum(one_hot(indices), axis=0)` sums over N,
        # so `sum_k cluster_size[h, k] == N` for EVERY head, identically, at
        # every step. The per-head totals are therefore always equal and
        # pooling over H would scale numerator and denominator by the same H;
        # the two answers differ by ~9e-6 (the eps floor alone) at H=3, K=8,
        # N=64. K is kept because it is the axis that stays correct if the
        # counts ever become per-head unequal, NOT because H is observably
        # wrong today. Do not write a test claiming to discriminate them.
        #
        # MEASURED (sibling layer, same fixture, 5 epochs): unique codes 1 and
        # max|codebook| 4521 before, 18 and 0.264 after. See decisions.md D-012.
        self.ema_step.assign_add(1.0)
        bias_correction = 1.0 - keras.ops.power(
            keras.ops.cast(self.ema_decay, self.ema_step.dtype), self.ema_step
        )

        debiased_cluster_size = self.ema_cluster_size / bias_correction
        debiased_embeddings = self.ema_embeddings / bias_correction

        # Laplace smoothing of the counts; per-head total mass preserved.
        total_count = keras.ops.sum(
            debiased_cluster_size, axis=-1, keepdims=True
        )  # (H, 1)
        smoothed_cluster_size = (
                (debiased_cluster_size + self.epsilon)
                / (total_count + self.num_embeddings * self.epsilon)
                * total_count
        )

        normalised = debiased_embeddings / keras.ops.expand_dims(
            smoothed_cluster_size, axis=-1
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
        encodings = keras.ops.one_hot(indices, self.num_embeddings)  # (N,H,K)
        used_this_call = keras.ops.cast(keras.ops.sum(encodings, axis=0) > 0, "float32")  # (H,K)

        # Increment unused counter for codes not used; reset for codes used.
        new_unused = (1.0 - used_this_call) * (self.dead_code_unused + 1.0)
        self.dead_code_unused.assign(new_unused)

        # Find dead codes: unused > threshold.
        # We then replace each dead code with a random encoder vector from this batch
        # (per head). For correctness in pure Keras keras.ops we sample via shuffle.
        dead_mask = keras.ops.cast(
            self.dead_code_unused > float(self.dead_code_threshold), "float32"
        )  # (H, K)

        # Sample replacement vectors per head from flat_heads (N, H, head_dim).
        n = keras.ops.shape(flat_heads)[0]
        # Random indices in [0, N), shape (H, K) — pick one batch vector per dead slot.
        rand_uniform = keras.random.uniform(
            shape=(self.num_heads, self.num_embeddings),
            minval=0.0,
            maxval=1.0,
        )
        rand_idx = keras.ops.cast(rand_uniform * keras.ops.cast(n, "float32"), "int32")
        rand_idx = keras.ops.clip(rand_idx, 0, n - 1)  # (H, K)

        # Gather: replacements[h, k, :] = flat_heads[rand_idx[h, k], h, :]
        # flat_heads is (N, H, D); we want (H, K, D).
        # Build with take + per-head indexing via vectorisation.
        # take along axis=0 with indices (H,K) -> result (H,K,H,D); we need diagonal in H.
        # Simpler: transpose flat_heads to (H, N, D) then gather along axis=1 per head.
        heads_first = keras.ops.transpose(flat_heads, (1, 0, 2))  # (H, N, D)
        # gather indices rand_idx (H, K) along axis=1.
        replacements = keras.ops.take_along_axis(
            heads_first,
            keras.ops.expand_dims(rand_idx, axis=-1),  # (H,K,1)
            axis=1,
        )  # (H, K, D)  -- keras.ops.take_along_axis broadcasts the last dim

        # Blend: new_codebook = dead_mask * replacements + (1 - dead_mask) * embeddings
        dead_mask_exp = keras.ops.expand_dims(dead_mask, axis=-1)  # (H,K,1)
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

    def warm_start_codebook(self, batches: Any) -> None:
        """Initialise the codebook from encoder outputs by MiniBatchKMeans.

        Interface contract (public entry point, called from
        :meth:`dl_techniques.models.vq_vae_rotation.model.VQVAERotationTrick.warm_start_codebook`
        and usable directly for a standalone quantizer):

        - **Parameters**: ``batches`` — either one array-like of shape
          ``(N, embedding_dim)`` or a sequence of such arrays, which are
          concatenated. These are the ENCODER OUTPUTS the quantizer will see,
          already flattened over every non-channel axis.
        - **Returns**: ``None``. It assigns ``self.embeddings`` and sets
          ``kmeans_init_done`` to 1.0.
        - **Failure mode**: ``RuntimeError`` if scikit-learn is missing;
          ``ValueError`` if the layer is not built, if ``kmeans_init`` is
          ``False`` (there is no ``kmeans_init_done`` variable in that case),
          or if the last axis is not ``embedding_dim``. A head with fewer
          samples than clusters logs a warning and pads from the current
          codebook rather than failing.

        **Call this EAGERLY, before training.** It runs scikit-learn and
        assigns a variable, neither of which is graph-safe; it must never be
        reached from inside ``call()``.

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

        all_batches = np.concatenate(
            [a.reshape(-1, self.num_heads, self.head_dim) for a in flat],
            axis=0,
        )  # (N_total, H, D_h)
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
                    keras.ops.convert_to_numpy(self.embeddings)
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
        logger.info(
            f"kmeans_init: codebook initialised from "
            f"{all_batches.shape[0]} samples across {self.num_heads} head(s)."
        )

    @property
    def is_codebook_warm_started(self) -> bool:
        """True once :meth:`warm_start_codebook` has run.

        Reads the ``kmeans_init_done`` variable EAGERLY. Never call this from
        inside ``call()`` -- that is the graph read this defect was.
        """
        if not self.kmeans_init or self.kmeans_init_done is None:
            return False
        return float(keras.ops.convert_to_numpy(self.kmeans_init_done)) >= 0.5

    # ------------------------------------------------------------------
    # aux losses
    # ------------------------------------------------------------------

    def _diversity_loss(self) -> keras.KerasTensor:
        """Penalise mean off-diagonal of unit-codebook gram matrix per head."""
        e = self.embeddings  # (H, K, D)
        norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(e), axis=-1, keepdims=True) + self.epsilon)
        unit = e / norm  # (H, K, D)
        gram = keras.ops.einsum("hkd,hjd->hkj", unit, unit)  # (H, K, K)
        eye = keras.ops.eye(self.num_embeddings)
        eye = keras.ops.expand_dims(eye, axis=0)  # (1, K, K)
        off_diag = gram - eye
        loss = keras.ops.mean(keras.ops.square(off_diag))
        return loss

    def _orthogonal_loss(self) -> keras.KerasTensor:
        """SRIP-style ``||E E^T - I||_F^2`` summed across heads."""
        e = self.embeddings  # (H, K, D)
        gram = keras.ops.einsum("hkd,hjd->hkj", e, e)
        eye = keras.ops.expand_dims(keras.ops.eye(self.num_embeddings), axis=0)
        diff = gram - eye
        return keras.ops.mean(keras.ops.sum(keras.ops.square(diff), axis=(-1, -2)))

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

        input_shape = keras.ops.shape(inputs)
        spatial_shape = input_shape[:-1]

        flat = keras.ops.reshape(inputs, (-1, self.embedding_dim))
        flat_heads = keras.ops.reshape(flat, (-1, self.num_heads, self.head_dim))
        indices, _ = self._lookup(flat_heads)  # (N, H)

        if self.num_heads == 1:
            indices_out = keras.ops.reshape(indices[:, 0], spatial_shape)
            return indices_out

        # Reshape (N, H) back to (B, ..., H).
        spatial_shape_i32 = keras.ops.cast(spatial_shape, "int32")
        h_tensor = keras.ops.convert_to_tensor([self.num_heads], dtype="int32")
        out_shape = keras.ops.concatenate([spatial_shape_i32, h_tensor], axis=0)
        return keras.ops.reshape(indices, out_shape)

    def quantize_from_indices(
            self, indices: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Convert indices back to embedding vectors.

        :param indices: ``(B, ..., num_heads)`` int, or ``(B, ...)`` when
            ``num_heads==1`` (parity with ``VectorQuantizer``).
        """
        if not self.built:
            raise ValueError("Layer must be built before calling quantize_from_indices")

        idx_shape = keras.ops.shape(indices)

        if self.num_heads == 1:
            flat_indices = keras.ops.reshape(indices, (-1,))  # (N,)
            flat_indices = keras.ops.expand_dims(flat_indices, axis=-1)  # (N, 1)
            spatial_shape_i32 = keras.ops.cast(idx_shape, "int32")
        else:
            flat_indices = keras.ops.reshape(indices, (-1, self.num_heads))  # (N, H)
            # Spatial shape is idx_shape[:-1] (the last axis is heads).
            spatial_shape_i32 = keras.ops.cast(idx_shape[:-1], "int32")

        encodings = keras.ops.one_hot(flat_indices, self.num_embeddings)  # (N, H, K)
        quantized = keras.ops.einsum("nhk,hkd->nhd", encodings, self.embeddings)  # (N,H,D)
        flat_q = keras.ops.reshape(quantized, (-1, self.embedding_dim))  # (N, D)

        d_tensor = keras.ops.convert_to_tensor([self.embedding_dim], dtype="int32")
        out_shape = keras.ops.concatenate([spatial_shape_i32, d_tensor], axis=0)
        return keras.ops.reshape(flat_q, out_shape)
