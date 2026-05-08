"""Memory banks for dual-tap memory-augmented transformer.

Two banks exposed:

- :class:`LongTermMemoryBank` — persistent ``(K_lt, V_lt)`` of fixed slot
  count ``S_lt``. Initialized via ``RandomNormal`` and replaced by an
  offline ``MiniBatchKMeans`` seeding at the start of Phase 2 via
  :meth:`assign_keys_from_kmeans`.

- :class:`WorkingMemoryBank` — stateless projector that maps a pre-block
  hidden state ``X_W (B, T, D)`` into ``(K_wm, V_wm)`` of shapes
  ``(B, T, d_k)`` and ``(B, T, d_v)``. Per blueprint, no bias on the K
  projection (positional-encoding-free key direction); bias allowed on V.

Both banks are decorated with ``@keras.saving.register_keras_serializable``
and provide full ``get_config()`` round-trip support.
"""

from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np
from keras import ops

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LongTermMemoryBank(keras.layers.Layer):
    """Persistent long-term memory: ``S_lt`` slots of ``(K_lt, V_lt)``.

    :param s_lt: Number of long-term memory slots.
    :param d_k: Key dimensionality.
    :param d_v: Value dimensionality.
    :param initializer_range: Stddev for ``RandomNormal`` init (placeholder
        for ``K_lt``; replaced via :meth:`assign_keys_from_kmeans` at the
        Phase 1->2 boundary).
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.

    Trainable variables (created in :meth:`build`):

    - ``memory_K_lt`` of shape ``(S_lt, d_k)``.
    - ``memory_V_lt`` of shape ``(S_lt, d_v)``.

    The ``memory_`` name prefix is load-bearing — the custom ``train_step``
    in :class:`WaveFieldMemoryLLM` splits gradients between the backbone
    and memory optimizers based on this prefix.
    """

    def __init__(
        self,
        s_lt: int,
        d_k: int,
        d_v: int,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if s_lt <= 0:
            raise ValueError(f"s_lt must be positive, got {s_lt}")
        if d_k <= 0:
            raise ValueError(f"d_k must be positive, got {d_k}")
        if d_v <= 0:
            raise ValueError(f"d_v must be positive, got {d_v}")
        if d_k == d_v:
            raise ValueError(
                f"d_k must differ from d_v (blueprint constraint); "
                f"got d_k=d_v={d_k}"
            )

        self.s_lt = s_lt
        self.d_k = d_k
        self.d_v = d_v
        self.initializer_range = initializer_range

        self._k_initializer = keras.initializers.RandomNormal(
            stddev=initializer_range,
        )
        self._v_initializer = keras.initializers.RandomNormal(
            stddev=initializer_range,
        )

    def build(self, input_shape: Optional[Tuple[Optional[int], ...]] = None) -> None:
        # Names carry the `memory_` prefix so the custom train_step can
        # route gradients to the memory optimizer.
        self.K_lt = self.add_weight(
            name="memory_K_lt",
            shape=(self.s_lt, self.d_k),
            initializer=self._k_initializer,
            trainable=True,
        )
        self.V_lt = self.add_weight(
            name="memory_V_lt",
            shape=(self.s_lt, self.d_v),
            initializer=self._v_initializer,
            trainable=True,
        )
        super().build(input_shape if input_shape is not None else ())

    def call(self, inputs: Optional[Any] = None) -> Tuple[Any, Any]:
        """Return ``(K_lt, V_lt)`` tensors. Input is ignored.

        Banks are stateless w.r.t. inputs — they hold variables, not
        per-batch state. The signature accepts ``inputs`` for Keras
        layer-call compatibility but is not used.
        """
        return self.K_lt, self.V_lt

    def assign_keys_from_kmeans(self, centroids: np.ndarray) -> None:
        """Replace ``K_lt`` with offline KMeans centroids.

        :param centroids: Numpy array of shape ``(s_lt, d_k)``. Raises if
            shape mismatch.
        """
        centroids = np.asarray(centroids, dtype=np.float32)
        if centroids.shape != (self.s_lt, self.d_k):
            raise ValueError(
                f"centroids shape {centroids.shape} does not match "
                f"(s_lt={self.s_lt}, d_k={self.d_k})"
            )
        self.K_lt.assign(centroids)
        logger.info(
            f"LongTermMemoryBank: K_lt seeded from KMeans "
            f"(shape={centroids.shape})"
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "s_lt": self.s_lt,
            "d_k": self.d_k,
            "d_v": self.d_v,
            "initializer_range": self.initializer_range,
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WorkingMemoryBank(keras.layers.Layer):
    """Stateless projector that maps ``X_W (B, T, D)`` to ``(K_wm, V_wm)``.

    Two ``Dense`` projections:

    - ``W_K`` projects to ``d_k`` (no bias — keys are positional-encoding-
      free direction vectors per blueprint).
    - ``W_V`` projects to ``d_v`` (with bias).

    :param d_k: Key dimensionality.
    :param d_v: Value dimensionality.
    :param embed_dim: Hidden-state dimensionality of ``X_W``.
    :param initializer_range: Stddev for ``TruncatedNormal`` init.
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.
    """

    def __init__(
        self,
        d_k: int,
        d_v: int,
        embed_dim: int,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if d_k <= 0:
            raise ValueError(f"d_k must be positive, got {d_k}")
        if d_v <= 0:
            raise ValueError(f"d_v must be positive, got {d_v}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if d_k == d_v:
            raise ValueError(
                f"d_k must differ from d_v; got d_k=d_v={d_k}"
            )
        if d_v >= embed_dim:
            raise ValueError(
                f"d_v ({d_v}) must be < embed_dim ({embed_dim}) "
                f"(bottleneck constraint)"
            )

        self.d_k = d_k
        self.d_v = d_v
        self.embed_dim = embed_dim
        self.initializer_range = initializer_range

        kernel_init = keras.initializers.TruncatedNormal(
            stddev=initializer_range,
        )

        # Variable name prefixes carry `memory_` so the custom train_step
        # can route gradients to the memory optimizer.
        self.W_K = keras.layers.Dense(
            d_k,
            use_bias=False,
            kernel_initializer=kernel_init,
            name="memory_wm_W_K",
        )
        self.W_V = keras.layers.Dense(
            d_v,
            use_bias=True,
            kernel_initializer=kernel_init,
            name="memory_wm_W_V",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # Eagerly build so weights are created at construction time.
        self.W_K.build(input_shape)
        self.W_V.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        x_w: Any,
        training: Optional[bool] = None,
    ) -> Tuple[Any, Any]:
        """Project ``X_W`` to ``(K_wm, V_wm)``.

        :param x_w: Hidden-state tensor of shape ``(B, T, D)``.
        :param training: Forwarded for parity (no dropout here).
        :returns: ``(K_wm (B, T, d_k), V_wm (B, T, d_v))``.
        """
        del training  # unused
        k_wm = self.W_K(x_w)
        v_wm = self.W_V(x_w)
        return k_wm, v_wm

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        b, t = input_shape[0], input_shape[1]
        return ((b, t, self.d_k), (b, t, self.d_v))

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "d_k": self.d_k,
            "d_v": self.d_v,
            "embed_dim": self.embed_dim,
            "initializer_range": self.initializer_range,
        })
        return config
