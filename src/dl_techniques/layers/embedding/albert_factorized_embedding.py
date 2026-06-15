"""ALBERT-style factorized token embedding.

A parameter-efficient drop-in replacement for ``keras.layers.Embedding``
that decomposes the embedding lookup into two stages:

    embed(i) = W @ E[i]

where ``E: [vocab_size, bottleneck_dim]`` is a small inner embedding
table and ``W: [bottleneck_dim, output_dim]`` is a shared projection.

Parameter count:
    - Standard: ``vocab_size * output_dim``
    - Factorized: ``vocab_size * bottleneck_dim + bottleneck_dim * output_dim``

Compression ratio approaches ``output_dim / bottleneck_dim`` as
``vocab_size`` grows, since the ``bottleneck_dim * output_dim``
projection becomes negligible. For ``vocab=50K, D=768, k=128``:
~6.5M params vs ~38.6M for standard ``Embedding`` (~6x smaller).

Compared to :class:`HierarchicalCodebookEmbedding`:
    - **ALBERT**: every token's embedding can independently occupy any
      direction in the k-dim subspace projected to D. Full-rank per-token
      manifold. Best when ``output_dim`` is large and you want maximum
      per-token expressivity.
    - **HCE**: embeddings live on the Minkowski sum of K finite codebook
      sets. Restricted manifold but ``O(K * 2^chunk_bits * D)`` params,
      orders of magnitude smaller. Best when ``output_dim`` is small or
      when extreme parameter compression is required.

References:
    - Lan, Z., et al. (2019). "ALBERT: A Lite BERT for Self-supervised
      Learning of Language Representations". arXiv:1909.11942.
"""

import keras
from keras import initializers, regularizers
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AlbertFactorizedEmbedding(keras.layers.Layer):
    """Factorized token embedding via a learnable bottleneck projection.

    See module docstring for full design rationale, parameter-count
    derivation, and comparison with :class:`HierarchicalCodebookEmbedding`.

    :param vocab_size: Number of distinct token IDs to support.
        Token IDs in ``[0, vocab_size)`` produce well-defined embeddings.
    :param bottleneck_dim: Inner embedding dimensionality ``k``. Must be
        positive. The compression vs. a standard ``Embedding(vocab,
        output_dim)`` is approximately ``output_dim / bottleneck_dim``
        for large vocab.
    :param output_dim: Final embedding dimensionality ``D``.
    :param embeddings_initializer: Initializer for both the inner
        embedding table and the projection kernel. Default ``"uniform"``
        (matches ``keras.layers.Embedding``).
    :param embeddings_regularizer: Optional regularizer applied to the
        inner embedding table.
    :param projection_regularizer: Optional regularizer applied to the
        projection kernel.

    :raises ValueError: If ``vocab_size`` is not greater than 1, or if
        ``bottleneck_dim``/``output_dim`` are non-positive.

    Example::

        embed = AlbertFactorizedEmbedding(
            vocab_size=50_261, bottleneck_dim=128, output_dim=768,
        )
        ids = keras.random.uniform((4, 32), 0, 50261, dtype="int32")
        x = embed(ids)              # shape: (4, 32, 768)
    """

    def __init__(
        self,
        vocab_size: int,
        bottleneck_dim: int,
        output_dim: int,
        embeddings_initializer: Union[str, initializers.Initializer] = "uniform",
        embeddings_regularizer: Optional[
            Union[str, regularizers.Regularizer]
        ] = None,
        projection_regularizer: Optional[
            Union[str, regularizers.Regularizer]
        ] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if vocab_size <= 1:
            raise ValueError(f"vocab_size must be > 1, got {vocab_size}")
        if bottleneck_dim <= 0:
            raise ValueError(
                f"bottleneck_dim must be positive, got {bottleneck_dim}"
            )
        if output_dim <= 0:
            raise ValueError(
                f"output_dim must be positive, got {output_dim}"
            )

        self.vocab_size = vocab_size
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.projection_regularizer = regularizers.get(projection_regularizer)

        self.inner_embedding = keras.layers.Embedding(
            vocab_size,
            bottleneck_dim,
            embeddings_initializer=self.embeddings_initializer,
            embeddings_regularizer=self.embeddings_regularizer,
            name="inner",
        )
        self.proj = keras.layers.Dense(
            output_dim,
            use_bias=False,
            kernel_initializer=self.embeddings_initializer,
            kernel_regularizer=self.projection_regularizer,
            name="proj",
        )

        n_params = vocab_size * bottleneck_dim + bottleneck_dim * output_dim
        n_dense = vocab_size * output_dim
        logger.info(
            f"AlbertFactorizedEmbedding(vocab={vocab_size}, "
            f"k={bottleneck_dim}, D={output_dim}): {n_params:,} params "
            f"(~{n_dense / max(1, n_params):.1f}x smaller than "
            f"Embedding({vocab_size},{output_dim})={n_dense:,} params)"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if self.built:
            return

        # Explicitly build sub-layers so save/load round-trip works without
        # Keras complaining about unbuilt internal state.
        self.inner_embedding.build(input_shape)
        bottleneck_shape = tuple(input_shape) + (self.bottleneck_dim,)
        self.proj.build(bottleneck_shape)
        super().build(input_shape)

    def call(
        self, inputs: keras.KerasTensor, training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        return self.proj(self.inner_embedding(inputs))

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        return tuple(input_shape) + (self.output_dim,)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "bottleneck_dim": self.bottleneck_dim,
            "output_dim": self.output_dim,
            "embeddings_initializer": initializers.serialize(
                self.embeddings_initializer,
            ),
            "embeddings_regularizer": regularizers.serialize(
                self.embeddings_regularizer,
            ),
            "projection_regularizer": regularizers.serialize(
                self.projection_regularizer,
            ),
        })
        return config

    @classmethod
    def from_config(
        cls, config: Dict[str, Any],
    ) -> "AlbertFactorizedEmbedding":
        if config.get("embeddings_initializer") and isinstance(
            config["embeddings_initializer"], dict,
        ):
            config["embeddings_initializer"] = initializers.deserialize(
                config["embeddings_initializer"],
            )
        for key in ("embeddings_regularizer", "projection_regularizer"):
            if config.get(key) and isinstance(config[key], dict):
                config[key] = regularizers.deserialize(config[key])
        return cls(**config)

# ---------------------------------------------------------------------
