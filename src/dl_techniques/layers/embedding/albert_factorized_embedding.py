"""
ALBERT-style factorized token embedding.

A parameter-efficient replacement for ``keras.layers.Embedding`` that splits one
big lookup table into a small table plus a shared projection. A token id is
looked up in an inner table of width ``bottleneck_dim``, and that narrow vector
is then projected up to ``output_dim`` by a single kernel shared by the whole
vocabulary:

    embed(i) = E[i] @ W

where ``E`` has shape ``(vocab_size, bottleneck_dim)`` and ``W`` has shape
``(bottleneck_dim, output_dim)``. The projection is a bias-free ``Dense``, so
``embed(i)`` is a linear image of ``E[i]``; the composition is what the ALBERT
paper calls a factorized embedding parameterization.

Parameter count:
    - Standard: ``vocab_size * output_dim``
    - Factorized: ``vocab_size * bottleneck_dim + bottleneck_dim * output_dim``

    The compression ratio approaches ``output_dim / bottleneck_dim`` as
    ``vocab_size`` grows, because the ``bottleneck_dim * output_dim`` projection
    becomes negligible next to the table. For ``vocab=50K, D=768, k=128`` that is
    about 6.5M parameters against about 38.6M for a standard ``Embedding``,
    roughly a 6x reduction.

Compared to :class:`HierarchicalCodebookEmbedding`:
    - ALBERT: each token's embedding can independently occupy any direction of
      the ``k``-dimensional subspace that ``W`` maps into ``D`` dimensions --
      a full-rank per-token manifold. Best when ``output_dim`` is large and
      per-token expressivity matters.
    - HCE: embeddings live on the Minkowski sum of ``K`` finite codebook sets.
      The manifold is restricted, but the cost is ``O(K * 2^chunk_bits * D)``
      parameters, orders of magnitude smaller. Best when ``output_dim`` is small
      or when extreme compression is required.

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

    Looks a token id up in a narrow inner table of width ``bottleneck_dim`` and
    projects the result to ``output_dim`` with a single bias-free kernel shared
    across the vocabulary. Both projections are learnable; the intermediate
    width is what buys the parameter reduction, and it is the only thing that
    distinguishes this layer from a plain ``Embedding(vocab_size, output_dim)``
    of the same input/output signature. See the module docstring for the
    parameter-count derivation and the comparison with
    :class:`HierarchicalCodebookEmbedding`.

    **Architecture Overview:**

    .. code-block:: text

               input ids  (batch, ...)  int
                             │
                             ▼
        ┌──────────────────────────────────────────┐
        │ inner_embedding: Embedding               │
        │   table E  (vocab_size, bottleneck_dim)  │
        └────────────────────┬─────────────────────┘
                             │   (batch, ..., bottleneck_dim)
                             ▼
        ┌──────────────────────────────────────────┐
        │ proj: Dense(output_dim, use_bias=False)  │
        │   kernel W  (bottleneck_dim, output_dim) │
        └────────────────────┬─────────────────────┘
                             │
                             ▼
             output  (batch, ..., output_dim)

    :param vocab_size: Number of distinct token IDs to support. Token IDs in
        ``[0, vocab_size)`` produce well-defined embeddings.
    :type vocab_size: int
    :param bottleneck_dim: Inner embedding dimensionality ``k``. Must be
        positive. The compression against a standard ``Embedding(vocab,
        output_dim)`` is approximately ``output_dim / bottleneck_dim`` for a
        large vocabulary.
    :type bottleneck_dim: int
    :param output_dim: Final embedding dimensionality ``D``.
    :type output_dim: int
    :param embeddings_initializer: Initializer for both the inner embedding
        table and the projection kernel. Default ``"uniform"``, which matches
        ``keras.layers.Embedding``.
    :type embeddings_initializer: Union[str, initializers.Initializer]
    :param embeddings_regularizer: Optional regularizer applied to the inner
        embedding table.
    :type embeddings_regularizer: Optional[Union[str,
        regularizers.Regularizer]]
    :param projection_regularizer: Optional regularizer applied to the
        projection kernel.
    :type projection_regularizer: Optional[Union[str,
        regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``vocab_size`` is not greater than 1, or if
        ``bottleneck_dim`` or ``output_dim`` is non-positive.

    Input shape:
        Integer tensor of any rank, typically ``(batch_size, seq_length)``.
        Values are token ids in ``[0, vocab_size)``.

    Output shape:
        The input shape with ``output_dim`` appended, e.g.
        ``(batch_size, seq_length, output_dim)``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding import (
            albert_factorized_embedding as afe,
        )

        embed = afe.AlbertFactorizedEmbedding(
            vocab_size=50_261, bottleneck_dim=128, output_dim=768,
        )
        ids = keras.random.randint((4, 32), 0, 50261, dtype="int32")
        embed(ids).shape  # (4, 32, 768)
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
        """Validate the configuration and create the two sub-layers.

        Both sub-layers are constructed here and built in :meth:`build`. See
        the class docstring for the meaning of each parameter.

        :param vocab_size: Number of distinct token IDs. Must be > 1.
        :type vocab_size: int
        :param bottleneck_dim: Inner embedding width ``k``. Must be positive.
        :type bottleneck_dim: int
        :param output_dim: Final embedding width ``D``. Must be positive.
        :type output_dim: int
        :param embeddings_initializer: Initializer shared by the inner table
            and the projection kernel.
        :type embeddings_initializer: Union[str, initializers.Initializer]
        :param embeddings_regularizer: Optional regularizer for the inner
            table.
        :type embeddings_regularizer: Optional[Union[str,
            regularizers.Regularizer]]
        :param projection_regularizer: Optional regularizer for the projection
            kernel.
        :type projection_regularizer: Optional[Union[str,
            regularizers.Regularizer]]
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        :raises ValueError: If ``vocab_size`` is not greater than 1, or if
            ``bottleneck_dim`` or ``output_dim`` is non-positive.
        """
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
        """Build the inner embedding table and the projection kernel.

        :param input_shape: Shape of the integer id tensor, e.g.
            ``(batch_size, seq_length)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
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
        """Look the ids up in the inner table and project them up.

        :param inputs: Integer token ids of any shape.
        :type inputs: keras.KerasTensor
        :param training: Accepted for the standard Keras call signature and
            unused: neither sub-layer has training-dependent behaviour.
        :type training: Optional[bool]
        :return: Embeddings with ``output_dim`` appended to the input shape.
        :rtype: keras.KerasTensor
        """
        return self.proj(self.inner_embedding(inputs))

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        """Append ``output_dim`` to the input shape.

        :param input_shape: Shape of the integer id tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(*input_shape, output_dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape) + (self.output_dim,)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        The initializer and both regularizers are serialized to dicts, so
        :meth:`from_config` has to deserialize them again.

        :return: The base ``Layer`` config plus every ``__init__`` parameter.
        :rtype: Dict[str, Any]
        """
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
        """Rebuild the layer, deserializing the initializer and regularizers.

        Each of the three entries is deserialized only when it is present and
        is a dict, so a config that already carries live objects, or that
        carries ``None``, passes through untouched.

        :param config: Configuration dictionary produced by
            :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new layer instance.
        :rtype: AlbertFactorizedEmbedding
        """
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
