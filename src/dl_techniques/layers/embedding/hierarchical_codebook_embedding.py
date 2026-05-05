"""Hierarchical Codebook Embedding (HCE).

A parameter-efficient drop-in replacement for ``keras.layers.Embedding``
inspired by routing-tree structure but using an *additive* (not
multiplicative) composition mechanism.

Mechanism:
    1. Each token ID ``i`` is decomposed into ``num_chunks`` integer
       chunks by reading consecutive groups of ``chunk_bits`` bits:
       ``chunk_k(i) = (i >> (chunk_bits * k)) & (2**chunk_bits - 1)``.
    2. The k-th chunk indexes into the k-th codebook, a learnable matrix
       of shape ``(2**chunk_bits, output_dim)``.
    3. The K codebook lookups are summed to produce the embedding:
       ``embed(i) = sum_k E_k[chunk_k(i)]``.
    4. Optional final LayerNorm stabilizes the variance of the sum.

Parameter count: ``num_chunks * 2**chunk_bits * output_dim``.
For vocab=50,261 and output_dim=128:

==============================  =================  ==============
config                          params             vs. Embedding
==============================  =================  ==============
standard Embedding              ~6,433,408         1.0x
HCE(num_chunks=2, M=256)        ~65,536            ~98x smaller
HCE(num_chunks=4, M=16)         ~8,192             ~785x smaller
==============================  =================  ==============

Asymmetry vs. ``RoutingProbabilitiesLayer``:
    - Routing layer (output side):  features  -> log2(N) sigmoid
      decisions -> multiplicative tree -> N probabilities. Compresses
      ``D x N`` projection cost to ``D x log(N)``.
    - HCE (input side):  token_id -> K codebook lookups -> additive sum
      -> D-dim vector. Compresses ``vocab x D`` storage cost to
      ``K * 2^chunk_bits * D``.
    - Mechanisms: additive sum (HCE) vs multiplicative tree (routing).
    - Roles: discrete -> continuous lookup (HCE) vs continuous ->
      discrete projection (routing).
    - Geometry: HCE embeddings live on the Minkowski sum of K
      finite point sets in R^D; routing probs live on a sigmoid
      manifold of dimension ``log2(N)``.

Embedding manifold:
    With K codebooks of M entries each, ``M^K`` distinct embeddings
    are representable. Their affine span has dimension at most
    ``K * (M - 1)`` (typically saturated), bounded above by ``D``. For
    K=2, M=256, D=128: span dim ~= D, no practical restriction. For
    K=4, M=16: span dim ~= 60 — meaningful restriction at D=128, but
    the model can still achieve usable LM-quality embeddings,
    especially when the vocab-to-leaf assignment respects semantic
    structure (see "Pairing with vocab permutation" below).

Sibling-token correlation:
    Tokens whose IDs differ in only one chunk share K-1 codebook
    contributions; their embeddings differ only by the difference of
    the differing chunk's two codebook entries. Adjacent token IDs
    (e.g. id=1234 vs 1235) share 3 of 4 chunks under default bit
    layout. This is the input-side analogue of the routing head's
    leaf-arrangement penalty: minimized by choosing a vocab
    permutation (Huffman or spectral cluster order) that aligns chunk
    boundaries with semantic boundaries.

Pairing with vocab permutation:
    HCE benefits substantially from a static permutation that places
    semantically related tokens at IDs sharing common high-order
    chunks. The same Huffman/spectral permutation that fixes the
    routing-head leaf-arrangement penalty also fixes HCE's chunk-
    sharing penalty, so a single precomputed permutation buys both
    benefits.

Alternative: ALBERT-style factorized embedding
    When ``output_dim`` is large (>= 384) and you want unrestricted
    expressivity per token (full-rank embedding manifold) instead of
    HCE's restricted Minkowski-sum manifold, the standard alternative
    is the ALBERT factorization::

        embed_inner = keras.layers.Embedding(vocab, k)         # vocab * k
        embed_proj  = keras.layers.Dense(D, use_bias=False)    # k * D
        embed(i) = embed_proj(embed_inner(i))

    Parameters: ``vocab * k + k * D`` (e.g. 50K * 64 + 64 * 768 = 3.25M
    for D=768, k=64 vs 38.6M standard). Each token's embedding can
    independently occupy any direction in the k-dim subspace projected
    to D, with no cross-token coupling. Use ALBERT factorization when
    you need full-rank per-token embeddings; use HCE when you want
    maximum parameter compression and can tolerate the manifold
    restriction.

References:
    - Jegou, H., Douze, M., Schmid, C. (2010). "Product Quantization
      for Nearest Neighbor Search." IEEE TPAMI. The additive Cartesian
      decomposition that inspires HCE's parameter saving.
    - Lan, Z., et al. (2019). "ALBERT: A Lite BERT for Self-supervised
      Learning of Language Representations." arXiv:1909.11942. The
      factorized-embedding alternative.
    - Chen, T., et al. (2018). "Learning K-way D-dimensional Discrete
      Codes for Compact Embedding Representations." ICML. Related
      learned-codebook embedding scheme.
"""

import keras
from keras import ops, initializers, regularizers
from typing import Any, Dict, Optional, Tuple, Union

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class HierarchicalCodebookEmbedding(keras.layers.Layer):
    """Additive multi-codebook embedding for parameter-efficient token lookup.

    See module docstring for full design rationale, asymmetry vs. the
    routing head, and trade-offs against ALBERT-style factorization.

    :param vocab_size: Number of distinct token IDs the layer must support.
        Token IDs in ``[0, vocab_size)`` produce well-defined embeddings;
        IDs in ``[vocab_size, 2**(num_chunks * chunk_bits))`` are valid
        inputs but address codebook regions that real tokens never visit
        (and thus never receive gradient).
    :param output_dim: Embedding dimensionality D.
    :param num_chunks: Number of codebooks K. Default 2.
    :param chunk_bits: Bits per chunk. If None (default), auto-computed
        as ``ceil(ceil(log2(vocab_size)) / num_chunks)``. Codebook size
        is ``2**chunk_bits``.
    :param use_layer_norm: Apply LayerNorm to the summed embedding.
        Recommended (default True) — sums of K independent codebook
        contributions can have non-unit variance that destabilizes
        downstream layers.
    :param embeddings_initializer: Initializer for codebook tables.
        Default ``"uniform"`` (matching ``keras.layers.Embedding``).
    :param embeddings_regularizer: Optional regularizer applied to each
        codebook.

    :raises ValueError: If ``vocab_size``, ``output_dim``, ``num_chunks``,
        or ``chunk_bits`` are non-positive, or if
        ``num_chunks * chunk_bits`` cannot address ``vocab_size`` codes.

    Example::

        # 50K vocab, D=128 -> 65K params instead of 6.4M (98x compression)
        embed = HierarchicalCodebookEmbedding(
            vocab_size=50261, output_dim=128, num_chunks=2,
        )
        ids = keras.random.uniform((4, 32), 0, 50261, dtype="int32")
        x = embed(ids)              # shape: (4, 32, 128)
    """

    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        num_chunks: int = 2,
        chunk_bits: Optional[int] = None,
        use_layer_norm: bool = True,
        embeddings_initializer: Union[str, initializers.Initializer] = "uniform",
        embeddings_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if vocab_size <= 1:
            raise ValueError(
                f"vocab_size must be > 1, got {vocab_size}"
            )
        if output_dim <= 0:
            raise ValueError(
                f"output_dim must be positive, got {output_dim}"
            )
        if num_chunks <= 0:
            raise ValueError(
                f"num_chunks must be positive, got {num_chunks}"
            )

        total_bits_needed = (vocab_size - 1).bit_length()
        if chunk_bits is None:
            # Round up so num_chunks * chunk_bits >= total_bits_needed.
            chunk_bits = (total_bits_needed + num_chunks - 1) // num_chunks
        elif chunk_bits <= 0:
            raise ValueError(
                f"chunk_bits must be positive, got {chunk_bits}"
            )

        if num_chunks * chunk_bits < total_bits_needed:
            raise ValueError(
                f"num_chunks * chunk_bits ({num_chunks} * {chunk_bits} = "
                f"{num_chunks * chunk_bits}) cannot address vocab_size="
                f"{vocab_size} (needs >= {total_bits_needed} bits)"
            )

        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.num_chunks = num_chunks
        self.chunk_bits = chunk_bits
        self.use_layer_norm = use_layer_norm
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)

        self._codebook_size = 1 << chunk_bits
        # Pre-computed integer divisors for chunk extraction (avoids
        # constructing a Python int per call).
        self._chunk_divisors = [1 << (chunk_bits * k) for k in range(num_chunks)]
        self._chunk_modulus = self._codebook_size

        self.codebooks = []  # filled in build()
        self.layer_norm = (
            keras.layers.LayerNormalization(name="hce_norm")
            if use_layer_norm
            else None
        )

        # Param-count summary for logging.
        n_params = num_chunks * self._codebook_size * output_dim
        n_dense = vocab_size * output_dim
        compression = n_dense / max(1, n_params)
        logger.info(
            f"HierarchicalCodebookEmbedding(vocab={vocab_size}, "
            f"D={output_dim}, K={num_chunks}, chunk_bits={chunk_bits}, "
            f"M={self._codebook_size}): {n_params:,} params "
            f"(~{compression:.1f}x smaller than Embedding({vocab_size},{output_dim})"
            f"={n_dense:,} params)"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        for k in range(self.num_chunks):
            cb = self.add_weight(
                name=f"codebook_{k}",
                shape=(self._codebook_size, self.output_dim),
                initializer=self.embeddings_initializer,
                regularizer=self.embeddings_regularizer,
                trainable=True,
            )
            self.codebooks.append(cb)

        if self.layer_norm is not None:
            self.layer_norm.build(tuple(input_shape) + (self.output_dim,))

        super().build(input_shape)

    def call(
        self, inputs: keras.KerasTensor, training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        # inputs: int tensor of token IDs, shape (..., )
        ids = ops.cast(inputs, "int32")

        out = None
        for k in range(self.num_chunks):
            # chunk_k(i) = (i // 2^(chunk_bits*k)) % codebook_size
            # Using integer arithmetic for backend portability (keras.ops
            # bitwise ops are not uniformly available across all backends).
            chunk_idx = ops.mod(
                ops.floor_divide(ids, self._chunk_divisors[k]),
                self._chunk_modulus,
            )
            lookup = ops.take(self.codebooks[k], chunk_idx, axis=0)
            out = lookup if out is None else out + lookup

        if self.layer_norm is not None:
            out = self.layer_norm(out, training=training)

        return out

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        return tuple(input_shape) + (self.output_dim,)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "output_dim": self.output_dim,
            "num_chunks": self.num_chunks,
            "chunk_bits": self.chunk_bits,
            "use_layer_norm": self.use_layer_norm,
            "embeddings_initializer": initializers.serialize(
                self.embeddings_initializer,
            ),
            "embeddings_regularizer": regularizers.serialize(
                self.embeddings_regularizer,
            ),
        })
        return config

    @classmethod
    def from_config(
        cls, config: Dict[str, Any],
    ) -> "HierarchicalCodebookEmbedding":
        for key in ("embeddings_initializer", "embeddings_regularizer"):
            if config.get(key) and isinstance(config[key], dict):
                if key == "embeddings_initializer":
                    config[key] = initializers.deserialize(config[key])
                else:
                    config[key] = regularizers.deserialize(config[key])
        return cls(**config)
