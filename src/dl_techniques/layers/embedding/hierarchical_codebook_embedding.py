"""
Hierarchical Codebook Embedding (HCE).

A parameter-efficient replacement for ``keras.layers.Embedding``. It stores
K small codebooks instead of one row per token, and it sums K lookups to
form each embedding.

The name says "hierarchical", but the mechanism is flat. The K chunks of a
token id are read in parallel and no chunk selects which codebook the next
chunk uses. The hierarchy is in the ADDRESSING, not in the computation: high
chunks name a coarse region of id space and low chunks name a position
inside it. That matters for how you order your vocabulary, and it is why the
"Sibling-token correlation" section below exists.

Mechanism:
    1. Each token ID ``i`` is split into ``num_chunks`` integer chunks by
       reading consecutive groups of ``chunk_bits`` bits::

           chunk_k(i) = (i >> (chunk_bits * k)) & (2**chunk_bits - 1)

       The code writes this with ``floor_divide`` and ``mod`` instead of
       shifts, because the backends do not all expose bitwise ops.
    2. Chunk k indexes codebook k, a learnable matrix of shape
       ``(2**chunk_bits, output_dim)``.
    3. The K lookups are summed::

           embed(i) = sum_k E_k[chunk_k(i)]

    4. An optional LayerNorm stabilizes the variance of that sum.

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
    The two layers look related and are not. They sit on opposite sides of
    the model and use opposite compositions.

    - Routing layer, output side: features -> log2(N) sigmoid decisions ->
      multiplicative tree -> N probabilities. It compresses a ``D x N``
      projection to ``D x log(N)``.
    - HCE, input side: token_id -> K codebook lookups -> additive sum ->
      D-dim vector. It compresses ``vocab x D`` storage to
      ``K * 2^chunk_bits * D``.
    - Composition: additive sum here, multiplicative tree there.
    - Direction: discrete to continuous here, continuous to discrete there.
    - Geometry: HCE embeddings live on the Minkowski sum of K finite point
      sets in R^D. Routing probabilities live on a sigmoid manifold of
      dimension ``log2(N)``.

Embedding manifold:
    With K codebooks of M entries each, ``M^K`` distinct embeddings are
    representable. Their affine span has dimension at most ``K * (M - 1)``,
    usually saturated, and bounded above by ``D``.

    For K=2, M=256, D=128 the span reaches D, so there is no practical
    restriction. For K=4, M=16 the bound is 60, which is a real restriction
    at D=128. The model can still reach usable language-model quality there,
    and it does so more easily when the vocabulary order respects chunk
    boundaries. See "Pairing with vocab permutation" below.

Sibling-token correlation:
    Two tokens whose IDs differ in one chunk share K-1 codebook
    contributions. Their embeddings then differ only by the difference of
    two entries in the one differing codebook. Adjacent IDs such as 1234 and
    1235 share 3 of 4 chunks under the default bit layout.

    This is the input-side counterpart of the routing head's leaf-arrangement
    penalty. Pick a vocabulary permutation, Huffman or spectral cluster
    order, that puts semantic boundaries on chunk boundaries.

Pairing with vocab permutation:
    HCE gains a lot from a static permutation that gives semantically
    related tokens IDs sharing high-order chunks. The same Huffman or
    spectral permutation that fixes the routing head's leaf-arrangement
    penalty also fixes HCE's chunk-sharing penalty, so one precomputed
    permutation buys both.

Alternative: ALBERT-style factorized embedding
    Use the ALBERT factorization instead when ``output_dim`` is large, say
    384 or more, and you need each token to occupy any direction
    independently::

        embed_inner = keras.layers.Embedding(vocab, k)         # vocab * k
        embed_proj  = keras.layers.Dense(D, use_bias=False)    # k * D
        embed(i) = embed_proj(embed_inner(i))

    That costs ``vocab * k + k * D`` parameters. For D=768 and k=64 over a
    50K vocabulary this is 3.25M against 38.6M for a standard embedding.
    There is no cross-token coupling. Choose the ALBERT form for full-rank
    per-token embeddings, and HCE for maximum compression when the manifold
    restriction is acceptable. ``AlbertFactorizedEmbedding`` in this package
    implements the ALBERT form.

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

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.embedding.hierarchical_codebook_embedding")
class HierarchicalCodebookEmbedding(keras.layers.Layer):
    """Sum K small codebook lookups into one token embedding.

    Each token id is split into ``num_chunks`` fixed-width chunks. Chunk k
    indexes codebook k. The K resulting vectors are summed, and an optional
    LayerNorm follows. The layer stores
    ``num_chunks * 2**chunk_bits * output_dim`` parameters instead of
    ``vocab_size * output_dim``.

    The chunks are read in parallel. No chunk decides which codebook the
    next chunk uses, so the structure is additive, not a tree. See the
    module docstring for the design rationale, the comparison with the
    routing head and the trade-off against ALBERT-style factorization.

    **Architecture Overview:**

    .. code-block:: text

        token id i  (e.g. 50260, chunk_bits=8, num_chunks=2)
             │
             │  chunk_k(i) = (i // 2^(chunk_bits*k)) % 2^chunk_bits
             │
             ├──────────────────────┬─────────────────  ...
             ▼                      ▼
        chunk_0 = i % 256      chunk_1 = (i // 256) % 256
        (low bits)             (high bits)
             │                      │
             ▼                      ▼
        ┌──────────────┐       ┌──────────────┐
        │ codebook_0   │       │ codebook_1   │    ... K tables
        │ (256, D)     │       │ (256, D)     │
        └──────┬───────┘       └──────┬───────┘
               │  take(row)           │  take(row)
               ▼                      ▼
            E_0[chunk_0]  ── + ──  E_1[chunk_1]  ... + E_k[chunk_k]
                              │
                              ▼
                   optional LayerNormalization
                              │
                              ▼
                  output (..., output_dim)

    :param vocab_size: Number of distinct token IDs the layer must support.
        Must be greater than 1. Token IDs in ``[0, vocab_size)`` produce
        well-defined embeddings. IDs in
        ``[vocab_size, 2**(num_chunks * chunk_bits))`` are accepted and
        address codebook regions no real token visits, so they never receive
        gradient.
    :type vocab_size: int
    :param output_dim: Embedding dimensionality D. Must be positive.
    :type output_dim: int
    :param num_chunks: Number of codebooks K. Must be positive. Defaults
        to 2.
    :type num_chunks: int
    :param chunk_bits: Bits per chunk. ``None`` (default) computes
        ``ceil(ceil(log2(vocab_size)) / num_chunks)``. Codebook size is
        ``2**chunk_bits``, so one extra bit doubles the parameter count.
    :type chunk_bits: Optional[int]
    :param use_layer_norm: Whether to apply LayerNorm to the summed
        embedding. Defaults to ``True``. A sum of K independent codebook
        contributions has roughly K times the variance of one, which
        destabilizes the layers below.
    :type use_layer_norm: bool
    :param epsilon: Variance floor added inside the internal
        ``LayerNormalization``; ignored when ``use_layer_norm`` is ``False``.
        Defaults to ``1e-3``.

        **The default is Keras' value, not the repo's usual ``1e-6``, and that
        is deliberate.** Before this parameter existed the sub-layer was built
        as ``LayerNormalization(name="hce_norm")`` and so took Keras' ``1e-3``;
        keeping that as the default means adding the knob moves no already
        trained model. The sibling classes ``BertEmbeddings`` and
        ``ModernBertEmbeddings``, and ``create_normalization_layer``, all use
        ``1e-6``. **Pass ``epsilon=1e-6`` explicitly for a new model** — this
        layer is a bespoke codebook design with no reference architecture to
        inherit a value from, so nothing here argues for ``1e-3`` on its
        merits; only continuity does.
    :type epsilon: float
    :param embeddings_initializer: Initializer for the codebook tables.
        Defaults to ``"uniform"``, matching ``keras.layers.Embedding``.
    :type embeddings_initializer: Union[str, initializers.Initializer]
    :param embeddings_regularizer: Optional regularizer applied to every
        codebook.
    :type embeddings_regularizer: Optional[Union[str, regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar chunk_bits: The resolved bit width, never ``None``. ``__init__``
        computes it when the argument is ``None``, and ``get_config()``
        reports the resolved integer.
    :vartype chunk_bits: int
    :ivar codebooks: The K weight tables, each of shape
        ``(2**chunk_bits, output_dim)``. Empty until ``build()`` runs.
    :vartype codebooks: list
    :ivar layer_norm: The LayerNormalization instance, or ``None`` when
        ``use_layer_norm`` is ``False``.
    :vartype layer_norm: keras.layers.LayerNormalization or None

    Input shape:
        Integer tensor of any shape holding token IDs. Non-integer inputs
        are cast to ``int32``.

    Output shape:
        The input shape with ``output_dim`` appended.

    :raises ValueError: If ``vocab_size`` is not greater than 1, if
        ``output_dim`` or ``num_chunks`` is not positive, if an explicit
        ``chunk_bits`` is not positive, or if ``num_chunks * chunk_bits``
        cannot address ``vocab_size`` codes. Raised from ``__init__``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding import (
            hierarchical_codebook_embedding as hce,
        )

        embed = hce.HierarchicalCodebookEmbedding(
            vocab_size=50261, output_dim=128, num_chunks=2,
        )
        ids = keras.random.randint((4, 32), 0, 50261)
        embed(ids).shape  # (4, 32, 128)
    """

    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        num_chunks: int = 2,
        chunk_bits: Optional[int] = None,
        use_layer_norm: bool = True,
        epsilon: float = 1e-3,
        embeddings_initializer: Union[str, initializers.Initializer] = "uniform",
        embeddings_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and resolve ``chunk_bits``.

        No weight is created here. The K codebooks are created in
        :meth:`build`.

        :param vocab_size: Number of distinct token IDs to support.
        :type vocab_size: int
        :param output_dim: Embedding dimensionality D.
        :type output_dim: int
        :param num_chunks: Number of codebooks K.
        :type num_chunks: int
        :param chunk_bits: Bits per chunk, or ``None`` to derive it.
        :type chunk_bits: Optional[int]
        :param use_layer_norm: Whether to normalize the summed embedding.
        :type use_layer_norm: bool
        :param epsilon: Variance floor of the internal
            ``LayerNormalization``. Ignored when ``use_layer_norm`` is
            ``False``. Defaults to ``1e-3`` -- see the anchor in ``__init__``
            for why the default is Keras' value and not the ``1e-6`` the
            sibling classes and the norms factory use.
        :type epsilon: float
        :param embeddings_initializer: Initializer for the codebooks.
        :type embeddings_initializer: Union[str, initializers.Initializer]
        :param embeddings_regularizer: Optional codebook regularizer.
        :type embeddings_regularizer: Optional[Union[str, regularizers.Regularizer]]
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        :raises ValueError: If ``vocab_size`` is not greater than 1, if
            ``output_dim`` or ``num_chunks`` is not positive, if an explicit
            ``chunk_bits`` is not positive, or if the chunk layout cannot
            address ``vocab_size`` codes.
        """
        super().__init__(**kwargs)

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

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
        self.epsilon = epsilon
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)

        self._codebook_size = 1 << chunk_bits
        # Pre-computed integer divisors for chunk extraction (avoids
        # constructing a Python int per call).
        self._chunk_divisors = [1 << (chunk_bits * k) for k in range(num_chunks)]
        self._chunk_modulus = self._codebook_size

        # Filled in build(), one entry per chunk.
        self.codebooks = []
        # DECISION plan-2026-08-28T181715-3870472c/D-010
        # `epsilon` DEFAULTS TO 1e-3, which is Keras' own `LayerNormalization`
        # default and therefore EXACTLY the value this layer used before the
        # parameter existed. Do NOT "fix" the default to 1e-6 to match
        # `bert_embeddings.py` / `modern_bert_embeddings.py` / the norms factory:
        # this is a bespoke codebook design with no reference architecture to
        # inherit a value from, so there is nothing to be faithful TO, and moving
        # the default would silently change the output of every already-trained
        # HierarchicalCodebookEmbedding checkpoint -- 1000x in the denominator,
        # with no shape symptom and no warning. Pass `epsilon=1e-6` explicitly
        # for a new model; that is the recommended value and the whole reason
        # this knob was added. See decisions.md D-010.
        #
        # The sub-layer is deliberately still created inside an `if` (guide
        # §1.3). Audited as SOFT, not a defect, because the one constructing
        # branch carries an explicit `name=`, so no downstream auto-generated
        # name can shift. Making it unconditional is NOT free: it would attach an
        # unused, tracked `LayerNormalization` to every `use_layer_norm=False`
        # layer, which is a change to the serialized structure and exactly the
        # kind of restructuring-a-working-layer this step was told not to do.
        self.layer_norm = (
            keras.layers.LayerNormalization(epsilon=epsilon, name="hce_norm")
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
        """Create the K codebook tables and build the optional LayerNorm.

        Each codebook has shape ``(2**chunk_bits, output_dim)`` and is
        trainable. The LayerNorm is built against the OUTPUT shape, the
        input shape with ``output_dim`` appended, because it runs after the
        sum rather than on the ids.

        :param input_shape: Shape of the token-id tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

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
        """Split each token id into chunks and sum the K codebook lookups.

        :param inputs: Integer tensor of token IDs, of any shape. Cast to
            ``int32``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode or
            inference mode. Forwarded to the LayerNorm only.
        :type training: Optional[bool]
        :return: Embeddings of shape ``inputs.shape + (output_dim,)``.
        :rtype: keras.KerasTensor
        """
        ids = ops.cast(inputs, "int32")

        out = None
        for k in range(self.num_chunks):
            # chunk_k(i) = (i // 2^(chunk_bits*k)) % codebook_size, written
            # with integer arithmetic rather than shifts: the keras.ops
            # bitwise operations are not available on every backend.
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
        """Append ``output_dim`` to the input shape.

        :param input_shape: Shape of the token-id tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The input shape with ``output_dim`` appended.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape) + (self.output_dim,)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer for serialization.

        ``chunk_bits`` is reported as the resolved integer, not as the
        ``None`` a caller may have passed.

        :return: Dictionary holding every ``__init__`` parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "output_dim": self.output_dim,
            "num_chunks": self.num_chunks,
            "chunk_bits": self.chunk_bits,
            "use_layer_norm": self.use_layer_norm,
            "epsilon": self.epsilon,
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
        """Rebuild the layer, deserializing the initializer and regularizer.

        The two entries arrive as dicts from ``.keras`` files and are turned
        back into objects here. The caller's mapping is COPIED first, so the
        dict passed in is left untouched and stays reusable.

        :param config: Configuration produced by ``get_config()``. Not
            modified.
        :type config: Dict[str, Any]
        :return: A new layer instance.
        :rtype: HierarchicalCodebookEmbedding
        """
        # Shallow copy is enough: only top-level keys are rebound below, and
        # the nested dicts are handed to `deserialize`, which reads them.
        config = dict(config)
        for key in ("embeddings_initializer", "embeddings_regularizer"):
            if config.get(key) and isinstance(config[key], dict):
                if key == "embeddings_initializer":
                    config[key] = initializers.deserialize(config[key])
                else:
                    config[key] = regularizers.deserialize(config[key])
        return cls(**config)

# ---------------------------------------------------------------------
