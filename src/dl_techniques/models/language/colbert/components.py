"""
The two layers that turn a token encoder into a late-interaction retriever:
:class:`ColBERTProjection` and :class:`MaxSimScorer`.

Late interaction encodes the query and the document independently, so a
document can be encoded once, offline, and compares their token embeddings
only afterwards, with a cheap similarity: each query token is scored against
its single best document token, and those per-term maxima are summed::

    S(q, d) = sum_i  max_j  E_q[i] . E_d[j]

The encoder is shared and the embeddings are L2-normalized, so each inner
product is a cosine similarity in [-1, 1], bounded by the query length.
:class:`ColBERTProjection` produces the normalized per-token embeddings from a
transformer's ``last_hidden_state``; :class:`MaxSimScorer` performs the
reduction.

Masking happens before normalizing, never after, so a fully-masked row starts
from a zero vector; normalization runs through :func:`_safe_l2_normalize`
rather than ``keras.ops.normalize``, which returns ``NaN`` for a zero row
under ``mixed_float16`` because its epsilon floor underflows at half
precision. :class:`MaxSimScorer` overwrites masked document scores with a
finite sentinel via ``where`` rather than adding ``(1 - mask) * -1e9``, since
the additive form underflows to ``-inf`` at float16 and turns into ``NaN``.

References:
    - Khattab and Zaharia, 2020. ColBERT: Efficient and Effective Passage
      Search via Contextualized Late Interaction over BERT.
      (https://arxiv.org/abs/2004.12832)
    - Santhanam et al., 2022. ColBERTv2: Effective and Efficient Retrieval via
      Lightweight Late Interaction. (https://arxiv.org/abs/2112.01488)
    - Devlin et al., 2018. BERT: Pre-training of Deep Bidirectional
      Transformers for Language Understanding.
      (https://arxiv.org/abs/1810.04805)
"""

import keras
from typing import Any, Dict, Optional, Tuple
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Reduced-precision compute dtypes whose range is too small to hold the MaxSim
# reduction; see :meth:`MaxSimScorer._reduction_dtype`.
_LOW_PRECISION_DTYPES = ("float16", "bfloat16")

# Default MaxSim padding sentinel.
#
# Derivation (this value is reasoned, not copied): the embeddings entering the
# scorer are L2-normalized, so every entry of the dense score matrix is a
# cosine similarity in [-1, 1]. A sentinel therefore only has to sit safely
# below -1 to be unwinnable by the max-reduce. The reference implementation
# uses -9999; -1e4 is the same order of magnitude, is exactly representable in
# float16 (the binary16 spacing at 1e4 is 8, and 10000 / 8 = 1250 exactly), and
# leaves four orders of magnitude of headroom before binary16 overflows to
# -inf, which is the failure the sentinel exists to avoid in the first place.
DEFAULT_MAXSIM_MASK_VALUE = -1e4

# Floor on the squared L2 norm used by :func:`_safe_l2_normalize`. Small enough
# to be invisible against a real row (whose squared norm is O(1)), large enough
# that ``rsqrt`` of it stays far inside float32's range.
_L2_NORM_SQUARED_FLOOR = 1e-12

# ---------------------------------------------------------------------


# DECISION plan-2026-08-25T121346-c71fc3ad/D-006: not keras.ops.normalize(x, axis=-1) — it NaNs
# an all-zero row under mixed_float16 since its epsilon floor underflows at half precision. See decisions.md.
def _safe_l2_normalize(x: Any) -> Any:
    """L2-normalize the last axis, returning exact zeros for a zero row.

    The reduction runs in ``float32`` regardless of the compute dtype and the
    result is cast back, so the epsilon floor cannot underflow.

    :param x: tensor of rank >= 1.
    :return: ``x`` normalized along its last axis, in ``x``'s own dtype.
    """
    compute_dtype = x.dtype
    x32 = keras.ops.cast(x, "float32")
    squared_norm = keras.ops.sum(keras.ops.square(x32), axis=-1, keepdims=True)
    inverse_norm = keras.ops.rsqrt(
        keras.ops.maximum(squared_norm, _L2_NORM_SQUARED_FLOOR)
    )
    return keras.ops.cast(x32 * inverse_norm, compute_dtype)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.colbert.components")
class ColBERTProjection(keras.layers.Layer):
    """Bias-free linear projection to the retrieval dimension, then L2 normalize.

    Maps a transformer's per-token hidden states to the low-dimensional space
    the index is built in, zeroes any masked position, and normalizes each
    surviving token vector to unit length so that a later inner product is a
    cosine similarity.

    The projection is a single ``Dense`` with no bias and no activation,
    matching the reference implementation
    (``nn.Linear(hidden_size, dim, bias=False)``): an activation would break
    the linearity the MaxSim score's geometric reading depends on, and a bias
    would give a fully-masked position a non-zero embedding, defeating the
    mask.

    The same layer instance is intended to serve the query path and the
    document path. Two instances would be two sets of weights and would no
    longer be the shared encoder the architecture specifies.

    :param dim: size of the projected embedding. Must be strictly positive.
        Defaults to ``128``, the reference default.
    :type dim: int
    :param kwargs: forwarded to :class:`keras.layers.Layer`.
    :raises ValueError: if ``dim`` is not a strictly positive integer.

    Example::

        projection = ColBERTProjection(dim=128)
        embeddings = projection(hidden_states, mask=attention_mask)
        # embeddings: (batch, seq_len, 128), unit-norm at unmasked positions,
        # exactly zero at masked positions.
    """

    def __init__(self, dim: int = 128, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError(
                f"dim must be a strictly positive integer, received {dim!r}"
            )

        self.dim = dim

        # Created here, built in build() -- the sub-layer tree must be
        # materialized so that .variables is populated after an explicit build.
        self.dense = keras.layers.Dense(
            units=self.dim,
            use_bias=False,
            name="projection",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Materialize the projection weight.

        :param input_shape: shape of ``hidden_states``, ``(batch, seq_len, hidden)``.
        :type input_shape: tuple
        """
        if self.built:
            return
        self.dense.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states: Any,
        mask: Optional[Any] = None,
        training: Optional[bool] = None,
    ) -> Any:
        """Project, mask, then L2-normalize.

        :param hidden_states: ``(batch, seq_len, hidden)`` encoder outputs.
        :type hidden_states: keras tensor
        :param mask: optional rank-2 ``(batch, seq_len)`` mask where 1 means
            keep and 0 means padding or filtered. Broadcast over the feature
            axis and multiplied onto the projected vectors *before*
            normalization.
        :type mask: keras tensor or None
        :param training: unused; present for the standard layer signature.
        :type training: bool or None
        :return: ``(batch, seq_len, dim)`` embeddings, unit-norm where kept and
            exactly zero where masked.
        """
        x = self.dense(hidden_states)

        if mask is not None:
            mask_f = keras.ops.cast(mask, x.dtype)
            x = x * keras.ops.expand_dims(mask_f, axis=-1)

        # Mask first, normalize second -- never the reverse (see the module
        # docstring).
        return _safe_l2_normalize(x)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """:return: ``input_shape`` with its last axis replaced by ``dim``."""
        return tuple(input_shape[:-1]) + (self.dim,)

    def get_config(self) -> Dict[str, Any]:
        """:return: full constructor configuration for serialization."""
        config = super().get_config()
        config.update({"dim": self.dim})
        return config


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.colbert.components")
class MaxSimScorer(keras.layers.Layer):
    """Late-interaction MaxSim reduction over a dense query-by-document matrix.

    Computes ``S(q, d) = sum_i max_j E_q[i] . E_d[j]``: the full dense
    ``(batch, query_len, doc_len)`` similarity matrix, a max over the document
    axis, then a sum over the query axis. The order is not symmetric: summing
    first and maxing second would score a document by its single
    best-matching token overall rather than by how well it covers every query
    term, which is the property late interaction exists to measure.

    Padded or punctuation-filtered document positions are removed from
    contention by overwriting their scores with a finite large-negative
    sentinel *before* the max, using ``where``. See the module docstring for why
    this is not an additive ``(1 - mask) * -1e9`` bias.

    :param mask_value: sentinel written into masked document positions. Must be
        finite and strictly negative. Defaults to
        :data:`DEFAULT_MAXSIM_MASK_VALUE`. The masking and both reductions are
        promoted to ``float32`` under a reduced-precision policy so that neither
        the sentinel nor the sum over it can overflow to ``-inf``.
    :type mask_value: float
    :param kwargs: forwarded to :class:`keras.layers.Layer`.
    :raises ValueError: if ``mask_value`` is not finite or is not negative.

    Example::

        scorer = MaxSimScorer()
        scores = scorer(query_embeddings, doc_embeddings, doc_mask=doc_mask)
        # scores: (batch,)
    """

    def __init__(
        self,
        mask_value: float = DEFAULT_MAXSIM_MASK_VALUE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        mask_value = float(mask_value)
        if mask_value != mask_value or mask_value in (float("inf"), float("-inf")):
            raise ValueError(
                f"mask_value must be finite, received {mask_value!r}. An "
                "infinite sentinel makes a fully-masked document score -inf, "
                "which becomes NaN downstream."
            )
        if mask_value >= 0.0:
            raise ValueError(
                f"mask_value must be strictly negative, received {mask_value!r}"
            )

        self.mask_value = mask_value

    def build(self, *args: Any, **kwargs: Any) -> None:
        """Stateless layer; no weights to create."""
        super().build(args[0] if args else None)

    # DECISION plan-2026-08-25T121346-c71fc3ad/D-007: reduce in float32 under mixed_float16,
    # never the compute dtype — summing query_len sentinels (32 * -1e4) overflows binary16 to -inf. See decisions.md.
    def _reduction_dtype(self, dtype: Any) -> str:
        """Dtype the masking and the two reductions are performed in.

        :param dtype: the dtype the raw score matrix came out in.
        :return: ``"float32"`` for a reduced-precision compute dtype, otherwise
            the incoming dtype unchanged.
        """
        # DECISION plan-2026-08-25T121346-c71fc3ad/D-014: not keras.backend.standardize_dtype
        # (banned Keras-2 residue) — read .name for a tf.DType, fall back to str for a plain string.
        standardized = getattr(dtype, "name", None) or str(dtype)
        if standardized in _LOW_PRECISION_DTYPES:
            return "float32"
        return standardized

    def call(
        self,
        query_embeddings: Any,
        doc_embeddings: Any,
        doc_mask: Optional[Any] = None,
        query_mask: Optional[Any] = None,
    ) -> Any:
        """Score every query against its paired document.

        :param query_embeddings: ``(batch, query_len, dim)``.
        :type query_embeddings: keras tensor
        :param doc_embeddings: ``(batch, doc_len, dim)``.
        :type doc_embeddings: keras tensor
        :param doc_mask: optional rank-2 ``(batch, doc_len)`` mask, 1 = keep.
            Masked positions are sentinel-filled before the max.
        :type doc_mask: keras tensor or None
        :param query_mask: optional rank-2 ``(batch, query_len)`` mask,
            1 = keep. Masked query terms are zeroed before the sum so a
            padding query term contributes exactly 0 rather than its own best
            match. Required for correctness whenever queries are padded, since
            the reference pads queries with ``[MASK]`` rather than with zeros,
            so a padding query position carries a real, non-zero embedding.
        :type query_mask: keras tensor or None
        :return: ``(batch,)`` MaxSim scores. ``float32`` under a
            reduced-precision policy, otherwise the input dtype.
        """
        scores = keras.ops.einsum("bqd,bsd->bqs", query_embeddings, doc_embeddings)
        scores = keras.ops.cast(scores, self._reduction_dtype(scores.dtype))

        if doc_mask is not None:
            keep = keras.ops.cast(
                keras.ops.expand_dims(doc_mask, axis=1), "bool"
            )  # (batch, 1, doc_len)
            sentinel = keras.ops.full_like(scores, self.mask_value)
            scores = keras.ops.where(keep, scores, sentinel)

        # max over the DOCUMENT axis -> (batch, query_len)
        best_per_query_term = keras.ops.max(scores, axis=-1)

        if query_mask is not None:
            best_per_query_term = best_per_query_term * keras.ops.cast(
                query_mask, best_per_query_term.dtype
            )

        # sum over the QUERY axis -> (batch,)
        return keras.ops.sum(best_per_query_term, axis=-1)

    def compute_output_shape(
        self,
        query_embeddings_shape: Tuple[Optional[int], ...],
        doc_embeddings_shape: Optional[Tuple[Optional[int], ...]] = None,
        doc_mask_shape: Optional[Tuple[Optional[int], ...]] = None,
        query_mask_shape: Optional[Tuple[Optional[int], ...]] = None,
    ) -> Tuple[Optional[int], ...]:
        """:return: ``(batch,)`` -- both sequence axes are reduced away."""
        return (query_embeddings_shape[0],)

    def get_config(self) -> Dict[str, Any]:
        """:return: full constructor configuration for serialization."""
        config = super().get_config()
        config.update({"mask_value": self.mask_value})
        return config
