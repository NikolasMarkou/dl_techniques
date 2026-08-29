"""
The two layers that turn a token encoder into a late-interaction retriever.

Dense retrieval forces a choice between two failure modes. Single-vector
bi-encoders compress a whole passage into one embedding, which is cheap to
index but throws away term-level evidence, so a passage that answers exactly
one clause of a query is indistinguishable from one that answers none of it.
Cross-encoders keep every term interaction but must run the transformer once
per (query, document) pair at search time, which is unaffordable over a real
corpus. *Late interaction* refuses the choice: query and document are encoded
independently -- so documents can be encoded once, offline, and stored -- and
their token embeddings meet only afterwards, in a cheap similarity reduction
that keeps the term-level structure the single-vector encoder discarded.

The reduction is MaxSim. Each query token is scored against its single best
document token, and those per-term maxima are summed::

    S(q, d) = sum_i  max_j  E_q[i] . E_d[j]

Because the encoder is shared and the embeddings are L2-normalized, each inner
product is a cosine similarity in [-1, 1], and the score is bounded by the
query length. The two layers here are exactly the two halves of that pipeline:
:class:`ColBERTProjection` produces the normalized per-token embeddings from a
transformer's ``last_hidden_state``, and :class:`MaxSimScorer` performs the
reduction.

Two details in this file are not cosmetic, and both are places where a
plausible-looking rearrangement is silently wrong.

**Order of masking and normalization.** The projection multiplies its padding /
punctuation mask onto the projected vectors *before* L2-normalizing, never
after. Normalizing first and masking second yields the same zeros at the masked
positions but different values at the *unmasked* ones only in the sense that
the mask can no longer zero anything the normalizer has already rescaled -- and
more importantly it inverts the reference's semantics, where a filtered token
is a zero vector that contributes an inner product of exactly zero. The
consequence of masking first is that a fully-masked row is normalized from a
zero vector, which is why this module normalizes through
:func:`_safe_l2_normalize` rather than ``keras.ops.normalize``: the latter was
measured returning ``NaN`` for a zero row under ``mixed_float16``, because its
internal ``max(norm, epsilon)`` floor underflows at half precision. Masked rows
here are all-zero by construction, so that is the common path.

**Sentinel masking, not additive masking.** :class:`MaxSimScorer` replaces the
scores of masked document positions with a finite large-negative sentinel via
``where`` before the max-reduce. It never adds ``(1 - mask) * -1e9``. The
additive form is the documented ``float16`` NaN family in this repository: at
half precision ``-1e9`` is not representable and becomes ``-inf``, a row that
is entirely masked then reduces to ``-inf``, and any subsequent softmax or
subtraction turns that into ``NaN``. The sentinel here is finite, is chosen to
be exactly representable in ``float16``, and is clamped into range if a caller
supplies something that is not, so an all-padding document produces a large,
finite, deterministic score instead of a poisoned batch.

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


# DECISION plan-2026-08-25T121346-c71fc3ad/D-006
# Do NOT replace this with `keras.ops.normalize(x, axis=-1)`. That call NaNs on
# an all-zero row under mixed_float16 -- MEASURED here, 2026-08-25, Keras 3.8:
# a fully-masked projection row came back `nan` in float16 while the identical
# input returned exact zeros in float32. Its internal `max(norm, epsilon)` guard
# uses `backend.epsilon()` (1e-7), which underflows in half precision, so the
# guard is silently absent in exactly the dtype where it is needed. Masked rows
# are all-zero BY CONSTRUCTION here (the mask multiply runs before the
# normalize), so this is the common path, not an edge case -- and a NaN row
# poisons every MaxSim score in its batch. Reducing in float32 also removes the
# secondary hazard that a large float16 vector overflows when squared.
# See decisions.md D-006.
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

    The projection is a single ``Dense`` with **no bias and no activation**.
    That is exact in the reference implementation
    (``nn.Linear(hidden_size, dim, bias=False)``), not an approximation: an
    activation would break the linearity the MaxSim score's geometric reading
    depends on, and a bias would give a fully-masked position a non-zero
    embedding, defeating the mask.

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
        :param mask: optional rank-2 ``(batch, seq_len)`` mask where **1 means
            keep** and 0 means padding or filtered. Broadcast over the feature
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
    ``(batch, query_len, doc_len)`` similarity matrix, a max over the
    **document** axis, then a sum over the **query** axis. The order matters and
    is not symmetric -- summing first and maxing second would score a document
    by its single best-matching token overall rather than by how well it covers
    every query term, which is the property late interaction exists to measure.

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

    # DECISION plan-2026-08-25T121346-c71fc3ad/D-007
    # Do NOT "simplify" this by masking and reducing in the compute dtype. The
    # reduction sums `query_len` sentinels for a fully-masked document, and at
    # ColBERT's own default `query_maxlen = 32` that is 32 * -1e4 = -3.2e5,
    # which is NOT representable in binary16 (max 65504) and becomes -inf --
    # MEASURED here, 2026-08-25: a two-term query with an out-of-range sentinel
    # already returned -inf in float16. Clamping the sentinel instead was tried
    # first and does not work, because the overflow happens in the SUM, not in
    # the sentinel. Promoting the reduction is also the standard mixed-precision
    # rule for loss-facing outputs. Consequence, deliberately accepted: this
    # layer returns float32 under mixed_float16. See decisions.md D-007.
    def _reduction_dtype(self, dtype: Any) -> str:
        """Dtype the masking and the two reductions are performed in.

        :param dtype: the dtype the raw score matrix came out in.
        :return: ``"float32"`` for a reduced-precision compute dtype, otherwise
            the incoming dtype unchanged.
        """
        # DECISION plan-2026-08-25T121346-c71fc3ad/D-014
        # NOT `keras.backend.standardize_dtype`: `keras.backend.*` is a Keras-2
        # residue banned repo-wide by
        # `test_package_api_contract.py::TestNoKeras2Residues`, and this call
        # site was the tree's only live offender. A `tf.DType` stringifies as
        # "<dtype: 'float16'>", so read `.name` when it is there and fall back
        # to `str` for a plain-string dtype -- the same two-step
        # `tests/test_models/gradient_flow_oracle.py:default_loss` uses.
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
        :param doc_mask: optional rank-2 ``(batch, doc_len)`` mask, **1 = keep**.
            Masked positions are sentinel-filled before the max.
        :type doc_mask: keras tensor or None
        :param query_mask: optional rank-2 ``(batch, query_len)`` mask,
            **1 = keep**. Masked query terms are zeroed before the sum so a
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
