"""
ColBERT, late interaction retrieval, packaged as one shared encoder.

Dense single-vector retrieval compresses a whole passage into one embedding and
scores it with a dot product. That is cheap -- one vector per document, one
approximate-nearest-neighbour probe per query -- and it is lossy in a specific
way: every term of the passage has to be averaged into the same 768 numbers, so
a query term that matches one rare word in a long document is diluted by every
other word around it. Cross-encoders sit at the other extreme. They feed the
query and the document through a transformer together, so every query term can
attend to every document term, and they are correspondingly accurate and
unusable at scale: nothing can be precomputed, and each (query, document) pair
costs a full forward pass.

Late interaction is the resolution of that tension, and it is a factorization
argument rather than a modelling trick. The encoder is run **independently** on
the query and on the document -- so every document embedding can be computed
once, offline, and stored -- and the only thing that happens at query time is a
cheap similarity between two already-computed matrices of per-token vectors:

.. math::

    S(q, d) = \\sum_{i \\in |E_q|} \\max_{j \\in |E_d|} E_{q_i} \\cdot E_{d_j}^{T}

Each query term takes its single best match in the document (the ``max``) and
those per-term bests are summed. Term-level evidence therefore survives to
scoring time, which is what the single-vector average destroys, while the
expensive interaction between the two texts never happens inside the network,
which is what the cross-encoder cannot avoid. The per-token vectors are
projected to 128 dimensions and L2-normalized, so each inner product is a
cosine and the whole score matrix is bounded -- that boundedness is what lets
padded and filtered document positions be masked with a finite sentinel rather
than ``-inf``.

**v1 and v2 are the same network.** This is a design fact of the reference
implementation, not a simplification made here: the official
``stanford-futuredata/ColBERT`` repository ships a single
``colbert/modeling/colbert.py`` for both papers, and v1 behaviour is v2's code
with ``use_ib_negatives=False``, ``nway=2``, no distillation scores and no
residual compression applied at indexing time
(https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/modeling/colbert.py).
ColBERTv2's own architecture section restates v1's encoder and the identical
MaxSim formula. What v2 changed is the *supervision* -- cross-encoder KL
distillation over ``nway`` tuples, plus in-batch negatives -- and it added an
index-time residual codec that never appears in ``call()`` and never appears in
any loss. Accordingly this module exports one :class:`ColBERT` class and two
genuine factories, :func:`create_colbert_v1` and :func:`create_colbert_v2`,
which build the same architecture and differ in the training recipe they are
documented to pair with. They are not aliases and they are not two networks;
the distinction lives in the trainers and in the v2-only codec.

Deliberate deviations from the reference, stated here rather than implied away:

- **Markers.** The reference marks queries and documents with ``[Q]`` / ``[D]``,
  implemented as BERT WordPiece's ``[unused0]`` / ``[unused1]`` slots. There is
  no WordPiece tokenizer in this library; the NLP stack is Tiktoken
  ``cl100k_base``, so :class:`~.tokenization.ColBERTTokenizer` allocates the two
  markers from free slots at the top of that vocabulary. Token-id parity with
  published ColBERT checkpoints is therefore permanently forfeited.
- **Backbone activation.** This library's :class:`~...bert.model.BERT` defaults
  to ``gelu_tanh``, the tanh approximation, not the exact erf form.
- **No pretrained weights.** None ship for ColBERT and none ship for the BERT
  backbone either; ``from_variant(pretrained=True)`` raises
  ``NotImplementedError`` on both. Every number produced by training this model
  is consequently a **wiring result** -- evidence that gradients flow and the
  loss decreases -- and never a retrieval-quality claim.

References:
    - Khattab & Zaharia, 2020. ColBERT: Efficient and Effective Passage Search
      via Contextualized Late Interaction over BERT.
      (https://arxiv.org/abs/2004.12832)
    - Santhanam et al., 2021. ColBERTv2: Effective and Efficient Retrieval via
      Lightweight Late Interaction. (https://arxiv.org/abs/2112.01488)
    - Devlin et al., 2018. BERT: Pre-training of Deep Bidirectional
      Transformers for Language Understanding.
      (https://arxiv.org/abs/1810.04805)
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.models.language.bert.model import BERT

from .components import (
    DEFAULT_MAXSIM_MASK_VALUE,
    ColBERTProjection,
    MaxSimScorer,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

#: Shortest usable query/document length: ``[CLS] [Q|D] <one token> [SEP]``.
MINIMUM_SEQUENCE_LENGTH = 4

#: Input keys :meth:`ColBERT.call` reads.
QUERY_INPUT_IDS_KEY = "query_input_ids"
QUERY_ATTENTION_MASK_KEY = "query_attention_mask"
DOC_INPUT_IDS_KEY = "doc_input_ids"
DOC_ATTENTION_MASK_KEY = "doc_attention_mask"
DOC_SKIPLIST_MASK_KEY = "doc_skiplist_mask"

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.colbert.model")
class ColBERT(keras.Model):
    """Late-interaction retrieval encoder: BERT, a shared projection, MaxSim.

    The model owns exactly three sub-components: one :class:`BERT` encoder, one
    :class:`~.components.ColBERTProjection` and one
    :class:`~.components.MaxSimScorer`. The projection instance is shared by the
    query path and the document path -- it is the *same object*, not two layers
    with the same configuration. Two instances would be two weight matrices and
    the model would no longer be the single shared encoder the architecture
    specifies; a test asserts the identity with ``is``.

    Queries and documents differ only in how they are prepared, never in which
    weights they traverse:

    * a query is padded to ``query_maxlen`` with ``[MASK]`` tokens which
      participate in scoring (query augmentation);
    * a document additionally carries a punctuation ``skiplist_mask``, so the
      participation mask on the document path is
      ``attention_mask * skiplist_mask``.

    Both preparations are the tokenizer's job
    (:class:`~.tokenization.ColBERTTokenizer`); this class consumes the arrays
    it emits.

    :param vocab_size: Backbone vocabulary size.
    :type vocab_size: int
    :param hidden_size: Backbone hidden width.
    :type hidden_size: int
    :param num_layers: Number of backbone transformer blocks.
    :type num_layers: int
    :param num_heads: Number of backbone attention heads.
    :type num_heads: int
    :param intermediate_size: Backbone feed-forward width.
    :type intermediate_size: int
    :param dim: Retrieval embedding width the projection maps to. Defaults to
        ``128``, the reference default.
    :type dim: int
    :param query_maxlen: Fixed query length the tokenizer pads to. Defaults to
        ``32``, the reference default.
    :type query_maxlen: int
    :param doc_maxlen: Maximum document length. Defaults to ``220``, the
        reference default.
    :type doc_maxlen: int
    :param max_position_embeddings: Backbone position-table size. Must be at
        least ``max(query_maxlen, doc_maxlen)``.
    :type max_position_embeddings: int
    :param hidden_dropout_rate: Backbone hidden/embedding dropout.
    :type hidden_dropout_rate: float
    :param attention_probs_dropout_rate: Backbone attention dropout.
    :type attention_probs_dropout_rate: float
    :param mask_value: Finite negative sentinel written over masked document
        positions before the max-reduce.
    :type mask_value: float
    :param kwargs: Forwarded to :class:`keras.Model`.
    :raises ValueError: If any geometry argument is non-positive, if
        ``query_maxlen`` or ``doc_maxlen`` is shorter than
        ``MINIMUM_SEQUENCE_LENGTH``, or if either exceeds
        ``max_position_embeddings``.

    Example:
        .. code-block:: python

            model = ColBERT.from_variant("tiny")
            outputs = model({
                "query_input_ids": q_ids,
                "query_attention_mask": q_mask,
                "doc_input_ids": d_ids,
                "doc_attention_mask": d_mask,
                "doc_skiplist_mask": d_skip,
            })
            outputs["score"].shape  # (batch,)
    """

    # Variant table. TWO PROVENANCE CLASSES, deliberately not blended
    # (guide v3 §13.2):
    #
    #   * The ColBERT-side numbers -- `dim=128`, `query_maxlen=32`,
    #     `doc_maxlen=220` -- are CLASS A: the official reference defaults, read
    #     from `colbert/infra/config/settings.py` in
    #     https://github.com/stanford-futuredata/ColBERT
    #     (`dim: int = 128`, `query_maxlen: int = 32`, `doc_maxlen: int = 220`).
    #     They are identical in every row because ColBERT does not scale them
    #     with the backbone; the reference ships one value for all sizes.
    #
    #   * The backbone geometry (`hidden_size`, `num_layers`, `num_heads`,
    #     `intermediate_size`) is NOT from any ColBERT source. It is copied from
    #     THIS repository's own `BERT.MODEL_VARIANTS`, which tracks the Turc et
    #     al. 2019 released checkpoints. Published ColBERT uses `bert-base` only;
    #     `tiny`/`small`/`large` are this library's capacity ladder, offered for
    #     cheap tests and for scaling, and no ColBERT paper reports them.
    #
    # Do NOT "unify" the two by citing one source for the whole table.
    MODEL_VARIANTS = {
        "large": {
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "intermediate_size": 4096,
            "dim": 128,
            "query_maxlen": 32,
            "doc_maxlen": 220,
            "description": (
                "ColBERT-Large: BERT-Large backbone; no ColBERT paper reports "
                "this size"
            ),
        },
        "base": {
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "intermediate_size": 3072,
            "dim": 128,
            "query_maxlen": 32,
            "doc_maxlen": 220,
            "description": (
                "ColBERT-Base: the published configuration's backbone size"
            ),
        },
        "small": {
            "hidden_size": 512,
            "num_layers": 6,
            "num_heads": 8,
            "intermediate_size": 2048,
            "dim": 128,
            "query_maxlen": 32,
            "doc_maxlen": 220,
            "description": "ColBERT-Small: reduced backbone for tight budgets",
        },
        "tiny": {
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 4,
            "intermediate_size": 1024,
            "dim": 128,
            "query_maxlen": 32,
            "doc_maxlen": 220,
            "description": "ColBERT-Tiny: ultra-light backbone for tests/edge",
        },
    }

    #: Reference default retrieval dimension (`settings.py`, `dim: int = 128`).
    DEFAULT_DIM = 128
    #: Reference default query length (`settings.py`, `query_maxlen: int = 32`).
    DEFAULT_QUERY_MAXLEN = 32
    #: Reference default document length (`settings.py`, `doc_maxlen: int = 220`).
    DEFAULT_DOC_MAXLEN = 220

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        dim: int = DEFAULT_DIM,
        query_maxlen: int = DEFAULT_QUERY_MAXLEN,
        doc_maxlen: int = DEFAULT_DOC_MAXLEN,
        # DECISION plan-2026-08-25T165753-704a9bcb/D-002
        # Do NOT re-add a model-side ``mask_punctuation`` parameter here (nor to
        # ``get_config``, nor to ``MODEL_VARIANTS``). It existed until 2026-08-25
        # and had no reader anywhere in the tree: the model applies whatever
        # ``doc_skiplist_mask`` it is handed, so the flag changed nothing while
        # serializing an intent nothing honored. The live flag is
        # ``ColBERTTokenizer.mask_punctuation`` (``tokenization.py:296``), which
        # gates skiplist construction; the trainer hardcodes it on the tokenizer
        # (``src/train/language/colbert/common.py:549``) and never reads a model
        # flag. Passing ``mask_punctuation=`` now raises ``ValueError`` from
        # Keras, which is the intended loud rejection.
        max_position_embeddings: int = 512,
        hidden_dropout_rate: float = 0.1,
        attention_probs_dropout_rate: float = 0.1,
        mask_value: float = DEFAULT_MAXSIM_MASK_VALUE,
        **kwargs: Any,
    ) -> None:
        """Initialize the encoder, the shared projection and the scorer."""
        super().__init__(**kwargs)

        self._validate_config(
            dim=dim,
            query_maxlen=query_maxlen,
            doc_maxlen=doc_maxlen,
            max_position_embeddings=max_position_embeddings,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.dim = dim
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_probs_dropout_rate = attention_probs_dropout_rate
        self.mask_value = float(mask_value)

        # BERT validates its own geometry (positive sizes, hidden % heads == 0)
        # and raises ValueError naming the offending value, so those checks are
        # not duplicated here.
        self.encoder = BERT(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_rate=hidden_dropout_rate,
            attention_probs_dropout_rate=attention_probs_dropout_rate,
            max_position_embeddings=max_position_embeddings,
            name="encoder",
        )

        # DECISION plan-2026-08-25T121346-c71fc3ad/D-011
        # ONE projection instance, deliberately shared by BOTH towers. Do NOT
        # give the document path its own `ColBERTProjection` -- not "for
        # symmetry", not "so each tower can be tuned". The reference builds a
        # single `nn.Linear(hidden, dim, bias=False)` and routes `query()` and
        # `doc()` through it, and the late-interaction score is only a cosine
        # between comparable spaces BECAUSE both sides land in the same one. Two
        # instances would train two independent projections into two unrelated
        # 128-d spaces while every shape, every serialization round trip and
        # every gradient test stayed green -- the score would just be
        # meaningless. Pinned by `test_the_query_and_document_share_one
        # _projection_instance`, which asserts object identity with `is` rather
        # than comparing configs. See decisions.md D-011.
        self.projection = ColBERTProjection(dim=dim, name="projection")

        self.scorer = MaxSimScorer(mask_value=self.mask_value, name="maxsim")

        logger.info(
            f"Created ColBERT: {num_layers}-layer backbone, "
            f"hidden_size={hidden_size}, dim={dim}, "
            f"query_maxlen={query_maxlen}, doc_maxlen={doc_maxlen}"
        )

    @staticmethod
    def _validate_config(
        dim: int,
        query_maxlen: int,
        doc_maxlen: int,
        max_position_embeddings: int,
    ) -> None:
        """Reject a configuration that cannot encode a real query or document.

        :param dim: Retrieval embedding width.
        :type dim: int
        :param query_maxlen: Fixed query length.
        :type query_maxlen: int
        :param doc_maxlen: Maximum document length.
        :type doc_maxlen: int
        :param max_position_embeddings: Backbone position-table size.
        :type max_position_embeddings: int
        :raises ValueError: Naming the offending value.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if max_position_embeddings <= 0:
            raise ValueError(
                "max_position_embeddings must be positive, got "
                f"{max_position_embeddings}"
            )
        for name, value in (
            ("query_maxlen", query_maxlen),
            ("doc_maxlen", doc_maxlen),
        ):
            if value < MINIMUM_SEQUENCE_LENGTH:
                raise ValueError(
                    f"{name} must be at least {MINIMUM_SEQUENCE_LENGTH} to hold "
                    f"the '[CLS] [Q|D] <token> [SEP]' frame, got {value}"
                )
            if value > max_position_embeddings:
                raise ValueError(
                    f"{name} ({value}) exceeds max_position_embeddings "
                    f"({max_position_embeddings}); the backbone has no learned "
                    "position for those slots"
                )

    # -----------------------------------------------------------------
    # build / shapes
    # -----------------------------------------------------------------

    def build(self, input_shape: Any) -> None:
        """Materialize the encoder, the projection and the scorer.

        A subclassed ``keras.Model`` that leaves ``build`` unimplemented is
        marked built while holding **zero** variables, and both
        ``model.build(shape)`` and ``.keras`` deserialization land on that
        defaulted path. Every sub-component is therefore built explicitly here.

        :param input_shape: Mapping of the shapes :meth:`call` receives. It
            must carry ``query_input_ids`` and ``doc_input_ids``; the optional
            mask entries may be omitted, since :meth:`call` resolves each of
            them to a concrete all-ones tensor.
        :type input_shape: Any
        :raises ValueError: If ``input_shape`` is not a mapping carrying both
            ``input_ids`` entries -- raised by :meth:`call` during the trace.
        """
        if self.built:
            return

        # The sub-layer tree is materialized by tracing `call()` symbolically
        # (`utils.model_build.materialize_sublayers`), NOT by a hand-written
        # chain of `sublayer.build(shape)` / `compute_output_shape(shape)`. A
        # hand-walk is a second, hand-maintained encoding of the forward
        # topology and it drifts from `call()` silently. MEASURED here,
        # 2026-08-25 on CPU: the symbolic trace succeeds on this model and
        # materializes 18 weights at a 1-layer test geometry -- the same
        # population a real forward call produces -- so the documented blocker
        # for the helper (a `call()` that reaches a raw-backend op or
        # `add_loss()`) does not apply. The float32 KerasTensor placeholders it
        # feeds `input_ids` are harmless: nothing here indexes by dtype, and no
        # weight shape depends on the ids.
        materialize_sublayers(self, input_shape)

        super().build(input_shape)

    def _resolve_id_shapes(
        self, input_shape: Any
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Extract the query and document ``input_ids`` shapes.

        :param input_shape: Mapping of shapes, or one ``(batch, seq_len)``
            tuple used for both towers.
        :type input_shape: Any
        :return: ``(query_ids_shape, doc_ids_shape)``.
        :rtype: tuple
        :raises ValueError: If a resolved shape is not rank 2.
        """
        if isinstance(input_shape, dict):
            query_shape = input_shape.get(QUERY_INPUT_IDS_KEY)
            doc_shape = input_shape.get(DOC_INPUT_IDS_KEY)
            if query_shape is None or doc_shape is None:
                raise ValueError(
                    "ColBERT.compute_output_shape expects a mapping carrying "
                    f"'{QUERY_INPUT_IDS_KEY}' and '{DOC_INPUT_IDS_KEY}'; got "
                    f"keys {sorted(input_shape)}"
                )
        else:
            query_shape = input_shape
            doc_shape = input_shape

        resolved = []
        for name, shape in (
            (QUERY_INPUT_IDS_KEY, query_shape),
            (DOC_INPUT_IDS_KEY, doc_shape),
        ):
            shape = tuple(shape)
            if len(shape) != 2:
                raise ValueError(
                    f"ColBERT.compute_output_shape expects the shape of {name}, i.e. "
                    f"(batch_size, seq_length); got {shape}"
                )
            resolved.append(shape)
        return resolved[0], resolved[1]

    def compute_output_shape(self, input_shape: Any) -> Dict[str, Any]:
        """Compute the shapes of the three returned tensors.

        :param input_shape: As accepted by :meth:`build`.
        :type input_shape: Any
        :return: Mapping with ``score``, ``query_embeddings`` and
            ``doc_embeddings``.
        :rtype: Dict[str, Any]
        """
        query_shape, doc_shape = self._resolve_id_shapes(input_shape)
        return {
            "score": (query_shape[0],),
            "query_embeddings": (query_shape[0], query_shape[1], self.dim),
            "doc_embeddings": (doc_shape[0], doc_shape[1], self.dim),
        }

    # -----------------------------------------------------------------
    # encoding
    # -----------------------------------------------------------------

    def _encode(
        self,
        input_ids: Any,
        attention_mask: Any,
        participation_mask: Optional[Any] = None,
        training: Optional[bool] = None,
    ) -> Any:
        """Run the shared encoder and the shared projection.

        The two masks are deliberately DISTINCT tensors, see the anchor below.

        :param input_ids: ``(batch, seq_len)`` integer token ids.
        :type input_ids: keras tensor
        :param attention_mask: ``(batch, seq_len)`` **padding** mask,
            **1 = visible**, and nothing else. This is what the backbone sees.
        :type attention_mask: keras tensor
        :param participation_mask: ``(batch, seq_len)`` mask, **1 = keep**,
            multiplied onto the projected pre-normalization embeddings and used
            as the MaxSim candidate set. Defaults to ``attention_mask`` when the
            caller has no extra filter to apply (the query path).
        :type participation_mask: Optional[keras tensor]
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: ``(batch, seq_len, dim)`` L2-normalized embeddings, exactly
            zero at positions ``participation_mask`` zeroes.
        """
        # DECISION plan-2026-08-25T121346-c71fc3ad/D-029
        # These two masks must stay SEPARATE. Do NOT collapse them back into one
        # tensor and pass it to both `self.encoder` and `self.projection` -- that
        # is the exact defect this signature exists to fix. Collapsing them feeds
        # the punctuation skiplist to the backbone as its ATTENTION mask, which
        # hides punctuation from every other token's self-attention; the
        # reference keeps punctuation fully visible as context and removes it
        # only from the MaxSim candidate set, by multiplying the skiplist onto
        # the projected pre-normalization embeddings. MEASURED over 40 SEEDED
        # random inits (`keras.utils.set_random_seed(0..39)`, 2-layer `tiny`, 10
        # doc positions, skiplist zeroing 3 and 7): under the collapsed ordering
        # the KEPT positions' embeddings moved purely because two OTHER
        # positions were skiplisted, by max|delta| min 0.00035694 / median
        # 0.00094578 / max 0.00241351 -- NEVER zero; under this split ordering
        # the same delta is exactly 0.0 at ALL 40 seeds. The magnitude is a
        # property of one random init, so the reproducible claim is "nonzero
        # before, exactly 0.0 after" -- not any single draw.
        #
        # The converse is equally load-bearing and equally easy to break: the
        # PADDING mask must reach `self.encoder`. Passing
        # `keras.ops.ones_like(attention_mask)` below makes the backbone attend
        # to padding as if it were content and, before 2026-08-25, reddened
        # NOTHING in the suite. Both directions are pinned now --
        # `test_a_kept_position_is_untouched_by_a_skiplist_elsewhere` and
        # `test_a_padded_document_position_cannot_influence_a_real_one`. Both
        # pre-existing skiplist guards pass under EITHER ordering, which is why
        # the collapse shipped. See decisions.md D-029 and D-030.
        if participation_mask is None:
            participation_mask = attention_mask
        encoded = self.encoder(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            training=training,
        )
        return self.projection(
            encoded["last_hidden_state"],
            mask=participation_mask,
            training=training,
        )

    def encode_query(
        self,
        inputs: Union[Any, Dict[str, Any]],
        training: Optional[bool] = None,
    ) -> Any:
        """Encode a batch of queries into per-token retrieval embeddings.

        Queries carry no punctuation skiplist: the reference passes an empty
        skiplist on this side, and the ``[MASK]`` slots that query augmentation
        writes are meant to participate in scoring.

        :param inputs: Either a ``(batch, query_len)`` id tensor, or a mapping
            with ``input_ids`` and an optional ``attention_mask``.
        :type inputs: Union[keras tensor, Dict[str, keras tensor]]
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: ``(batch, query_len, dim)`` embeddings.
        :raises ValueError: If a mapping is passed without ``input_ids``.
        """
        input_ids, attention_mask = self._unpack(inputs, "encode_query")
        return self._encode(input_ids, attention_mask, training=training)

    def encode_document(
        self,
        inputs: Union[Any, Dict[str, Any]],
        training: Optional[bool] = None,
    ) -> Any:
        """Encode a batch of documents into per-token retrieval embeddings.

        The document participation mask is ``attention_mask * skiplist_mask``
        -- both 1 = keep -- so a punctuation position is zeroed exactly like a
        padding position, before the projection normalizes. A zeroed position
        has an all-zero embedding and therefore an inner product of exactly 0
        with every query term. The backbone, however, still receives the plain
        ``attention_mask``: a skiplisted punctuation token stays fully visible
        as attention CONTEXT to its neighbours, as in the reference. See D-029.

        :param inputs: Either a ``(batch, doc_len)`` id tensor, or a mapping
            with ``input_ids`` and optional ``attention_mask`` /
            ``skiplist_mask``.
        :type inputs: Union[keras tensor, Dict[str, keras tensor]]
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: ``(batch, doc_len, dim)`` embeddings.
        :raises ValueError: If a mapping is passed without ``input_ids``.
        """
        input_ids, attention_mask = self._unpack(inputs, "encode_document")
        skiplist_mask = (
            inputs.get("skiplist_mask") if isinstance(inputs, dict) else None
        )
        participation = self._participation(attention_mask, skiplist_mask)
        return self._encode(
            input_ids,
            attention_mask,
            participation_mask=participation,
            training=training,
        )

    @staticmethod
    def _participation(attention_mask: Any, skiplist_mask: Optional[Any]) -> Any:
        """Combine the attention mask and the punctuation skiplist.

        :param attention_mask: ``(batch, seq_len)`` mask, 1 = real token.
        :type attention_mask: keras tensor
        :param skiplist_mask: ``(batch, seq_len)`` mask, 1 = content position,
            or ``None``.
        :type skiplist_mask: Optional[keras tensor]
        :return: The elementwise product, in ``attention_mask``'s dtype.
        """
        if skiplist_mask is None:
            return attention_mask
        return attention_mask * keras.ops.cast(skiplist_mask, attention_mask.dtype)

    @staticmethod
    def _unpack(inputs: Any, caller: str) -> Tuple[Any, Any]:
        """Resolve ``(input_ids, attention_mask)`` from a tensor or a mapping.

        A missing ``attention_mask`` becomes all-ones rather than being derived
        from a pad id: the backbone here treats ``pad_token_id`` as advisory and
        never derives a mask from it, and silently inventing one at this level
        would make two adjacent components disagree about what padding is.

        :param inputs: Id tensor or mapping.
        :type inputs: Any
        :param caller: Method name, for the error message.
        :type caller: str
        :return: ``(input_ids, attention_mask)``.
        :rtype: tuple
        :raises ValueError: If a mapping carries no ``input_ids``.
        """
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    f"{caller} received a mapping without an 'input_ids' key; "
                    f"got {sorted(inputs)}"
                )
            attention_mask = inputs.get("attention_mask")
        else:
            input_ids = inputs
            attention_mask = None

        if attention_mask is None:
            attention_mask = keras.ops.ones_like(input_ids)
        return input_ids, attention_mask

    def score(
        self,
        query_embeddings: Any,
        doc_embeddings: Any,
        doc_mask: Optional[Any] = None,
        query_mask: Optional[Any] = None,
    ) -> Any:
        """MaxSim-score paired query and document embeddings.

        :param query_embeddings: ``(batch, query_len, dim)``.
        :type query_embeddings: keras tensor
        :param doc_embeddings: ``(batch, doc_len, dim)``.
        :type doc_embeddings: keras tensor
        :param doc_mask: Optional ``(batch, doc_len)`` mask, 1 = keep. Masked
            positions are sentinel-filled before the max-reduce.
        :type doc_mask: Optional[keras tensor]
        :param query_mask: Optional ``(batch, query_len)`` mask, 1 = keep.
        :type query_mask: Optional[keras tensor]
        :return: ``(batch,)`` scores.
        """
        return self.scorer(
            query_embeddings,
            doc_embeddings,
            doc_mask=doc_mask,
            query_mask=query_mask,
        )

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def call(
        self,
        inputs: Dict[str, Any],
        training: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Encode both towers and score each pair.

        :param inputs: Mapping with ``query_input_ids`` and ``doc_input_ids``,
            plus optional ``query_attention_mask``, ``doc_attention_mask`` and
            ``doc_skiplist_mask``.
        :type inputs: Dict[str, keras tensor]
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: Mapping with a **fixed** key set: ``score`` ``(batch,)``,
            ``query_embeddings`` ``(batch, query_len, dim)`` and
            ``doc_embeddings`` ``(batch, doc_len, dim)``.
        :rtype: Dict[str, keras tensor]
        :raises ValueError: If ``inputs`` is not a mapping, or lacks either
            ``input_ids`` entry.
        """
        if not isinstance(inputs, dict):
            raise ValueError(
                "ColBERT.call expects a mapping with "
                f"'{QUERY_INPUT_IDS_KEY}' and '{DOC_INPUT_IDS_KEY}'; got "
                f"{type(inputs).__name__}"
            )
        for key in (QUERY_INPUT_IDS_KEY, DOC_INPUT_IDS_KEY):
            if inputs.get(key) is None:
                raise ValueError(
                    f"ColBERT.call requires '{key}'; got keys {sorted(inputs)}"
                )

        query_ids = inputs[QUERY_INPUT_IDS_KEY]
        doc_ids = inputs[DOC_INPUT_IDS_KEY]

        query_mask = inputs.get(QUERY_ATTENTION_MASK_KEY)
        if query_mask is None:
            query_mask = keras.ops.ones_like(query_ids)

        doc_mask = inputs.get(DOC_ATTENTION_MASK_KEY)
        if doc_mask is None:
            doc_mask = keras.ops.ones_like(doc_ids)
        doc_participation = self._participation(
            doc_mask, inputs.get(DOC_SKIPLIST_MASK_KEY)
        )

        query_embeddings = self._encode(query_ids, query_mask, training=training)
        doc_embeddings = self._encode(
            doc_ids,
            doc_mask,
            participation_mask=doc_participation,
            training=training,
        )

        # DECISION plan-2026-08-25T121346-c71fc3ad/D-012
        # The output STRUCTURE is fixed here and resolved ONCE, at the return.
        # Do NOT drop a key when its optional input was absent, and do NOT return
        # a bare score tensor for a mask-less call: an output structure that
        # depends on which inputs were supplied breaks `.predict()`, which
        # concatenates per-batch outputs and cannot align a slot that exists in
        # one batch and not another. `model(inputs)` keeps working either way,
        # which is exactly why the defect ships -- BERT itself shipped it, and
        # its `D-032` records the same rule for the same reason. Every optional
        # input is resolved to a concrete tensor ABOVE, so this dict is
        # unconditional. See decisions.md D-012.
        return {
            "score": self.scorer(
                query_embeddings,
                doc_embeddings,
                doc_mask=doc_participation,
                query_mask=query_mask,
            ),
            "query_embeddings": query_embeddings,
            "doc_embeddings": doc_embeddings,
        }

    # -----------------------------------------------------------------
    # variants / serialization
    # -----------------------------------------------------------------

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
    ) -> "ColBERT":
        """Create a ColBERT model from a named variant.

        :param variant: One of ``"tiny"``, ``"small"``, ``"base"``, ``"large"``.
        :type variant: str
        :param pretrained: ``False`` (default) for random initialization, or a
            path to a local ``.keras`` checkpoint. ``True`` raises: no ColBERT
            weights are distributed with this library, and none exist for the
            BERT backbone either.
        :type pretrained: Union[bool, str]
        :param kwargs: Overrides applied on top of the variant's row.
        :type kwargs: Any
        :return: A configured, randomly-initialized (or locally-loaded) model.
        :rtype: ColBERT
        :raises ValueError: If ``variant`` is not a known key.
        :raises NotImplementedError: If ``pretrained is True``.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        if pretrained is True:
            raise NotImplementedError(
                f"No pretrained ColBERT weights are distributed with "
                f"dl_techniques (requested variant '{variant}'), and no "
                f"pretrained weights exist for the BERT backbone either. Build "
                f"a random-init model with "
                f"ColBERT.from_variant('{variant}', pretrained=False) and load "
                f"your own checkpoint with model.load_weights(path)."
            )

        config = dict(cls.MODEL_VARIANTS[variant])
        description = config.pop("description", "")
        config.update(kwargs)

        logger.info(f"Creating ColBERT-{variant.upper()}: {description}")

        model = cls(**config)

        if pretrained:
            model.build(
                {
                    QUERY_INPUT_IDS_KEY: (None, model.query_maxlen),
                    DOC_INPUT_IDS_KEY: (None, model.doc_maxlen),
                }
            )
            model.load_weights(pretrained)

        return model

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument by name.

        :return: A configuration dict accepted verbatim by :meth:`from_config`.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "dim": self.dim,
                "query_maxlen": self.query_maxlen,
                "doc_maxlen": self.doc_maxlen,
                "max_position_embeddings": self.max_position_embeddings,
                "hidden_dropout_rate": self.hidden_dropout_rate,
                "attention_probs_dropout_rate": self.attention_probs_dropout_rate,
                "mask_value": self.mask_value,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ColBERT":
        """Rebuild a model from :meth:`get_config` output.

        :param config: Constructor arguments.
        :type config: Dict[str, Any]
        :return: An equivalent, unbuilt model.
        :rtype: ColBERT
        """
        return cls(**config)


# ---------------------------------------------------------------------
# Module-level factories
# ---------------------------------------------------------------------


def create_colbert(
    variant: str = "base",
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> ColBERT:
    """Build a ColBERT late-interaction encoder from a named variant.

    Version-neutral entry point. :func:`create_colbert_v1` and
    :func:`create_colbert_v2` build the same network; use them when the calling
    code wants to record which training recipe it is targeting.

    :param variant: ``"tiny"``, ``"small"``, ``"base"`` or ``"large"``.
    :type variant: str
    :param pretrained: ``False``, or a path to a local ``.keras`` checkpoint.
        ``True`` raises ``NotImplementedError``.
    :type pretrained: Union[bool, str]
    :param kwargs: Overrides forwarded to :meth:`ColBERT.from_variant`.
    :type kwargs: Any
    :return: A configured model.
    :rtype: ColBERT
    :raises ValueError: If ``variant`` is unknown.
    :raises NotImplementedError: If ``pretrained is True``.

    Example:
        >>> model = create_colbert("tiny", vocab_size=512)
    """
    return ColBERT.from_variant(variant, pretrained=pretrained, **kwargs)


def create_colbert_v1(
    variant: str = "base",
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> ColBERT:
    """Build a ColBERT **v1** encoder, for the pairwise-softmax recipe.

    Pairs with ``ColBERTPairwiseSoftmaxLoss`` and
    ``src/train/language/colbert/train_colbert_v1.py``: softmax cross-entropy
    over the ``nway`` candidate scores of a ``<query, positive, negatives...>``
    tuple, positive first, ``nway=2`` in the original recipe.

    **This builds exactly the same network as** :func:`create_colbert_v2` --
    same class, same weights, same ``MODEL_VARIANTS`` row. That is not a
    shortcut taken here: the official ``stanford-futuredata/ColBERT``
    repository has no v1-only code path, and v1 behaviour is v2's code with
    ``use_ib_negatives=False``, ``nway=2``, no distillation scores and no
    residual compression. The v1/v2 distinction is the training recipe and the
    v2-only index-time codec, and a test asserts the two factories produce
    identical weight-path sets.

    :param variant: ``"tiny"``, ``"small"``, ``"base"`` or ``"large"``.
    :type variant: str
    :param pretrained: ``False``, or a path to a local ``.keras`` checkpoint.
        ``True`` raises ``NotImplementedError`` -- no ColBERT weights and no
        backbone weights ship here, so any trained result is a wiring result.
    :type pretrained: Union[bool, str]
    :param kwargs: Overrides forwarded to :meth:`ColBERT.from_variant`.
    :type kwargs: Any
    :return: A configured model.
    :rtype: ColBERT
    :raises ValueError: If ``variant`` is unknown.
    :raises NotImplementedError: If ``pretrained is True``.

    Example:
        >>> model = create_colbert_v1("tiny", vocab_size=512)
    """
    return ColBERT.from_variant(variant, pretrained=pretrained, **kwargs)


def create_colbert_v2(
    variant: str = "base",
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> ColBERT:
    """Build a ColBERT **v2** encoder, for the distillation recipe.

    Pairs with ``ColBERTDistillationLoss`` and
    ``src/train/language/colbert/train_colbert_v2.py``: KL divergence between
    ``log_softmax`` of the student scores and ``log_softmax`` of
    ``distillation_alpha``-scaled cross-encoder teacher scores over ``nway``
    (typically 64) candidates. Index-time residual compression
    (``compression.ResidualCompressionCodec``) is the other v2 addition; it is
    never part of the forward pass or of any loss.

    **This builds exactly the same network as** :func:`create_colbert_v1` --
    see that docstring for the citation. v2 changed the supervision and the
    index, not the encoder.

    :param variant: ``"tiny"``, ``"small"``, ``"base"`` or ``"large"``.
    :type variant: str
    :param pretrained: ``False``, or a path to a local ``.keras`` checkpoint.
        ``True`` raises ``NotImplementedError`` -- no ColBERT weights and no
        backbone weights ship here, so any trained result is a wiring result.
    :type pretrained: Union[bool, str]
    :param kwargs: Overrides forwarded to :meth:`ColBERT.from_variant`.
    :type kwargs: Any
    :return: A configured model.
    :rtype: ColBERT
    :raises ValueError: If ``variant`` is unknown.
    :raises NotImplementedError: If ``pretrained is True``.

    Example:
        >>> model = create_colbert_v2("tiny", vocab_size=512)
    """
    return ColBERT.from_variant(variant, pretrained=pretrained, **kwargs)
